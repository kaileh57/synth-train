#!/usr/bin/env python3
"""
Main training script for quality vs quantity experiments
Trains models on different quality/quantity subsets and compares results
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
import wandb

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import from exp1-pure-synthetic
exp1_path = parent_dir / "exp1-pure-synthetic"
sys.path.insert(0, str(exp1_path))
from model_configs import create_model_config, print_model_info
from evaluate import evaluate_model

# Import local modules
from training_configs import get_config_by_name, estimate_training_time, get_lr_schedule
from evaluate_quality_tradeoff import QualityQuantityEvaluator
from generation_analysis import analyze_generation_quality

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityExperimentTrainer:
    """Trainer for quality vs quantity experiments"""
    
    def __init__(
        self,
        experiment_name: str,
        output_base_dir: str = "./results",
        use_wandb: bool = True,
        seed: int = 42
    ):
        self.experiment_name = experiment_name
        self.config = get_config_by_name(experiment_name)
        
        if self.config is None:
            raise ValueError(f"Unknown experiment: {experiment_name}")
        
        self.output_dir = Path(output_base_dir) / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_wandb = use_wandb
        self.seed = seed
        
        # Set seed
        set_seed(seed)
        
        # Save experiment config
        self._save_config()
    
    def _save_config(self):
        """Save experiment configuration"""
        config_path = self.output_dir / "experiment_config.json"
        config_dict = self.config.to_dict()
        config_dict["seed"] = self.seed
        config_dict["timestamp"] = datetime.now().isoformat()
        
        with open(config_path, "w") as f:
            json.dump(config_dict, f, indent=2)
    
    def setup_wandb(self):
        """Initialize Weights & Biases"""
        if self.use_wandb:
            wandb.init(
                project="synth-train-exp2",
                name=self.experiment_name,
                config=self.config.to_dict(),
                tags=[
                    f"model_{self.config.model_size}",
                    f"samples_{self.config.num_train_samples}",
                    "quality_vs_quantity"
                ]
            )
    
    def load_data_and_tokenizer(self) -> tuple:
        """Load dataset and tokenizer"""
        logger.info(f"Loading dataset from {self.config.dataset_path}")
        
        # Load tokenizer
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Load dataset
        dataset = load_from_disk(self.config.dataset_path)
        
        # Verify dataset
        logger.info(f"Train samples: {len(dataset['train'])}")
        logger.info(f"Validation samples: {len(dataset['test'])}")
        
        # If we need to subsample (for ablation studies)
        if self.config.num_train_samples < len(dataset['train']):
            logger.info(f"Subsampling to {self.config.num_train_samples} samples")
            dataset['train'] = dataset['train'].select(range(self.config.num_train_samples))
        
        return tokenizer, dataset
    
    def preprocess_dataset(self, dataset: DatasetDict, tokenizer) -> DatasetDict:
        """Preprocess dataset for training"""
        
        def format_example(example):
            """Format instruction-response into training text"""
            if "conversations" in example:
                # Handle conversation format
                text = ""
                for turn in example["conversations"]:
                    role = turn.get("from", "")
                    content = turn.get("value", "")
                    if role == "system":
                        text += f"System: {content}\n\n"
                    elif role == "human":
                        text += f"### Instruction:\n{content}\n\n"
                    elif role == "gpt":
                        text += f"### Response:\n{content}\n\n"
                return {"text": text.strip()}
            else:
                # Handle simple format
                instruction = example.get("instruction", "")
                response = example.get("output", example.get("response", ""))
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
                return {"text": text}
        
        # Format dataset
        dataset = dataset.map(format_example)
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=2048,
                return_tensors="pt"
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=dataset["train"].column_names,
            desc="Tokenizing"
        )
        
        # Add labels
        def add_labels(examples):
            examples["labels"] = examples["input_ids"].copy()
            return examples
        
        tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
        
        return tokenized_dataset
    
    def create_model(self):
        """Create model based on configuration"""
        logger.info(f"Creating {self.config.model_size} model")
        
        model_config = create_model_config(
            model_size=self.config.model_size,
            gradient_checkpointing=self.config.gradient_checkpointing
        )
        
        print_model_info(model_config, self.config.model_size)
        
        model = GPT2LMHeadModel(model_config)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Total parameters: {total_params:,}")
        
        return model
    
    def get_training_args(self) -> TrainingArguments:
        """Create training arguments"""
        
        # Get learning rate schedule
        lr_schedule = get_lr_schedule(self.config.num_train_samples)
        
        # Calculate steps
        steps_per_epoch = self.config.num_train_samples // self.config.get_effective_batch_size()
        max_steps = int(steps_per_epoch * self.config.max_epochs)
        
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.max_epochs,
            max_steps=max_steps if self.config.training_steps is None else self.config.training_steps,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            eval_accumulation_steps=4,
            learning_rate=self.config.learning_rate,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            lr_scheduler_type=lr_schedule["scheduler_type"],
            fp16=self.config.fp16,
            fp16_opt_level="O2",
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=50,
            evaluation_strategy="steps",
            eval_steps=self.config.eval_steps,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=["wandb"] if self.use_wandb else ["tensorboard"],
            run_name=self.experiment_name,
            gradient_checkpointing=self.config.gradient_checkpointing,
            remove_unused_columns=False,
            dataloader_num_workers=4,
            seed=self.seed,
            data_seed=self.seed,
        )
        
        return training_args
    
    def train(self, resume_from_checkpoint: Optional[str] = None):
        """Run the training"""
        
        # Setup wandb
        self.setup_wandb()
        
        # Load data and tokenizer
        tokenizer, dataset = self.load_data_and_tokenizer()
        
        # Preprocess dataset
        tokenized_dataset = self.preprocess_dataset(dataset, tokenizer)
        
        # Create model
        model = self.create_model()
        
        # Get training args
        training_args = self.get_training_args()
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8 if training_args.fp16 else None
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["test"],
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        
        # Log initial metrics
        if hasattr(dataset["train"], "features") and "quality_score" in dataset["train"].features:
            quality_scores = dataset["train"]["quality_score"]
            avg_quality = sum(quality_scores) / len(quality_scores)
            logger.info(f"Average quality score of training data: {avg_quality:.3f}")
            if self.use_wandb:
                wandb.log({"avg_train_quality": avg_quality})
        
        # Train
        logger.info("Starting training...")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save model
        logger.info("Saving model...")
        trainer.save_model()
        tokenizer.save_pretrained(self.output_dir)
        
        # Save training results
        with open(self.output_dir / "train_results.json", "w") as f:
            json.dump(train_result.metrics, f, indent=2)
        
        # Evaluate
        logger.info("Running evaluation...")
        self.evaluate(trainer.model, tokenizer, tokenized_dataset["test"])
        
        # Finish wandb
        if self.use_wandb:
            wandb.finish()
        
        return trainer.model, train_result
    
    def evaluate(self, model, tokenizer, eval_dataset):
        """Run comprehensive evaluation"""
        
        # Standard evaluation
        eval_results = evaluate_model(
            model,
            tokenizer,
            eval_dataset,
            output_dir=self.output_dir / "evaluation",
            num_samples=min(1000, len(eval_dataset))
        )
        
        # Quality-specific analysis
        evaluator = QualityQuantityEvaluator(model, tokenizer)
        quality_results = evaluator.evaluate_quality_impact(
            eval_dataset,
            output_dir=self.output_dir / "quality_analysis"
        )
        
        # Generation analysis
        gen_results = analyze_generation_quality(
            model,
            tokenizer,
            num_samples=100,
            output_dir=self.output_dir / "generation_analysis"
        )
        
        # Combine results
        combined_results = {
            "standard_metrics": eval_results,
            "quality_analysis": quality_results,
            "generation_analysis": gen_results,
            "experiment_config": self.config.to_dict()
        }
        
        # Save combined results
        with open(self.output_dir / "evaluation_summary.json", "w") as f:
            json.dump(combined_results, f, indent=2)
        
        # Log to wandb
        if self.use_wandb:
            wandb.log(combined_results["standard_metrics"])
            wandb.log({"quality_analysis": quality_results})
        
        return combined_results


def run_all_experiments(
    experiments: List[str],
    output_dir: str = "./results",
    use_wandb: bool = True,
    skip_existing: bool = True
):
    """Run multiple experiments in sequence"""
    
    results = {}
    
    for exp_name in experiments:
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting experiment: {exp_name}")
        logger.info(f"{'='*80}\n")
        
        # Check if already exists
        exp_output = Path(output_dir) / exp_name
        if skip_existing and (exp_output / "train_results.json").exists():
            logger.info(f"Skipping {exp_name} - already exists")
            continue
        
        # Estimate time
        config = get_config_by_name(exp_name)
        time_est = estimate_training_time(config)
        logger.info(f"Estimated training time: {time_est['training_hours']:.2f} hours")
        
        # Run experiment
        try:
            trainer = QualityExperimentTrainer(
                exp_name,
                output_base_dir=output_dir,
                use_wandb=use_wandb
            )
            
            model, train_result = trainer.train()
            results[exp_name] = {
                "status": "completed",
                "metrics": train_result.metrics
            }
            
        except Exception as e:
            logger.error(f"Error in experiment {exp_name}: {str(e)}")
            results[exp_name] = {
                "status": "failed",
                "error": str(e)
            }
    
    # Save summary
    summary_path = Path(output_dir) / "experiment_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Train quality vs quantity experiments")
    parser.add_argument("experiment", type=str, help="Experiment name or 'all'")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Output directory")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable Weights & Biases")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip experiments that already have results")
    
    args = parser.parse_args()
    
    if args.experiment == "all":
        # Run all main experiments
        from training_configs import EXPERIMENTS
        experiments = list(EXPERIMENTS.keys())
        
        results = run_all_experiments(
            experiments,
            output_dir=args.output_dir,
            use_wandb=not args.no_wandb,
            skip_existing=args.skip_existing
        )
        
        # Print summary
        print("\n" + "="*80)
        print("EXPERIMENT SUMMARY")
        print("="*80)
        for exp, result in results.items():
            status = result["status"]
            print(f"{exp:30} | {status:10}")
        
    else:
        # Run single experiment
        trainer = QualityExperimentTrainer(
            args.experiment,
            output_base_dir=args.output_dir,
            use_wandb=not args.no_wandb
        )
        
        model, train_result = trainer.train(resume_from_checkpoint=args.resume_from)
        
        print(f"\nTraining complete for {args.experiment}")
        print(f"Results saved to: {trainer.output_dir}")


if __name__ == "__main__":
    main() 