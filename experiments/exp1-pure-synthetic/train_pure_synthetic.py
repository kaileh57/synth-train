#!/usr/bin/env python3
"""
Main training script for Experiment 1: Pure Synthetic Excellence
Trains GPT-2 style models on OpenHermes-2.5 dataset
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
import logging

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    set_seed
)
from accelerate import Accelerator
import wandb

# Import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_configs import create_model_config, get_training_config, print_model_info
from evaluate import evaluate_model
from generate_samples import generate_samples


# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class ExperimentConfig:
    """Configuration for the experiment"""
    def __init__(self, args):
        self.model_size = args.model_size
        self.dataset_path = args.dataset_path
        self.output_dir = args.output_dir
        self.run_name = args.run_name or f"exp1-{self.model_size}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.seed = args.seed
        self.num_train_epochs = args.num_train_epochs
        self.eval_steps = args.eval_steps
        self.save_steps = args.save_steps
        self.logging_steps = args.logging_steps
        self.warmup_steps = args.warmup_steps
        self.gradient_checkpointing = args.gradient_checkpointing
        self.fp16 = args.fp16
        self.use_wandb = args.use_wandb
        self.push_to_hub = args.push_to_hub
        self.hub_model_id = args.hub_model_id
        self.resume_from_checkpoint = args.resume_from_checkpoint
        
        # Create output directory
        self.output_dir = Path(self.output_dir) / self.run_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self.save()
    
    def save(self):
        """Save configuration to file"""
        config_path = self.output_dir / "experiment_config.json"
        with open(config_path, "w") as f:
            json.dump(self.__dict__, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path):
        """Load configuration from file"""
        with open(path, "r") as f:
            config_dict = json.load(f)
        # Create dummy args object
        args = argparse.Namespace(**config_dict)
        return cls(args)


def setup_wandb(config: ExperimentConfig):
    """Initialize Weights & Biases tracking"""
    if config.use_wandb:
        wandb.init(
            project="synth-train-exp1",
            name=config.run_name,
            config={
                "model_size": config.model_size,
                "dataset": "OpenHermes-2.5",
                "epochs": config.num_train_epochs,
                "seed": config.seed,
            }
        )


def load_tokenizer_and_data(config: ExperimentConfig):
    """Load tokenizer and datasets"""
    logger.info(f"Loading data from {config.dataset_path}")
    
    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load datasets
    datasets = load_from_disk(config.dataset_path)
    
    # Verify datasets
    logger.info(f"Train samples: {len(datasets['train'])}")
    logger.info(f"Validation samples: {len(datasets['validation'])}")
    
    return tokenizer, datasets


def create_model(config: ExperimentConfig):
    """Create and initialize model"""
    logger.info(f"Creating {config.model_size} model")
    
    # Create model config
    model_config = create_model_config(
        model_size=config.model_size,
        gradient_checkpointing=config.gradient_checkpointing
    )
    
    # Print model info
    print_model_info(model_config, config.model_size)
    
    # Create model
    model = GPT2LMHeadModel(model_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    return model, model_config


def get_training_args(config: ExperimentConfig, train_config: dict):
    """Create training arguments"""
    
    # Calculate total steps
    # This is approximate - actual steps depend on data loading
    steps_per_epoch = 100000  # Rough estimate, will be updated
    total_steps = steps_per_epoch * config.num_train_epochs
    
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=train_config["per_device_train_batch_size"],
        per_device_eval_batch_size=train_config["per_device_train_batch_size"] * 2,
        gradient_accumulation_steps=train_config["gradient_accumulation_steps"],
        eval_accumulation_steps=4,
        learning_rate=train_config["learning_rate"],
        warmup_steps=config.warmup_steps,
        weight_decay=train_config["weight_decay"],
        adam_beta1=train_config["adam_beta1"],
        adam_beta2=train_config["adam_beta2"],
        adam_epsilon=train_config["adam_epsilon"],
        max_grad_norm=train_config["gradient_clip"],
        fp16=config.fp16,
        fp16_opt_level="O2",
        logging_dir=str(config.output_dir / "logs"),
        logging_steps=config.logging_steps,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=["wandb"] if config.use_wandb else ["tensorboard"],
        run_name=config.run_name,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        hub_strategy="every_save",
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        seed=config.seed,
        data_seed=config.seed,
    )
    
    return training_args


def train_model(
    model,
    tokenizer,
    datasets,
    training_args,
    config: ExperimentConfig
):
    """Train the model"""
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 uses CLM, not MLM
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Train
    logger.info("Starting training...")
    
    if config.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {config.resume_from_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    else:
        train_result = trainer.train()
    
    # Save final model
    logger.info("Saving final model...")
    trainer.save_model()
    
    # Save training results
    with open(config.output_dir / "train_results.json", "w") as f:
        json.dump(train_result.metrics, f, indent=2)
    
    return trainer, train_result


def run_final_evaluation(
    model,
    tokenizer,
    datasets,
    config: ExperimentConfig
):
    """Run comprehensive final evaluation"""
    logger.info("Running final evaluation...")
    
    # Evaluate on validation set
    eval_results = evaluate_model(
        model,
        tokenizer,
        datasets["validation"],
        output_dir=config.output_dir,
        num_samples=1000
    )
    
    # Generate samples
    logger.info("Generating sample outputs...")
    generation_results = generate_samples(
        model,
        tokenizer,
        num_samples=100,
        output_dir=config.output_dir
    )
    
    # Combine results
    final_results = {
        "evaluation": eval_results,
        "generation": generation_results,
        "model_size": config.model_size,
        "training_epochs": config.num_train_epochs,
    }
    
    # Save final results
    with open(config.output_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 model on OpenHermes-2.5")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="500M",
                        choices=["125M", "350M", "500M", "1B", "1.5B"],
                        help="Model size to train")
    
    # Data arguments
    parser.add_argument("--dataset_path", type=str, default="./data/openhermes_processed",
                        help="Path to processed dataset")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./models",
                        help="Output directory for model and logs")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this run")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--eval_steps", type=int, default=1000,
                        help="Evaluate every N steps")
    parser.add_argument("--save_steps", type=int, default=5000,
                        help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log every N steps")
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps")
    
    # Optimization arguments
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Use gradient checkpointing to save memory")
    parser.add_argument("--fp16", action="store_true",
                        help="Use mixed precision training")
    
    # Other arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Use Weights & Biases for logging")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to HuggingFace Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="HuggingFace Hub model ID")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create experiment config
    config = ExperimentConfig(args)
    
    # Setup wandb if requested
    setup_wandb(config)
    
    # Load tokenizer and data
    tokenizer, datasets = load_tokenizer_and_data(config)
    
    # Create model
    model, model_config = create_model(config)
    
    # Get training configuration
    train_config = get_training_config(config.model_size)
    
    # Create training arguments
    training_args = get_training_args(config, train_config)
    
    # Train model
    trainer, train_result = train_model(
        model,
        tokenizer,
        datasets,
        training_args,
        config
    )
    
    # Run final evaluation
    final_results = run_final_evaluation(
        trainer.model,
        tokenizer,
        datasets,
        config
    )
    
    # Log final results
    logger.info("Training complete!")
    logger.info(f"Final validation loss: {final_results['evaluation']['perplexity']:.3f}")
    logger.info(f"Results saved to: {config.output_dir}")
    
    # Finish wandb run
    if config.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main() 