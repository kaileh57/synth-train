#!/usr/bin/env python3
"""
Evaluation utilities for Experiment 1: Pure Synthetic Excellence
Provides comprehensive evaluation metrics for trained models
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    DataCollatorForLanguageModeling
)
from tqdm import tqdm
from rouge_score import rouge_scorer
import evaluate

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for evaluation"""
    batch_size: int = 8
    max_length: int = 2048
    num_samples: int = 1000
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    compute_rouge: bool = True
    compute_bleu: bool = True
    compute_diversity: bool = True
    save_predictions: bool = True


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: GPT2TokenizerFast,
        config: Optional[EvaluationConfig] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or EvaluationConfig()
        
        # Move model to device
        self.model.to(self.config.device)
        self.model.eval()
        
        # Initialize metrics
        self.perplexity_metric = evaluate.load("perplexity", module_type="metric")
        if self.config.compute_rouge:
            self.rouge_scorer = rouge_scorer.RougeScorer(
                ['rouge1', 'rouge2', 'rougeL'], use_stemmer=True
            )
        if self.config.compute_bleu:
            self.bleu_metric = evaluate.load("bleu")
    
    def compute_perplexity(self, dataset: Dataset) -> float:
        """Compute perplexity on dataset"""
        logger.info("Computing perplexity...")
        
        # Create data loader
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            collate_fn=data_collator,
            shuffle=False
        )
        
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Computing perplexity"):
                # Move batch to device
                batch = {k: v.to(self.config.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Calculate loss
                loss = outputs.loss
                
                # Count non-padding tokens
                attention_mask = batch.get("attention_mask", torch.ones_like(batch["input_ids"]))
                num_tokens = attention_mask.sum().item()
                
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)
        
        return perplexity
    
    def evaluate_generation_quality(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None
    ) -> Dict[str, float]:
        """Evaluate generation quality on instruction-following tasks"""
        logger.info("Evaluating generation quality...")
        
        num_samples = num_samples or self.config.num_samples
        samples = dataset.select(range(min(num_samples, len(dataset))))
        
        results = {
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "length_ratio": [],
            "diversity_scores": []
        }
        
        predictions = []
        references = []
        
        for idx, sample in enumerate(tqdm(samples, desc="Generating responses")):
            # Extract instruction and reference response
            text = self.tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
            
            # Find instruction and response split
            instruction_marker = "### Instruction:"
            response_marker = "### Response:"
            
            if instruction_marker in text and response_marker in text:
                parts = text.split(response_marker)
                instruction_part = parts[0]
                reference_response = parts[1].strip() if len(parts) > 1 else ""
                
                # Generate response
                prompt = instruction_part + response_marker + "\n"
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.config.max_length // 2
                ).to(self.config.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
                
                predictions.append(generated_text)
                references.append(reference_response)
                
                # Compute ROUGE scores
                if self.config.compute_rouge and reference_response:
                    scores = self.rouge_scorer.score(reference_response, generated_text)
                    results["rouge1"].append(scores["rouge1"].fmeasure)
                    results["rouge2"].append(scores["rouge2"].fmeasure)
                    results["rougeL"].append(scores["rougeL"].fmeasure)
                
                # Length ratio
                if reference_response:
                    length_ratio = len(generated_text.split()) / max(len(reference_response.split()), 1)
                    results["length_ratio"].append(length_ratio)
        
        # Compute aggregate metrics
        aggregated_results = {}
        
        # ROUGE scores
        if results["rouge1"]:
            aggregated_results["rouge1"] = np.mean(results["rouge1"])
            aggregated_results["rouge2"] = np.mean(results["rouge2"])
            aggregated_results["rougeL"] = np.mean(results["rougeL"])
        
        # BLEU score
        if self.config.compute_bleu and references and predictions:
            bleu_result = self.bleu_metric.compute(
                predictions=predictions,
                references=[[ref] for ref in references]
            )
            aggregated_results["bleu"] = bleu_result["bleu"]
        
        # Length statistics
        if results["length_ratio"]:
            aggregated_results["avg_length_ratio"] = np.mean(results["length_ratio"])
            aggregated_results["std_length_ratio"] = np.std(results["length_ratio"])
        
        # Diversity metrics
        if self.config.compute_diversity:
            diversity_scores = self.compute_diversity_metrics(predictions)
            aggregated_results.update(diversity_scores)
        
        return aggregated_results, predictions, references
    
    def compute_diversity_metrics(self, texts: List[str]) -> Dict[str, float]:
        """Compute diversity metrics for generated texts"""
        
        # N-gram diversity
        diversity_scores = {}
        
        for n in [1, 2, 3, 4]:
            all_ngrams = []
            for text in texts:
                tokens = text.split()
                ngrams = [' '.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
                all_ngrams.extend(ngrams)
            
            if all_ngrams:
                unique_ngrams = len(set(all_ngrams))
                total_ngrams = len(all_ngrams)
                diversity_scores[f"distinct_{n}"] = unique_ngrams / total_ngrams
        
        # Self-BLEU (diversity between generated samples)
        if len(texts) > 10:
            self_bleu_scores = []
            sample_size = min(len(texts), 100)
            sampled_texts = np.random.choice(texts, sample_size, replace=False)
            
            for i, text in enumerate(sampled_texts[:50]):  # Limit computation
                other_texts = [t for j, t in enumerate(sampled_texts) if j != i]
                
                try:
                    bleu_result = self.bleu_metric.compute(
                        predictions=[text],
                        references=[other_texts[:10]]  # Compare with 10 others
                    )
                    self_bleu_scores.append(bleu_result["bleu"])
                except:
                    pass
            
            if self_bleu_scores:
                diversity_scores["self_bleu"] = 1 - np.mean(self_bleu_scores)
        
        return diversity_scores
    
    def evaluate_few_shot_tasks(self, tasks: Optional[List[Dict]] = None) -> Dict[str, float]:
        """Evaluate on few-shot tasks"""
        if tasks is None:
            # Default few-shot tasks
            tasks = [
                {
                    "name": "simple_qa",
                    "prompt": "Q: What is the capital of France?\nA:",
                    "expected": ["Paris", "paris"]
                },
                {
                    "name": "arithmetic",
                    "prompt": "Q: What is 15 + 27?\nA:",
                    "expected": ["42"]
                },
                {
                    "name": "completion",
                    "prompt": "The quick brown fox",
                    "expected": ["jumps", "jumped"]
                }
            ]
        
        results = {}
        
        for task in tasks:
            inputs = self.tokenizer(task["prompt"], return_tensors="pt").to(self.config.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=20,
                    temperature=0.1,
                    do_sample=True
                )
            
            generated = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            ).strip().lower()
            
            # Check if any expected answer is in the generation
            correct = any(exp.lower() in generated for exp in task["expected"])
            results[task["name"]] = 1.0 if correct else 0.0
        
        return results


def evaluate_model(
    model: Union[GPT2LMHeadModel, str],
    tokenizer: Union[GPT2TokenizerFast, str],
    dataset: Dataset,
    output_dir: Optional[Union[str, Path]] = None,
    num_samples: int = 1000,
    config: Optional[EvaluationConfig] = None
) -> Dict[str, float]:
    """
    Main evaluation function
    
    Args:
        model: Model or path to model
        tokenizer: Tokenizer or path to tokenizer
        dataset: Evaluation dataset
        output_dir: Directory to save results
        num_samples: Number of samples for generation evaluation
        config: Evaluation configuration
        
    Returns:
        Dictionary of evaluation metrics
    """
    
    # Load model and tokenizer if paths provided
    if isinstance(model, str):
        model = GPT2LMHeadModel.from_pretrained(model)
    if isinstance(tokenizer, str):
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer)
    
    # Create evaluator
    config = config or EvaluationConfig(num_samples=num_samples)
    evaluator = ModelEvaluator(model, tokenizer, config)
    
    # Compute metrics
    results = {}
    
    # Perplexity
    perplexity = evaluator.compute_perplexity(dataset)
    results["perplexity"] = perplexity
    logger.info(f"Perplexity: {perplexity:.3f}")
    
    # Generation quality
    gen_results, predictions, references = evaluator.evaluate_generation_quality(
        dataset, num_samples
    )
    results.update(gen_results)
    
    # Few-shot tasks
    few_shot_results = evaluator.evaluate_few_shot_tasks()
    results["few_shot"] = few_shot_results
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metrics
        with open(output_dir / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save predictions if requested
        if config.save_predictions and predictions:
            with open(output_dir / "predictions.json", "w") as f:
                json.dump({
                    "predictions": predictions[:100],  # Save first 100
                    "references": references[:100]
                }, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    # Test evaluation
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="./eval_results", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=1000, help="Number of samples")
    
    args = parser.parse_args()
    
    # Load tokenizer and dataset
    from datasets import load_from_disk
    
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    datasets = load_from_disk(args.dataset_path)
    eval_dataset = datasets["validation"]
    
    # Run evaluation
    results = evaluate_model(
        args.model_path,
        tokenizer,
        eval_dataset,
        args.output_dir,
        args.num_samples
    )
    
    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    for key, value in results.items():
        if isinstance(value, dict):
            print(f"{key}:")
            for k, v in value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"{key}: {value:.4f}") 