#!/usr/bin/env python3
"""
Data preparation for Experiment 1: Pure Synthetic Excellence
Handles loading and preprocessing of OpenHermes-2.5 dataset
"""

import os
import json
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch
from datasets import load_dataset, Dataset, DatasetDict
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np


@dataclass
class DataConfig:
    """Configuration for data preparation"""
    dataset_name: str = "teknium/OpenHermes-2.5"
    max_length: int = 2048
    train_split: float = 0.95
    seed: int = 42
    cache_dir: str = "./cache"
    streaming: bool = False
    max_samples: Optional[int] = None  # For testing


class OpenHermesProcessor:
    """Process OpenHermes-2.5 dataset for training"""
    
    def __init__(self, tokenizer: AutoTokenizer, config: DataConfig):
        self.tokenizer = tokenizer
        self.config = config
        self.tokenizer.pad_token = tokenizer.eos_token
        
    def format_instruction_response(self, example: Dict) -> str:
        """Format a single example into instruction-response format"""
        # Handle different possible formats in the dataset
        if "conversations" in example:
            # Multi-turn conversation format
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
            return text.strip()
        else:
            # Simple instruction-response format
            instruction = example.get("instruction", "")
            response = example.get("output", example.get("response", ""))
            
            # Include system prompt if available
            system = example.get("system", "")
            if system:
                return f"System: {system}\n\n### Instruction:\n{instruction}\n\n### Response:\n{response}"
            else:
                return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
    
    def tokenize_function(self, examples: Dict) -> Dict:
        """Tokenize a batch of examples"""
        # Format all examples
        texts = []
        for i in range(len(examples[list(examples.keys())[0]])):
            example = {k: v[i] for k, v in examples.items()}
            text = self.format_instruction_response(example)
            texts.append(text)
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding="longest",
            max_length=self.config.max_length,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids for language modeling)
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    def prepare_dataset(self) -> DatasetDict:
        """Load and prepare the dataset"""
        print(f"Loading {self.config.dataset_name}...")
        
        # Load dataset
        if self.config.streaming:
            dataset = load_dataset(
                self.config.dataset_name,
                split="train",
                streaming=True,
                cache_dir=self.config.cache_dir
            )
            
            # If max_samples is set, take only that many
            if self.config.max_samples:
                dataset = dataset.take(self.config.max_samples)
                # Convert to regular dataset for splitting
                dataset = Dataset.from_generator(lambda: dataset)
        else:
            dataset = load_dataset(
                self.config.dataset_name,
                split="train",
                cache_dir=self.config.cache_dir
            )
            
            # Limit samples if specified
            if self.config.max_samples:
                dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
        
        print(f"Dataset loaded. Total samples: {len(dataset)}")
        
        # Create train/validation split
        dataset_dict = dataset.train_test_split(
            test_size=1 - self.config.train_split,
            seed=self.config.seed
        )
        
        print(f"Train samples: {len(dataset_dict['train'])}")
        print(f"Validation samples: {len(dataset_dict['test'])}")
        
        # Rename test to validation
        dataset_dict["validation"] = dataset_dict.pop("test")
        
        # Tokenize datasets
        print("Tokenizing datasets...")
        tokenized_datasets = dataset_dict.map(
            self.tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=dataset_dict["train"].column_names,
            desc="Tokenizing"
        )
        
        return tokenized_datasets
    
    def analyze_dataset(self, dataset: Dataset, num_samples: int = 1000):
        """Analyze dataset statistics"""
        print("\nDataset Analysis:")
        print("-" * 50)
        
        # Sample examples
        samples = dataset.select(range(min(num_samples, len(dataset))))
        
        # Length statistics
        lengths = []
        for example in tqdm(samples, desc="Analyzing lengths"):
            text = self.format_instruction_response(example)
            tokens = self.tokenizer.encode(text)
            lengths.append(len(tokens))
        
        print(f"Average token length: {np.mean(lengths):.2f}")
        print(f"Median token length: {np.median(lengths):.2f}")
        print(f"Max token length: {np.max(lengths)}")
        print(f"Min token length: {np.min(lengths)}")
        print(f"Std token length: {np.std(lengths):.2f}")
        
        # Show length distribution
        print("\nLength distribution:")
        bins = [0, 256, 512, 1024, 2048, 4096, 8192]
        hist, _ = np.histogram(lengths, bins=bins)
        for i in range(len(hist)):
            print(f"  {bins[i]:>4} - {bins[i+1]:>4}: {hist[i]:>6} ({hist[i]/len(lengths)*100:>5.1f}%)")
        
        # Show example
        print("\nExample formatted text:")
        print("-" * 50)
        example_text = self.format_instruction_response(samples[0])
        print(example_text[:500] + "..." if len(example_text) > 500 else example_text)
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenHermes-2.5 dataset")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name for tokenizer")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--train_split", type=float, default=0.95, help="Train split ratio")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum samples to use (for testing)")
    parser.add_argument("--analyze", action="store_true", help="Analyze dataset statistics")
    parser.add_argument("--save_dir", type=str, default="./data", help="Directory to save processed data")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode for large datasets")
    
    args = parser.parse_args()
    
    # Create config
    config = DataConfig(
        max_length=args.max_length,
        train_split=args.train_split,
        max_samples=args.max_samples,
        streaming=args.streaming
    )
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create processor
    processor = OpenHermesProcessor(tokenizer, config)
    
    # Analyze if requested
    if args.analyze:
        dataset = load_dataset(
            config.dataset_name,
            split="train",
            streaming=False,
            cache_dir=config.cache_dir
        )
        if args.max_samples:
            dataset = dataset.select(range(min(args.max_samples, len(dataset))))
        processor.analyze_dataset(dataset)
        return
    
    # Prepare dataset
    tokenized_datasets = processor.prepare_dataset()
    
    # Save processed datasets
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "openhermes_processed")
    
    print(f"\nSaving processed datasets to {save_path}")
    tokenized_datasets.save_to_disk(save_path)
    
    # Save config
    config_path = os.path.join(save_path, "config.json")
    with open(config_path, "w") as f:
        json.dump({
            "model_name": args.model_name,
            "max_length": config.max_length,
            "train_split": config.train_split,
            "dataset_name": config.dataset_name,
            "train_samples": len(tokenized_datasets["train"]),
            "validation_samples": len(tokenized_datasets["validation"])
        }, f, indent=2)
    
    print("Data preparation complete!")
    print(f"Train samples: {len(tokenized_datasets['train'])}")
    print(f"Validation samples: {len(tokenized_datasets['validation'])}")


if __name__ == "__main__":
    main() 