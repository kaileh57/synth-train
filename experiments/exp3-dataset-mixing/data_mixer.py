#!/usr/bin/env python3
"""
Dataset mixing implementation for Experiment 3
Creates optimal combinations of different synthetic datasets
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
from collections import Counter
import logging

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetMixer:
    """Mix different synthetic datasets with specified ratios"""
    
    def __init__(
        self,
        cache_dir: str = "./cache",
        save_dir: str = "./data/mixed_datasets",
        seed: int = 42
    ):
        self.cache_dir = Path(cache_dir)
        self.save_dir = Path(save_dir)
        self.seed = seed
        
        # Create directories
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Dataset configurations
        self.dataset_configs = {
            'openhermes': {
                'name': 'teknium/OpenHermes-2.5',
                'split': 'train',
                'format_func': self._format_openhermes,
                'description': 'High-quality instruction-following data from GPT-4'
            },
            'cosmopedia': {
                'name': 'HuggingFaceTB/cosmopedia',
                'split': 'train',
                'format_func': self._format_cosmopedia,
                'description': 'Educational content generated by Mixtral-8x7B',
                'config': 'auto_math_text'  # Use smaller subset
            },
            'magpie': {
                'name': 'Magpie-Align/MagpieLM-Pro-300K-v0.1',
                'split': 'train',
                'format_func': self._format_magpie,
                'description': 'Multi-turn conversations from Llama-3.1-70B'
            },
            'fineweb': {
                'name': 'HuggingFaceFW/fineweb-edu',
                'split': 'train',
                'format_func': self._format_fineweb,
                'description': 'High-quality educational web content'
            }
        }
        
        self.loaded_datasets = {}
        self.dataset_stats = {}
    
    def load_dataset_sample(self, dataset_name: str, max_samples: int) -> List[Dict]:
        """Load a sample from a dataset"""
        
        if dataset_name not in self.dataset_configs:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        config = self.dataset_configs[dataset_name]
        logger.info(f"Loading {dataset_name} ({config['description']})")
        
        # Load dataset with streaming for memory efficiency
        try:
            if 'config' in config:
                dataset = load_dataset(
                    config['name'],
                    config['config'],
                    split=config['split'],
                    streaming=True,
                    cache_dir=self.cache_dir
                )
            else:
                dataset = load_dataset(
                    config['name'],
                    split=config['split'],
                    streaming=True,
                    cache_dir=self.cache_dir
                )
            
            # Take samples
            samples = []
            for i, sample in enumerate(dataset):
                if i >= max_samples:
                    break
                
                # Format sample
                formatted = config['format_func'](sample)
                if formatted:  # Skip if formatting failed
                    formatted['source'] = dataset_name
                    formatted['source_index'] = i
                    samples.append(formatted)
            
            logger.info(f"Loaded {len(samples)} samples from {dataset_name}")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            return []
    
    def _format_openhermes(self, sample: Dict) -> Optional[Dict]:
        """Format OpenHermes sample"""
        try:
            if "conversations" in sample:
                # Multi-turn conversation format
                text = ""
                for turn in sample["conversations"]:
                    role = turn.get("from", "")
                    content = turn.get("value", "")
                    if role == "system":
                        text += f"System: {content}\n\n"
                    elif role == "human":
                        text += f"### Instruction:\n{content}\n\n"
                    elif role == "gpt":
                        text += f"### Response:\n{content}\n\n"
                
                return {
                    "text": text.strip(),
                    "type": "instruction",
                    "length": len(text.split())
                }
            else:
                # Simple format
                instruction = sample.get("instruction", "")
                response = sample.get("output", sample.get("response", ""))
                
                if not instruction or not response:
                    return None
                
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"
                return {
                    "text": text,
                    "type": "instruction",
                    "length": len(text.split())
                }
        except Exception:
            return None
    
    def _format_cosmopedia(self, sample: Dict) -> Optional[Dict]:
        """Format Cosmopedia sample"""
        try:
            text = sample.get("text", "")
            if not text or len(text) < 100:  # Skip very short texts
                return None
            
            # Limit length for memory efficiency
            words = text.split()
            if len(words) > 1000:  # Limit to ~1000 words
                text = " ".join(words[:1000])
            
            return {
                "text": text,
                "type": "educational",
                "length": len(text.split())
            }
        except Exception:
            return None
    
    def _format_magpie(self, sample: Dict) -> Optional[Dict]:
        """Format Magpie sample"""
        try:
            conversations = sample.get("conversations", [])
            if not conversations:
                return None
            
            text = ""
            for turn in conversations:
                role = turn.get("role", "")
                content = turn.get("content", "")
                
                if role == "user":
                    text += f"User: {content}\n\n"
                elif role == "assistant":
                    text += f"Assistant: {content}\n\n"
            
            if not text:
                return None
            
            return {
                "text": text.strip(),
                "type": "conversation",
                "length": len(text.split())
            }
        except Exception:
            return None
    
    def _format_fineweb(self, sample: Dict) -> Optional[Dict]:
        """Format FineWeb sample"""
        try:
            text = sample.get("text", "")
            if not text or len(text) < 200:  # Skip very short texts
                return None
            
            # Limit length for memory efficiency
            words = text.split()
            if len(words) > 800:  # Limit to ~800 words
                text = " ".join(words[:800])
            
            return {
                "text": text,
                "type": "web_educational",
                "length": len(text.split())
            }
        except Exception:
            return None
    
    def create_mixed_dataset(
        self,
        mixing_ratios: Dict[str, float],
        total_samples: int = 100000,
        strategy_name: str = "custom_mix"
    ) -> Dataset:
        """Create a mixed dataset according to specified ratios"""
        
        # Validate ratios
        total_ratio = sum(mixing_ratios.values())
        if abs(total_ratio - 1.0) > 0.01:
            logger.warning(f"Mixing ratios sum to {total_ratio}, normalizing...")
            mixing_ratios = {k: v/total_ratio for k, v in mixing_ratios.items()}
        
        # Calculate samples per dataset
        samples_per_dataset = {}
        for dataset_name, ratio in mixing_ratios.items():
            samples_per_dataset[dataset_name] = int(ratio * total_samples)
        
        logger.info(f"Creating {strategy_name} with {total_samples} total samples:")
        for dataset_name, count in samples_per_dataset.items():
            logger.info(f"  {dataset_name}: {count} samples ({mixing_ratios[dataset_name]*100:.1f}%)")
        
        # Load samples from each dataset
        all_samples = []
        dataset_contributions = {}
        
        for dataset_name, num_samples in samples_per_dataset.items():
            if num_samples == 0:
                continue
                
            samples = self.load_dataset_sample(dataset_name, num_samples)
            all_samples.extend(samples)
            dataset_contributions[dataset_name] = len(samples)
            
            logger.info(f"Added {len(samples)} samples from {dataset_name}")
        
        # Shuffle for good mixing
        random.shuffle(all_samples)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_list(all_samples)
        
        # Add metadata
        metadata = {
            "strategy_name": strategy_name,
            "mixing_ratios": mixing_ratios,
            "total_samples": len(all_samples),
            "dataset_contributions": dataset_contributions,
            "creation_time": pd.Timestamp.now().isoformat(),
            "seed": self.seed
        }
        
        return dataset, metadata
    
    def save_mixed_dataset(
        self,
        dataset: Dataset,
        metadata: Dict,
        strategy_name: str
    ):
        """Save mixed dataset and associated metadata"""
        
        # Create directory for this strategy
        strategy_dir = self.save_dir / strategy_name
        strategy_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into train/validation
        dataset_split = dataset.train_test_split(test_size=0.05, seed=self.seed)
        
        # Save dataset
        dataset_split.save_to_disk(str(strategy_dir))
        
        # Save metadata
        with open(strategy_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved {strategy_name} to {strategy_dir}")
        
        return strategy_dir


# Predefined mixing strategies
MIXING_STRATEGIES = {
    "equal_mix": {
        "openhermes": 0.25,
        "cosmopedia": 0.25,
        "magpie": 0.25,
        "fineweb": 0.25
    },
    "instruction_heavy": {
        "openhermes": 0.40,
        "cosmopedia": 0.20,
        "magpie": 0.30,
        "fineweb": 0.10
    },
    "knowledge_heavy": {
        "openhermes": 0.20,
        "cosmopedia": 0.40,
        "magpie": 0.10,
        "fineweb": 0.30
    },
    "conversation_heavy": {
        "openhermes": 0.20,
        "cosmopedia": 0.10,
        "magpie": 0.50,
        "fineweb": 0.20
    },
    "quality_weighted": {
        "openhermes": 0.35,  # Highest quality
        "cosmopedia": 0.25,
        "magpie": 0.30,
        "fineweb": 0.10
    },
    "capability_balanced": {
        "openhermes": 0.30,
        "cosmopedia": 0.25,
        "magpie": 0.25,
        "fineweb": 0.20
    }
}


def main():
    parser = argparse.ArgumentParser(description="Create mixed datasets for Experiment 3")
    parser.add_argument("--strategy", type=str, choices=list(MIXING_STRATEGIES.keys()) + ["all"],
                        default="all", help="Mixing strategy to create")
    parser.add_argument("--total_samples", type=int, default=100000,
                        help="Total number of samples in mixed dataset")
    parser.add_argument("--output_dir", type=str, default="./data/mixed_datasets",
                        help="Output directory for mixed datasets")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Cache directory for downloads")
    
    args = parser.parse_args()
    
    # Initialize mixer
    mixer = DatasetMixer(
        cache_dir=args.cache_dir,
        save_dir=args.output_dir
    )
    
    # Determine which strategies to run
    if args.strategy == "all":
        strategies_to_run = MIXING_STRATEGIES
    else:
        strategies_to_run = {args.strategy: MIXING_STRATEGIES[args.strategy]}
    
    # Create each strategy
    for strategy_name, mixing_ratios in strategies_to_run.items():
        logger.info(f"\nCreating strategy: {strategy_name}")
        
        # Create mixed dataset
        dataset, metadata = mixer.create_mixed_dataset(
            mixing_ratios,
            total_samples=args.total_samples,
            strategy_name=strategy_name
        )
        
        # Save everything
        mixer.save_mixed_dataset(dataset, metadata, strategy_name)
        
        logger.info(f"Strategy {strategy_name} complete!")
    
    logger.info(f"\nAll mixing strategies complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()