#!/usr/bin/env python3
"""
Filter datasets based on quality scores
Creates ultra-high, high, and medium quality subsets for comparison
"""

import os
import json
import pickle
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from itertools import islice

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from quality_scorer import QualityScorer, QualityDimensions

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetFilter:
    """Filter and create quality-based subsets of synthetic datasets"""
    
    def __init__(
        self,
        quality_scorer: Optional[QualityScorer] = None,
        cache_dir: str = "./cache",
        save_dir: str = "./data"
    ):
        self.scorer = quality_scorer or QualityScorer()
        self.cache_dir = Path(cache_dir)
        self.save_dir = Path(save_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def score_dataset(
        self,
        dataset_name: str = "teknium/OpenHermes-2.5",
        max_samples: Optional[int] = None,
        batch_size: int = 1000,
        force_rescore: bool = False
    ) -> List[Dict]:
        """Score all samples in a dataset"""
        
        # Check for cached scores
        cache_file = self.cache_dir / f"{dataset_name.replace('/', '_')}_scores.pkl"
        
        if cache_file.exists() and not force_rescore:
            logger.info(f"Loading cached scores from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        
        logger.info(f"Loading dataset {dataset_name} in streaming mode")
        dataset = load_dataset(dataset_name, split="train", streaming=True)
        
        if max_samples:
            logger.info(f"Limiting to {max_samples} samples")
            dataset = dataset.take(max_samples)
        
        logger.info("Scoring samples from streamed dataset...")
        
        scored_data = []
        dataset_iterator = iter(dataset)
        
        with tqdm(desc="Scoring batches") as pbar:
            while True:
                batch_samples = list(islice(dataset_iterator, batch_size))
                if not batch_samples:
                    break
                
                # Score batch
                batch_scores = self.scorer.score_batch(batch_samples, show_progress=False)
                
                # Combine samples with scores
                for j, (score, components) in enumerate(batch_scores):
                    scored_data.append({
                        'idx': len(scored_data),
                        'sample': batch_samples[j],
                        'score': score,
                        'components': components
                    })

                pbar.update(len(batch_samples))
                
                # Save checkpoint every 10k samples
                num_scored = len(scored_data)
                if num_scored > 0 and (num_scored // 10000) > ((num_scored - len(batch_samples)) // 10000):
                    checkpoint_file = self.cache_dir / f"{dataset_name.replace('/', '_')}_scores_checkpoint_{num_scored}.pkl"
                    logger.info(f"Saving scoring checkpoint to {checkpoint_file}")
                    with open(checkpoint_file, "wb") as f:
                        pickle.dump(scored_data, f)

        # Save final scores
        with open(cache_file, "wb") as f:
            pickle.dump(scored_data, f)
        
        logger.info(f"Scoring complete. Saved to {cache_file}")
        
        return scored_data
    
    def create_quality_subsets(
        self,
        scored_data: List[Dict],
        subset_sizes: Dict[str, int] = None
    ) -> Dict[str, List[Dict]]:
        """Create quality-based subsets"""
        
        if subset_sizes is None:
            subset_sizes = {
                'ultra_high_10k': 10000,
                'high_100k': 100000,
                'medium_1m': 1000000
            }
        
        # Sort by quality score
        logger.info("Sorting by quality score...")
        scored_data.sort(key=lambda x: x['score'], reverse=True)
        
        # Create subsets
        subsets = {}
        
        for name, size in subset_sizes.items():
            actual_size = min(size, len(scored_data))
            subsets[name] = scored_data[:actual_size]
            logger.info(f"Created {name} subset with {actual_size} samples")
        
        # Analyze score distributions
        self._analyze_subsets(subsets)
        
        return subsets
    
    def _analyze_subsets(self, subsets: Dict[str, List[Dict]]):
        """Analyze and visualize subset statistics"""
        
        stats = {}
        
        for name, data in subsets.items():
            scores = [item['score'] for item in data]
            component_scores = {
                'clarity': [item['components']['clarity'] for item in data],
                'response_quality': [item['components']['response_quality'] for item in data],
                'diversity': [item['components']['diversity'] for item in data],
                'complexity': [item['components']['complexity'] for item in data]
            }
            
            stats[name] = {
                'total': {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores),
                    'median': np.median(scores)
                }
            }
            
            # Component stats
            for comp_name, comp_scores in component_scores.items():
                stats[name][comp_name] = {
                    'mean': np.mean(comp_scores),
                    'std': np.std(comp_scores)
                }
            
            # Print statistics
            print(f"\n{name.upper()} Statistics:")
            print(f"  Total Score - Mean: {stats[name]['total']['mean']:.3f} (±{stats[name]['total']['std']:.3f})")
            print(f"  Range: [{stats[name]['total']['min']:.3f}, {stats[name]['total']['max']:.3f}]")
            print(f"  Component Means:")
            for comp in ['clarity', 'response_quality', 'diversity', 'complexity']:
                print(f"    {comp}: {stats[name][comp]['mean']:.3f}")
        
        # Save statistics
        stats_file = self.save_dir / "subset_statistics.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        
        # Create visualizations
        self._create_visualizations(subsets)
    
    def _create_visualizations(self, subsets: Dict[str, List[Dict]]):
        """Create quality distribution visualizations"""
        
        # Set up the plot style
        sns.set_style("whitegrid")
        
        # 1. Score distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Quality Score Distributions by Subset', fontsize=16)
        
        colors = {'ultra_high_10k': '#2ecc71', 'high_100k': '#3498db', 'medium_1m': '#e74c3c'}
        
        # Total scores
        ax = axes[0, 0]
        for name, data in subsets.items():
            scores = [item['score'] for item in data[:10000]]  # Limit for visualization
            ax.hist(scores, bins=50, alpha=0.6, label=name, color=colors.get(name))
        ax.set_xlabel('Total Quality Score')
        ax.set_ylabel('Count')
        ax.set_title('Overall Quality Distribution')
        ax.legend()
        
        # Component scores
        components = ['clarity', 'response_quality', 'diversity']
        for i, comp in enumerate(components):
            ax = axes.flatten()[i + 1]
            for name, data in subsets.items():
                comp_scores = [item['components'][comp] for item in data[:10000]]
                ax.hist(comp_scores, bins=30, alpha=0.6, label=name, color=colors.get(name))
            ax.set_xlabel(f'{comp.replace("_", " ").title()} Score')
            ax.set_ylabel('Count')
            ax.set_title(f'{comp.replace("_", " ").title()} Distribution')
            if i == 0:
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'quality_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Component correlation heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Get component scores for correlation
        all_components = []
        for data in subsets['medium_1m'][:10000]:  # Use medium subset
            all_components.append([
                data['components']['clarity'],
                data['components']['response_quality'],
                data['components']['diversity'],
                data['components']['complexity']
            ])
        
        df = pd.DataFrame(all_components, columns=['Clarity', 'Response Quality', 'Diversity', 'Complexity'])
        corr_matrix = df.corr()
        
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Quality Component Correlations')
        plt.savefig(self.save_dir / 'component_correlations.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Quality threshold plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get scores for all data
        all_scores = sorted([item['score'] for item in subsets['medium_1m']], reverse=True)
        
        # Plot score vs rank
        ranks = np.arange(len(all_scores))
        ax.plot(ranks, all_scores, 'b-', alpha=0.6)
        
        # Mark subset boundaries
        for name, size in [('ultra_high_10k', 10000), ('high_100k', 100000)]:
            if size < len(all_scores):
                ax.axvline(x=size, color='red', linestyle='--', alpha=0.7)
                ax.text(size, all_scores[size], f' {name} cutoff', rotation=90, va='bottom')
        
        ax.set_xlabel('Sample Rank')
        ax.set_ylabel('Quality Score')
        ax.set_title('Quality Score vs Sample Rank')
        ax.set_xlim(0, min(1000000, len(all_scores)))
        ax.grid(True, alpha=0.3)
        
        # Add log scale option
        ax.set_xscale('log')
        
        plt.savefig(self.save_dir / 'quality_thresholds.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_filtered_datasets(
        self,
        subsets: Dict[str, List[Dict]],
        tokenizer_name: str = "gpt2"
    ):
        """Save filtered datasets in HuggingFace format"""
        
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        for name, data in subsets.items():
            logger.info(f"Saving {name} dataset...")
            
            # Extract samples
            samples = []
            for item in data:
                sample = item['sample'].copy()
                # Add quality metadata
                sample['quality_score'] = item['score']
                sample['quality_components'] = item['components']
                samples.append(sample)
            
            # Create dataset
            dataset = Dataset.from_list(samples)
            
            # Create train/validation split
            dataset_dict = dataset.train_test_split(test_size=0.05, seed=42)
            
            # Save
            save_path = self.save_dir / name
            dataset_dict.save_to_disk(str(save_path))
            
            # Save metadata
            metadata = {
                'subset_name': name,
                'total_samples': len(data),
                'train_samples': len(dataset_dict['train']),
                'validation_samples': len(dataset_dict['test']),
                'score_threshold': data[-1]['score'],  # Minimum score in subset
                'mean_score': np.mean([d['score'] for d in data]),
                'score_range': [
                    min(d['score'] for d in data),
                    max(d['score'] for d in data)
                ]
            }
            
            with open(save_path / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved {name} to {save_path}")
            
            # Also save score details
            scores_df = pd.DataFrame([
                {
                    'idx': item['idx'],
                    'score': item['score'],
                    **item['components']
                }
                for item in data
            ])
            scores_df.to_csv(save_path / 'quality_scores.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description="Filter datasets by quality")
    parser.add_argument("--dataset", type=str, default="teknium/OpenHermes-2.5",
                        help="Dataset to filter")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to process")
    parser.add_argument("--save_dir", type=str, default="./data",
                        help="Directory to save filtered datasets")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory for cached scores")
    parser.add_argument("--force_rescore", action="store_true",
                        help="Force rescoring even if cache exists")
    
    # Subset size arguments
    parser.add_argument("--ultra_high_size", type=int, default=10000,
                        help="Size of ultra-high quality subset")
    parser.add_argument("--high_size", type=int, default=100000,
                        help="Size of high quality subset")
    parser.add_argument("--medium_size", type=int, default=1000000,
                        help="Size of medium quality subset")
    
    args = parser.parse_args()
    
    # Create filter
    filter = DatasetFilter(
        cache_dir=args.cache_dir,
        save_dir=args.save_dir
    )
    
    # Score dataset
    scored_data = filter.score_dataset(
        dataset_name=args.dataset,
        max_samples=args.max_samples,
        force_rescore=args.force_rescore
    )
    
    # Create subsets
    subset_sizes = {
        f'ultra_high_{args.ultra_high_size//1000}k': args.ultra_high_size,
        f'high_{args.high_size//1000}k': args.high_size,
        f'medium_{args.medium_size//1000}k': args.medium_size
    }
    
    subsets = filter.create_quality_subsets(scored_data, subset_sizes)
    
    # Save datasets
    filter.save_filtered_datasets(subsets)
    
    logger.info("Dataset filtering complete!")


if __name__ == "__main__":
    main() 