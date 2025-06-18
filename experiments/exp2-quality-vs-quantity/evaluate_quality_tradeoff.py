#!/usr/bin/env python3
"""
Evaluate the quality vs quantity tradeoff
Analyze how models trained on different quality/quantity subsets perform
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

logger = logging.getLogger(__name__)


@dataclass
class QualityBin:
    """Represents a quality score bin"""
    name: str
    min_score: float
    max_score: float
    samples: List[Dict]
    
    def __len__(self):
        return len(self.samples)
    
    @property
    def mean_score(self):
        if not self.samples:
            return 0.0
        scores = [s.get('quality_score', 0) for s in self.samples]
        return np.mean(scores)


class QualityQuantityEvaluator:
    """Evaluate model performance across quality dimensions"""
    
    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: GPT2TokenizerFast,
        device: Optional[str] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate_quality_impact(
        self,
        eval_dataset: Dataset,
        output_dir: Optional[Path] = None,
        num_quality_bins: int = 5
    ) -> Dict:
        """Evaluate how model performs on different quality levels"""
        
        # Create quality bins if dataset has quality scores
        if hasattr(eval_dataset, "features") and "quality_score" in eval_dataset.features:
            quality_bins = self._create_quality_bins(eval_dataset, num_quality_bins)
        else:
            # Fallback: treat all as one bin
            quality_bins = [QualityBin(
                name="all",
                min_score=0,
                max_score=1,
                samples=[eval_dataset[i] for i in range(len(eval_dataset))]
            )]
        
        # Evaluate each quality bin
        results = {
            "by_quality_bin": {},
            "overall_metrics": {},
            "quality_correlation": {}
        }
        
        all_perplexities = []
        all_quality_scores = []
        
        for bin in tqdm(quality_bins, desc="Evaluating quality bins"):
            if len(bin) == 0:
                continue
                
            bin_results = self._evaluate_bin(bin)
            results["by_quality_bin"][bin.name] = {
                "num_samples": len(bin),
                "mean_quality_score": bin.mean_score,
                "score_range": [bin.min_score, bin.max_score],
                **bin_results
            }
            
            # Collect for correlation analysis
            if "perplexity" in bin_results:
                all_perplexities.extend([bin_results["perplexity"]] * len(bin))
                all_quality_scores.extend([s.get('quality_score', bin.mean_score) for s in bin.samples])
        
        # Calculate correlations
        if all_perplexities and all_quality_scores:
            from scipy.stats import pearsonr, spearmanr
            
            pearson_corr, pearson_p = pearsonr(all_quality_scores, all_perplexities)
            spearman_corr, spearman_p = spearmanr(all_quality_scores, all_perplexities)
            
            results["quality_correlation"] = {
                "pearson_r": pearson_corr,
                "pearson_p": pearson_p,
                "spearman_r": spearman_corr,
                "spearman_p": spearman_p
            }
        
        # Calculate overall metrics
        results["overall_metrics"] = self._calculate_overall_metrics(results["by_quality_bin"])
        
        # Save results and create visualizations
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save results
            with open(output_dir / "quality_impact_results.json", "w") as f:
                json.dump(results, f, indent=2)
            
            # Create visualizations
            self._create_quality_visualizations(results, output_dir)
        
        return results
    
    def _create_quality_bins(self, dataset: Dataset, num_bins: int) -> List[QualityBin]:
        """Create quality score bins"""
        
        # Get quality scores
        quality_scores = []
        samples_with_scores = []
        
        for i in range(len(dataset)):
            sample = dataset[i]
            if "quality_score" in sample:
                quality_scores.append(sample["quality_score"])
                samples_with_scores.append((sample["quality_score"], sample))
        
        if not quality_scores:
            # No quality scores available
            return []
        
        # Calculate percentile boundaries
        percentiles = np.linspace(0, 100, num_bins + 1)
        boundaries = np.percentile(quality_scores, percentiles)
        
        # Create bins
        bins = []
        for i in range(num_bins):
            min_score = boundaries[i]
            max_score = boundaries[i + 1]
            
            # Filter samples for this bin
            bin_samples = [
                sample for score, sample in samples_with_scores
                if min_score <= score <= max_score
            ]
            
            # Avoid duplicates at boundaries
            if i > 0:
                bin_samples = [
                    sample for score, sample in samples_with_scores
                    if min_score < score <= max_score
                ]
            
            bin_name = f"bin_{i+1}_q{int(percentiles[i])}-{int(percentiles[i+1])}"
            
            bins.append(QualityBin(
                name=bin_name,
                min_score=min_score,
                max_score=max_score,
                samples=bin_samples
            ))
        
        return bins
    
    def _evaluate_bin(self, bin: QualityBin) -> Dict:
        """Evaluate a single quality bin"""
        
        if not bin.samples:
            return {}
        
        # Calculate perplexity
        perplexities = []
        response_lengths = []
        
        for sample in tqdm(bin.samples[:min(100, len(bin))], desc=f"Evaluating {bin.name}", leave=False):
            # Calculate perplexity
            input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                perplexity = torch.exp(outputs.loss).item()
                
                if not np.isnan(perplexity) and not np.isinf(perplexity):
                    perplexities.append(perplexity)
            
            # Get response length
            if "labels" in sample:
                response_length = len([t for t in sample["labels"] if t != -100])
                response_lengths.append(response_length)
        
        results = {}
        
        if perplexities:
            results["perplexity"] = np.mean(perplexities)
            results["perplexity_std"] = np.std(perplexities)
            results["perplexity_median"] = np.median(perplexities)
        
        if response_lengths:
            results["avg_response_length"] = np.mean(response_lengths)
            results["response_length_std"] = np.std(response_lengths)
        
        # Sample generation quality
        generation_scores = self._evaluate_generation_quality(bin.samples[:10])
        results.update(generation_scores)
        
        return results
    
    def _evaluate_generation_quality(self, samples: List[Dict]) -> Dict:
        """Evaluate generation quality on a small set of samples"""
        
        diversity_scores = []
        coherence_scores = []
        
        for sample in samples:
            # Generate response
            if "input_ids" in sample:
                input_ids = torch.tensor(sample["input_ids"][:512]).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=100,
                        temperature=0.8,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(
                    outputs[0][input_ids.shape[1]:],
                    skip_special_tokens=True
                )
                
                # Simple diversity score (unique tokens ratio)
                tokens = generated_text.split()
                if tokens:
                    diversity = len(set(tokens)) / len(tokens)
                    diversity_scores.append(diversity)
                
                # Simple coherence score (based on sentence endings)
                if generated_text.strip():
                    ends_properly = generated_text.strip()[-1] in '.!?'
                    coherence_scores.append(1.0 if ends_properly else 0.5)
        
        results = {}
        
        if diversity_scores:
            results["generation_diversity"] = np.mean(diversity_scores)
        
        if coherence_scores:
            results["generation_coherence"] = np.mean(coherence_scores)
        
        return results
    
    def _calculate_overall_metrics(self, bin_results: Dict) -> Dict:
        """Calculate overall metrics across all bins"""
        
        # Weighted averages based on sample count
        total_samples = sum(r["num_samples"] for r in bin_results.values())
        
        metrics = {}
        metric_names = ["perplexity", "generation_diversity", "generation_coherence"]
        
        for metric in metric_names:
            weighted_sum = 0
            weight_sum = 0
            
            for bin_name, results in bin_results.items():
                if metric in results:
                    weight = results["num_samples"]
                    weighted_sum += results[metric] * weight
                    weight_sum += weight
            
            if weight_sum > 0:
                metrics[f"weighted_avg_{metric}"] = weighted_sum / weight_sum
        
        # Quality distribution stats
        quality_scores = []
        for results in bin_results.values():
            quality_scores.extend([results["mean_quality_score"]] * results["num_samples"])
        
        if quality_scores:
            metrics["quality_mean"] = np.mean(quality_scores)
            metrics["quality_std"] = np.std(quality_scores)
        
        return metrics
    
    def _create_quality_visualizations(self, results: Dict, output_dir: Path):
        """Create visualizations for quality impact analysis"""
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Perplexity vs Quality plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bin_names = []
        perplexities = []
        quality_means = []
        sample_counts = []
        
        for bin_name, bin_results in results["by_quality_bin"].items():
            if "perplexity" in bin_results:
                bin_names.append(bin_name)
                perplexities.append(bin_results["perplexity"])
                quality_means.append(bin_results["mean_quality_score"])
                sample_counts.append(bin_results["num_samples"])
        
        if perplexities:
            # Create scatter plot with bubble size based on sample count
            scatter = ax.scatter(
                quality_means,
                perplexities,
                s=[np.sqrt(n) * 10 for n in sample_counts],
                alpha=0.6,
                c=quality_means,
                cmap='viridis'
            )
            
            ax.set_xlabel('Mean Quality Score')
            ax.set_ylabel('Perplexity')
            ax.set_title('Model Perplexity vs Data Quality')
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Quality Score')
            
            # Add trend line if correlation exists
            if "quality_correlation" in results and results["quality_correlation"]:
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(quality_means, perplexities)
                line_x = np.array([min(quality_means), max(quality_means)])
                line_y = slope * line_x + intercept
                ax.plot(line_x, line_y, 'r--', alpha=0.8,
                       label=f'Linear fit (r={r_value:.3f}, p={p_value:.3f})')
                ax.legend()
            
            plt.savefig(output_dir / 'perplexity_vs_quality.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Quality bin comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Performance Metrics by Quality Bin', fontsize=16)
        
        metrics = ['perplexity', 'generation_diversity', 'generation_coherence', 'num_samples']
        titles = ['Perplexity', 'Generation Diversity', 'Generation Coherence', 'Sample Count']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes.flatten()[idx]
            
            bin_labels = []
            values = []
            
            for bin_name, bin_results in sorted(results["by_quality_bin"].items()):
                if metric in bin_results:
                    bin_labels.append(bin_name.split('_')[1])  # Extract bin number
                    values.append(bin_results[metric])
            
            if values:
                bars = ax.bar(bin_labels, values)
                ax.set_xlabel('Quality Bin')
                ax.set_ylabel(title)
                ax.set_title(f'{title} by Quality Bin')
                
                # Color bars by value
                if metric == 'perplexity':
                    # Lower is better for perplexity
                    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(values)))
                else:
                    # Higher is better for other metrics
                    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(values)))
                
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'quality_bin_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()


def compare_experiments(
    experiment_results: Dict[str, Dict],
    output_dir: Path
):
    """Compare results across different quality/quantity experiments"""
    
    # Create comparison DataFrame
    comparison_data = []
    
    for exp_name, results in experiment_results.items():
        if "overall_metrics" in results:
            row = {
                "experiment": exp_name,
                **results["overall_metrics"]
            }
            
            # Extract experiment details from name
            if "10k" in exp_name:
                row["dataset_size"] = 10000
            elif "100k" in exp_name:
                row["dataset_size"] = 100000
            elif "1M" in exp_name:
                row["dataset_size"] = 1000000
            
            if "ultra" in exp_name:
                row["quality_level"] = "ultra_high"
            elif "high" in exp_name:
                row["quality_level"] = "high"
            elif "medium" in exp_name:
                row["quality_level"] = "medium"
            
            comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Save comparison table
    df.to_csv(output_dir / "experiment_comparison.csv", index=False)
    
    # Create comparison visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Perplexity comparison
    ax = axes[0]
    if "weighted_avg_perplexity" in df.columns:
        df_pivot = df.pivot(index="dataset_size", columns="quality_level", values="weighted_avg_perplexity")
        df_pivot.plot(kind="bar", ax=ax)
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel("Perplexity")
        ax.set_title("Perplexity: Quality vs Quantity")
        ax.set_xscale("log")
    
    # 2. Quality-Quantity tradeoff curve
    ax = axes[1]
    if "dataset_size" in df.columns and "weighted_avg_perplexity" in df.columns:
        for quality in df["quality_level"].unique():
            df_quality = df[df["quality_level"] == quality]
            ax.plot(df_quality["dataset_size"], df_quality["weighted_avg_perplexity"],
                   marker='o', label=quality, linewidth=2, markersize=8)
        
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel("Perplexity")
        ax.set_title("Quality-Quantity Tradeoff")
        ax.set_xscale("log")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "experiment_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


if __name__ == "__main__":
    # Test the evaluator
    print("Quality-Quantity Evaluator Test")
    print("This module is meant to be imported by the training script.")
    print("Run train_quality_comparison.py to use this evaluator.") 