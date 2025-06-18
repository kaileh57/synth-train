#!/usr/bin/env python3
"""
Analyze generation quality for models trained with different quality/quantity tradeoffs
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import textstat
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logger = logging.getLogger(__name__)


class GenerationAnalyzer:
    """Analyze the quality of generated text"""
    
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
        
        # Test prompts for different capabilities
        self.test_prompts = self._get_test_prompts()
    
    def _get_test_prompts(self) -> List[Dict]:
        """Get diverse test prompts"""
        return [
            # Instruction following
            {
                "category": "instruction",
                "prompt": "### Instruction:\nWrite a Python function that checks if a number is prime.\n\n### Response:\n",
                "expected_elements": ["def", "prime", "return", "for", "if"],
                "max_tokens": 200
            },
            {
                "category": "instruction",
                "prompt": "### Instruction:\nList three benefits of regular exercise.\n\n### Response:\n",
                "expected_elements": ["health", "improve", "reduce", "benefit"],
                "max_tokens": 150
            },
            
            # Reasoning
            {
                "category": "reasoning",
                "prompt": "### Instruction:\nIf it takes 5 machines 5 minutes to make 5 widgets, how long does it take 100 machines to make 100 widgets? Explain your reasoning.\n\n### Response:\n",
                "expected_elements": ["5 minutes", "rate", "same", "each machine"],
                "max_tokens": 150
            },
            
            # Creative
            {
                "category": "creative",
                "prompt": "### Instruction:\nWrite a creative description of a sunset over the ocean.\n\n### Response:\n",
                "expected_elements": ["sun", "ocean", "color", "sky", "water"],
                "max_tokens": 200
            },
            
            # Factual QA
            {
                "category": "factual",
                "prompt": "### Instruction:\nWhat is photosynthesis and why is it important?\n\n### Response:\n",
                "expected_elements": ["plants", "light", "oxygen", "carbon dioxide", "energy"],
                "max_tokens": 150
            },
            
            # Complex instruction
            {
                "category": "complex",
                "prompt": "### Instruction:\nCreate a step-by-step plan for learning a new programming language, including resources and milestones.\n\n### Response:\n",
                "expected_elements": ["step", "learn", "practice", "project", "resource"],
                "max_tokens": 250
            }
        ]
    
    def analyze_generations(
        self,
        num_samples_per_prompt: int = 5,
        temperature_range: Tuple[float, float] = (0.7, 1.0)
    ) -> Dict:
        """Analyze model generations across different prompts and settings"""
        
        results = {
            "by_category": {},
            "overall_metrics": {},
            "detailed_samples": []
        }
        
        temperatures = np.linspace(temperature_range[0], temperature_range[1], num_samples_per_prompt)
        
        for prompt_info in tqdm(self.test_prompts, desc="Analyzing prompts"):
            category = prompt_info["category"]
            
            if category not in results["by_category"]:
                results["by_category"][category] = {
                    "samples": [],
                    "metrics": {}
                }
            
            # Generate samples at different temperatures
            for temp in temperatures:
                sample_result = self._analyze_single_generation(prompt_info, temperature=temp)
                results["by_category"][category]["samples"].append(sample_result)
                results["detailed_samples"].append(sample_result)
        
        # Calculate aggregate metrics
        for category in results["by_category"]:
            results["by_category"][category]["metrics"] = self._calculate_category_metrics(
                results["by_category"][category]["samples"]
            )
        
        # Overall metrics
        results["overall_metrics"] = self._calculate_overall_metrics(results["detailed_samples"])
        
        return results
    
    def _analyze_single_generation(
        self,
        prompt_info: Dict,
        temperature: float = 0.8
    ) -> Dict:
        """Analyze a single generation"""
        
        # Generate text
        prompt = prompt_info["prompt"]
        max_tokens = prompt_info.get("max_tokens", 200)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - max_tokens
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        )
        
        # Analyze the generation
        analysis = {
            "category": prompt_info["category"],
            "prompt": prompt,
            "generated": generated_text,
            "temperature": temperature,
            "metrics": {}
        }
        
        # Length metrics
        analysis["metrics"]["length_chars"] = len(generated_text)
        analysis["metrics"]["length_words"] = len(generated_text.split())
        analysis["metrics"]["length_sentences"] = len(sent_tokenize(generated_text))
        
        # Readability
        try:
            analysis["metrics"]["flesch_reading_ease"] = textstat.flesch_reading_ease(generated_text)
            analysis["metrics"]["flesch_kincaid_grade"] = textstat.flesch_kincaid_grade(generated_text)
        except:
            analysis["metrics"]["flesch_reading_ease"] = None
            analysis["metrics"]["flesch_kincaid_grade"] = None
        
        # Diversity
        tokens = word_tokenize(generated_text.lower())
        if tokens:
            analysis["metrics"]["vocabulary_diversity"] = len(set(tokens)) / len(tokens)
        else:
            analysis["metrics"]["vocabulary_diversity"] = 0
        
        # Check for expected elements
        expected = prompt_info.get("expected_elements", [])
        if expected:
            found_count = sum(1 for elem in expected if elem.lower() in generated_text.lower())
            analysis["metrics"]["expected_elements_ratio"] = found_count / len(expected)
        else:
            analysis["metrics"]["expected_elements_ratio"] = 1.0
        
        # Structure analysis
        analysis["metrics"]["has_code_block"] = "```" in generated_text or "def " in generated_text
        analysis["metrics"]["has_list"] = bool(re.search(r'^\s*[\d\-\*]\s*', generated_text, re.MULTILINE))
        analysis["metrics"]["has_paragraphs"] = "\n\n" in generated_text
        
        # Coherence indicators
        analysis["metrics"]["ends_properly"] = bool(generated_text.strip() and 
                                                   generated_text.strip()[-1] in '.!?')
        analysis["metrics"]["incomplete_sentence"] = generated_text.strip().endswith((',', ';', 'and', 'or'))
        
        # Repetition check
        sentences = sent_tokenize(generated_text)
        if len(sentences) > 1:
            # Check for repeated sentences
            unique_sentences = len(set(sentences))
            analysis["metrics"]["sentence_repetition"] = 1 - (unique_sentences / len(sentences))
        else:
            analysis["metrics"]["sentence_repetition"] = 0
        
        # N-gram repetition
        if len(tokens) > 3:
            trigrams = [tuple(tokens[i:i+3]) for i in range(len(tokens)-2)]
            if trigrams:
                analysis["metrics"]["trigram_repetition"] = 1 - (len(set(trigrams)) / len(trigrams))
            else:
                analysis["metrics"]["trigram_repetition"] = 0
        else:
            analysis["metrics"]["trigram_repetition"] = 0
        
        return analysis
    
    def _calculate_category_metrics(self, samples: List[Dict]) -> Dict:
        """Calculate aggregate metrics for a category"""
        
        if not samples:
            return {}
        
        metrics = {}
        
        # Collect all metric values
        metric_values = {}
        for sample in samples:
            for metric_name, value in sample["metrics"].items():
                if value is not None:
                    if metric_name not in metric_values:
                        metric_values[metric_name] = []
                    metric_values[metric_name].append(value)
        
        # Calculate statistics
        for metric_name, values in metric_values.items():
            if values:
                metrics[f"{metric_name}_mean"] = np.mean(values)
                metrics[f"{metric_name}_std"] = np.std(values)
                
                # For boolean metrics, mean is the ratio
                if all(isinstance(v, bool) or v in [0, 1] for v in values):
                    metrics[f"{metric_name}_ratio"] = np.mean(values)
        
        return metrics
    
    def _calculate_overall_metrics(self, all_samples: List[Dict]) -> Dict:
        """Calculate overall metrics across all samples"""
        
        metrics = {
            "total_samples": len(all_samples),
            "avg_length_words": np.mean([s["metrics"]["length_words"] for s in all_samples]),
            "avg_vocabulary_diversity": np.mean([s["metrics"]["vocabulary_diversity"] for s in all_samples]),
            "coherence_rate": np.mean([s["metrics"]["ends_properly"] for s in all_samples]),
            "expected_elements_coverage": np.mean([s["metrics"]["expected_elements_ratio"] for s in all_samples])
        }
        
        # Category distribution
        category_counts = Counter(s["category"] for s in all_samples)
        metrics["samples_per_category"] = dict(category_counts)
        
        # Temperature impact
        temp_groups = {}
        for sample in all_samples:
            temp = round(sample["temperature"], 2)
            if temp not in temp_groups:
                temp_groups[temp] = []
            temp_groups[temp].append(sample["metrics"]["vocabulary_diversity"])
        
        metrics["diversity_by_temperature"] = {
            temp: np.mean(diversities) for temp, diversities in temp_groups.items()
        }
        
        return metrics
    
    def create_visualizations(self, results: Dict, output_dir: Path):
        """Create visualizations for generation analysis"""
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Quality metrics by category
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Generation Quality Metrics by Category', fontsize=16)
        
        categories = list(results["by_category"].keys())
        
        # Vocabulary diversity
        ax = axes[0, 0]
        diversity_means = [results["by_category"][cat]["metrics"].get("vocabulary_diversity_mean", 0) 
                          for cat in categories]
        diversity_stds = [results["by_category"][cat]["metrics"].get("vocabulary_diversity_std", 0) 
                         for cat in categories]
        ax.bar(categories, diversity_means, yerr=diversity_stds, capsize=5)
        ax.set_ylabel('Vocabulary Diversity')
        ax.set_title('Vocabulary Diversity by Task Category')
        ax.tick_params(axis='x', rotation=45)
        
        # Expected elements coverage
        ax = axes[0, 1]
        coverage_means = [results["by_category"][cat]["metrics"].get("expected_elements_ratio_mean", 0) 
                         for cat in categories]
        ax.bar(categories, coverage_means, color='green', alpha=0.7)
        ax.set_ylabel('Expected Elements Coverage')
        ax.set_title('Task Completion Rate')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1.1)
        
        # Coherence rate
        ax = axes[1, 0]
        coherence_rates = [results["by_category"][cat]["metrics"].get("ends_properly_ratio", 0) 
                          for cat in categories]
        ax.bar(categories, coherence_rates, color='blue', alpha=0.7)
        ax.set_ylabel('Coherence Rate')
        ax.set_title('Properly Ended Responses')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0, 1.1)
        
        # Average length
        ax = axes[1, 1]
        length_means = [results["by_category"][cat]["metrics"].get("length_words_mean", 0) 
                       for cat in categories]
        length_stds = [results["by_category"][cat]["metrics"].get("length_words_std", 0) 
                      for cat in categories]
        ax.bar(categories, length_means, yerr=length_stds, capsize=5, color='orange', alpha=0.7)
        ax.set_ylabel('Average Length (words)')
        ax.set_title('Response Length by Category')
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'generation_quality_by_category.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Temperature impact
        if "diversity_by_temperature" in results["overall_metrics"]:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            temps = sorted(results["overall_metrics"]["diversity_by_temperature"].keys())
            diversities = [results["overall_metrics"]["diversity_by_temperature"][t] for t in temps]
            
            ax.plot(temps, diversities, marker='o', linewidth=2, markersize=8)
            ax.set_xlabel('Temperature')
            ax.set_ylabel('Vocabulary Diversity')
            ax.set_title('Impact of Temperature on Generation Diversity')
            ax.grid(True, alpha=0.3)
            
            plt.savefig(output_dir / 'temperature_impact.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Sample quality distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        
        all_scores = []
        for sample in results["detailed_samples"]:
            # Create composite quality score
            score = (
                sample["metrics"]["vocabulary_diversity"] * 0.3 +
                sample["metrics"]["expected_elements_ratio"] * 0.4 +
                sample["metrics"]["ends_properly"] * 0.3
            )
            all_scores.append(score)
        
        ax.hist(all_scores, bins=30, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Composite Quality Score')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Generation Quality Scores')
        ax.axvline(np.mean(all_scores), color='red', linestyle='--', 
                  label=f'Mean: {np.mean(all_scores):.3f}')
        ax.legend()
        
        plt.savefig(output_dir / 'quality_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()


def analyze_generation_quality(
    model: GPT2LMHeadModel,
    tokenizer: GPT2TokenizerFast,
    num_samples: int = 50,
    output_dir: Optional[Path] = None
) -> Dict:
    """Main function to analyze generation quality"""
    
    analyzer = GenerationAnalyzer(model, tokenizer)
    
    # Calculate samples per prompt
    num_prompts = len(analyzer.test_prompts)
    samples_per_prompt = max(1, num_samples // num_prompts)
    
    # Run analysis
    logger.info(f"Analyzing {samples_per_prompt} samples per prompt across {num_prompts} prompts")
    results = analyzer.analyze_generations(num_samples_per_prompt=samples_per_prompt)
    
    # Save results and create visualizations
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        with open(output_dir / "generation_analysis_results.json", "w") as f:
            # Convert numpy types for JSON serialization
            json_results = json.loads(json.dumps(results, default=lambda x: float(x) if isinstance(x, np.number) else x))
            json.dump(json_results, f, indent=2)
        
        # Save sample outputs
        with open(output_dir / "generation_samples.txt", "w", encoding="utf-8") as f:
            for i, sample in enumerate(results["detailed_samples"][:20]):  # First 20 samples
                f.write(f"Sample {i+1}\n")
                f.write(f"Category: {sample['category']}\n")
                f.write(f"Temperature: {sample['temperature']:.2f}\n")
                f.write(f"Prompt: {sample['prompt']}\n")
                f.write(f"Generated:\n{sample['generated']}\n")
                f.write(f"Metrics: {json.dumps(sample['metrics'], indent=2)}\n")
                f.write("-" * 80 + "\n\n")
        
        # Create visualizations
        analyzer.create_visualizations(results, output_dir)
        
        logger.info(f"Results saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    # Test the analyzer
    print("Generation Quality Analyzer")
    print("This module is meant to be imported by the training script.")
    print("Run train_quality_comparison.py to use this analyzer.") 