#!/usr/bin/env python3
"""
Generation utilities for Experiment 1: Pure Synthetic Excellence
Generates sample outputs from trained models for qualitative analysis
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union
from datetime import datetime

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


# Sample prompts for generation
DEFAULT_PROMPTS = [
    # Instruction-following
    {
        "category": "instruction",
        "prompt": "### Instruction:\nWrite a Python function to calculate the factorial of a number.\n\n### Response:\n",
        "max_tokens": 200
    },
    {
        "category": "instruction",
        "prompt": "### Instruction:\nExplain photosynthesis in simple terms suitable for a 10-year-old.\n\n### Response:\n",
        "max_tokens": 150
    },
    {
        "category": "instruction", 
        "prompt": "### Instruction:\nList 5 tips for improving sleep quality.\n\n### Response:\n",
        "max_tokens": 200
    },
    
    # Creative writing
    {
        "category": "creative",
        "prompt": "### Instruction:\nWrite a short story about a robot learning to paint.\n\n### Response:\n",
        "max_tokens": 300
    },
    {
        "category": "creative",
        "prompt": "### Instruction:\nCompose a haiku about artificial intelligence.\n\n### Response:\n",
        "max_tokens": 50
    },
    
    # Question answering
    {
        "category": "qa",
        "prompt": "### Instruction:\nWhat are the main differences between machine learning and deep learning?\n\n### Response:\n",
        "max_tokens": 200
    },
    {
        "category": "qa",
        "prompt": "### Instruction:\nWhy is the sky blue?\n\n### Response:\n",
        "max_tokens": 150
    },
    
    # Reasoning
    {
        "category": "reasoning",
        "prompt": "### Instruction:\nIf all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly? Explain your reasoning.\n\n### Response:\n",
        "max_tokens": 150
    },
    {
        "category": "reasoning",
        "prompt": "### Instruction:\nA farmer has 17 sheep. All but 9 die. How many are left? Explain step by step.\n\n### Response:\n",
        "max_tokens": 100
    },
    
    # Summarization
    {
        "category": "summarization",
        "prompt": "### Instruction:\nSummarize the key points of climate change in 3 bullet points.\n\n### Response:\n",
        "max_tokens": 150
    },
    
    # Code generation
    {
        "category": "code",
        "prompt": "### Instruction:\nWrite a JavaScript function to reverse a string without using the built-in reverse method.\n\n### Response:\n",
        "max_tokens": 150
    },
    
    # Math
    {
        "category": "math",
        "prompt": "### Instruction:\nSolve this equation step by step: 2x + 5 = 13\n\n### Response:\n",
        "max_tokens": 100
    }
]


class SampleGenerator:
    """Generate samples from a trained model"""
    
    def __init__(
        self,
        model: GPT2LMHeadModel,
        tokenizer: GPT2TokenizerFast,
        device: str = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
    
    def generate_sample(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 50,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.1
    ) -> List[str]:
        """Generate text from a prompt"""
        
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048 - max_new_tokens
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=do_sample,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode outputs
        generated_texts = []
        for output in outputs:
            # Skip the prompt tokens
            generated_tokens = output[inputs["input_ids"].shape[1]:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            generated_texts.append(generated_text)
        
        return generated_texts
    
    def generate_samples_from_prompts(
        self,
        prompts: List[Dict],
        temperature_range: tuple = (0.7, 1.0),
        num_variations: int = 3
    ) -> List[Dict]:
        """Generate samples from a list of prompts with variations"""
        
        all_samples = []
        
        for prompt_info in tqdm(prompts, desc="Generating samples"):
            prompt = prompt_info["prompt"]
            max_tokens = prompt_info.get("max_tokens", 200)
            category = prompt_info.get("category", "general")
            
            # Generate with different temperatures
            temperatures = np.linspace(temperature_range[0], temperature_range[1], num_variations)
            
            for temp in temperatures:
                generated = self.generate_sample(
                    prompt,
                    max_new_tokens=max_tokens,
                    temperature=temp,
                    top_p=0.9,
                    do_sample=True
                )
                
                sample = {
                    "prompt": prompt,
                    "generated": generated[0],
                    "temperature": float(temp),
                    "category": category,
                    "timestamp": datetime.now().isoformat()
                }
                
                all_samples.append(sample)
        
        return all_samples
    
    def analyze_generation_quality(self, samples: List[Dict]) -> Dict:
        """Analyze the quality of generated samples"""
        
        analysis = {
            "total_samples": len(samples),
            "by_category": {},
            "length_stats": {},
            "quality_indicators": {}
        }
        
        # Group by category
        categories = {}
        for sample in samples:
            cat = sample["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(sample)
        
        # Analyze each category
        for cat, cat_samples in categories.items():
            lengths = [len(s["generated"].split()) for s in cat_samples]
            
            analysis["by_category"][cat] = {
                "count": len(cat_samples),
                "avg_length": np.mean(lengths),
                "std_length": np.std(lengths),
                "min_length": np.min(lengths),
                "max_length": np.max(lengths)
            }
            
            # Check for quality indicators
            quality_checks = {
                "has_code_blocks": 0,
                "has_bullet_points": 0,
                "has_numbers": 0,
                "ends_properly": 0,
                "repetitive": 0
            }
            
            for sample in cat_samples:
                text = sample["generated"]
                
                # Check for code blocks
                if "```" in text or "def " in text or "function " in text:
                    quality_checks["has_code_blocks"] += 1
                
                # Check for bullet points
                if "•" in text or "- " in text or "* " in text or "\n1." in text:
                    quality_checks["has_bullet_points"] += 1
                
                # Check for numbers
                if any(char.isdigit() for char in text):
                    quality_checks["has_numbers"] += 1
                
                # Check if ends properly (with punctuation)
                if text.strip() and text.strip()[-1] in ".!?":
                    quality_checks["ends_properly"] += 1
                
                # Check for repetition
                words = text.lower().split()
                if len(words) > 10:
                    unique_ratio = len(set(words)) / len(words)
                    if unique_ratio < 0.5:
                        quality_checks["repetitive"] += 1
            
            # Convert to percentages
            for check, count in quality_checks.items():
                quality_checks[check] = count / len(cat_samples) * 100
            
            analysis["by_category"][cat]["quality_indicators"] = quality_checks
        
        # Overall length statistics
        all_lengths = [len(s["generated"].split()) for s in samples]
        analysis["length_stats"] = {
            "mean": np.mean(all_lengths),
            "std": np.std(all_lengths),
            "min": np.min(all_lengths),
            "max": np.max(all_lengths),
            "median": np.median(all_lengths)
        }
        
        return analysis


def generate_samples(
    model: Union[GPT2LMHeadModel, str],
    tokenizer: Union[GPT2TokenizerFast, str],
    prompts: Optional[List[Dict]] = None,
    num_samples: int = 100,
    output_dir: Optional[Union[str, Path]] = None,
    save_html: bool = True
) -> Dict:
    """
    Generate samples from a model
    
    Args:
        model: Model or path to model
        tokenizer: Tokenizer or path to tokenizer
        prompts: List of prompts to use (uses defaults if None)
        num_samples: Total number of samples to generate
        output_dir: Directory to save outputs
        save_html: Whether to save an HTML visualization
        
    Returns:
        Dictionary with samples and analysis
    """
    
    # Load model and tokenizer if paths provided
    if isinstance(model, str):
        logger.info(f"Loading model from {model}")
        model = GPT2LMHeadModel.from_pretrained(model)
    if isinstance(tokenizer, str):
        logger.info(f"Loading tokenizer from {tokenizer}")
        tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer)
    
    # Use default prompts if none provided
    if prompts is None:
        prompts = DEFAULT_PROMPTS
        # Adjust number of variations based on desired total samples
        num_variations = max(1, num_samples // len(prompts))
    else:
        num_variations = 3
    
    # Create generator
    generator = SampleGenerator(model, tokenizer)
    
    # Generate samples
    logger.info(f"Generating {len(prompts) * num_variations} samples...")
    samples = generator.generate_samples_from_prompts(
        prompts,
        temperature_range=(0.7, 1.0),
        num_variations=num_variations
    )
    
    # Analyze samples
    logger.info("Analyzing generation quality...")
    analysis = generator.analyze_generation_quality(samples)
    
    # Prepare results
    results = {
        "samples": samples,
        "analysis": analysis,
        "metadata": {
            "model_type": model.config.model_type,
            "vocab_size": model.config.vocab_size,
            "total_params": sum(p.numel() for p in model.parameters()),
            "timestamp": datetime.now().isoformat()
        }
    }
    
    # Save results
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(output_dir / "generation_samples.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save readable text
        with open(output_dir / "generation_samples.txt", "w") as f:
            f.write("Generated Samples\n")
            f.write("=" * 80 + "\n\n")
            
            for i, sample in enumerate(samples):
                f.write(f"Sample {i+1}\n")
                f.write(f"Category: {sample['category']}\n")
                f.write(f"Temperature: {sample['temperature']:.2f}\n")
                f.write(f"Prompt: {sample['prompt']}\n")
                f.write(f"Generated:\n{sample['generated']}\n")
                f.write("-" * 80 + "\n\n")
        
        # Save HTML visualization
        if save_html:
            save_html_visualization(samples, analysis, output_dir / "generation_samples.html")
        
        logger.info(f"Results saved to {output_dir}")
    
    return results


def save_html_visualization(samples: List[Dict], analysis: Dict, output_path: Path):
    """Save an HTML visualization of the samples"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Generation Samples</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; }
            .sample { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .prompt { color: #2c3e50; font-weight: bold; margin-bottom: 10px; }
            .generated { color: #34495e; white-space: pre-wrap; background: #ecf0f1; padding: 15px; border-radius: 4px; }
            .metadata { color: #7f8c8d; font-size: 0.9em; margin-top: 10px; }
            .category { display: inline-block; background: #3498db; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; }
            .temperature { display: inline-block; background: #e74c3c; color: white; padding: 4px 8px; border-radius: 4px; font-size: 0.8em; margin-left: 10px; }
            .analysis { background: white; padding: 20px; margin: 20px 0; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            h1, h2 { color: #2c3e50; }
            table { width: 100%; border-collapse: collapse; margin: 10px 0; }
            th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
            th { background-color: #3498db; color: white; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Generation Samples Analysis</h1>
            
            <div class="analysis">
                <h2>Summary Statistics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    <tr>
                        <td>Total Samples</td>
                        <td>{total_samples}</td>
                    </tr>
                    <tr>
                        <td>Average Length (words)</td>
                        <td>{avg_length:.1f}</td>
                    </tr>
                    <tr>
                        <td>Categories</td>
                        <td>{num_categories}</td>
                    </tr>
                </table>
            </div>
            
            <h2>Generated Samples</h2>
    """.format(
        total_samples=analysis["total_samples"],
        avg_length=analysis["length_stats"]["mean"],
        num_categories=len(analysis["by_category"])
    )
    
    # Add samples
    for i, sample in enumerate(samples):
        html_content += f"""
            <div class="sample">
                <div class="metadata">
                    <span class="category">{sample['category']}</span>
                    <span class="temperature">temp: {sample['temperature']:.2f}</span>
                </div>
                <div class="prompt">Prompt: {sample['prompt']}</div>
                <div class="generated">{sample['generated']}</div>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate samples from a trained model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--output_dir", type=str, default="./generation_results", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to generate")
    parser.add_argument("--custom_prompts", type=str, help="Path to JSON file with custom prompts")
    
    args = parser.parse_args()
    
    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load custom prompts if provided
    prompts = None
    if args.custom_prompts:
        with open(args.custom_prompts, "r") as f:
            prompts = json.load(f)
    
    # Generate samples
    results = generate_samples(
        args.model_path,
        tokenizer,
        prompts=prompts,
        num_samples=args.num_samples,
        output_dir=args.output_dir
    )
    
    # Print summary
    print(f"\nGenerated {len(results['samples'])} samples")
    print(f"Average length: {results['analysis']['length_stats']['mean']:.1f} words")
    print(f"Results saved to: {args.output_dir}") 