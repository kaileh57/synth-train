# Experiment 3: Optimal Dataset Mixing - Instructions

## Overview
This experiment explores how to optimally combine different synthetic datasets to maximize model capabilities. We'll test various mixing strategies to find the best balance between instruction-following, knowledge, and conversational abilities.

## Hypothesis
Carefully mixed synthetic datasets can produce models with broader capabilities than any single dataset, and there exists an optimal mixing ratio that maximizes performance across diverse tasks.

## Objectives
1. Test 6 different mixing strategies across 4 high-quality synthetic datasets
2. Identify which datasets contribute which capabilities
3. Find optimal mixing ratios for balanced performance
4. Develop automated mixing optimization methods

## Datasets to Mix

### 1. OpenHermes-2.5
- **Strength**: Instruction-following, task completion
- **Size**: 1M samples
- **Source**: GPT-4 generated
- **Use case**: General assistant capabilities

### 2. Cosmopedia
- **Strength**: Educational content, factual knowledge
- **Size**: 25B tokens
- **Source**: Mixtral-8x7B generated
- **Use case**: Knowledge and explanations

### 3. Magpie-Pro
- **Strength**: Natural conversations, reasoning
- **Size**: 500K samples
- **Source**: Llama-3.1-70B self-generated
- **Use case**: Conversational AI

### 4. FineWeb-Edu
- **Strength**: High-quality educational web content
- **Size**: 1.3T tokens (filtered)
- **Source**: Web data filtered by Llama-70B
- **Use case**: Diverse real-world knowledge

## Mixing Strategies to Test

### Strategy 1: Equal Mix (Baseline)
```python
mixing_ratios = {
    'openhermes': 0.25,
    'cosmopedia': 0.25,
    'magpie': 0.25,
    'fineweb': 0.25
}
```

### Strategy 2: Instruction-Heavy
```python
mixing_ratios = {
    'openhermes': 0.40,  # Focus on instructions
    'cosmopedia': 0.20,
    'magpie': 0.30,
    'fineweb': 0.10
}
```

### Strategy 3: Knowledge-Heavy
```python
mixing_ratios = {
    'openhermes': 0.20,
    'cosmopedia': 0.40,  # Focus on knowledge
    'magpie': 0.10,
    'fineweb': 0.30
}
```

### Strategy 4: Conversation-Heavy
```python
mixing_ratios = {
    'openhermes': 0.20,
    'cosmopedia': 0.10,
    'magpie': 0.50,  # Focus on conversation
    'fineweb': 0.20
}
```

### Strategy 5: Quality-Weighted
```python
# Based on Exp2 quality scores
mixing_ratios = {
    'openhermes': 0.35,  # Highest quality
    'cosmopedia': 0.25,
    'magpie': 0.30,      # High quality
    'fineweb': 0.10      # Lower synthetic quality
}
```

### Strategy 6: Capability-Balanced
```python
# Optimized for diverse capabilities
mixing_ratios = {
    'openhermes': 0.30,  # Instructions
    'cosmopedia': 0.25,  # Knowledge
    'magpie': 0.25,      # Reasoning
    'fineweb': 0.20      # Diversity
}
```

## Implementation Steps

### Step 1: Data Preparation Pipeline

```python
# data_mixer.py

import numpy as np
from datasets import load_dataset, concatenate_datasets
from typing import Dict, List
import json

class DatasetMixer:
    def __init__(self, max_samples_per_dataset=250000):
        self.max_samples = max_samples_per_dataset
        self.datasets = {}
        self.dataset_stats = {}
        
    def load_datasets(self):
        """Load all datasets with streaming for efficiency."""
        
        print("Loading OpenHermes-2.5...")
        self.datasets['openhermes'] = load_dataset(
            "teknium/OpenHermes-2.5", 
            split="train",
            streaming=True
        )
        
        print("Loading Cosmopedia sample...")
        self.datasets['cosmopedia'] = load_dataset(
            "HuggingFaceTB/cosmopedia",
            split="train",
            streaming=True
        )
        
        print("Loading Magpie-Pro...")
        self.datasets['magpie'] = load_dataset(
            "Magpie-Align/MagpieLM-Pro-300K-v0.1",
            split="train",
            streaming=True
        )
        
        print("Loading FineWeb-Edu sample...")
        self.datasets['fineweb'] = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            split="train",
            streaming=True
        )
        
    def analyze_datasets(self):
        """Analyze dataset characteristics."""
        
        for name, dataset in self.datasets.items():
            print(f"\nAnalyzing {name}...")
            
            samples = list(dataset.take(1000))
            
            # Calculate statistics
            lengths = [len(str(sample).split()) for sample in samples]
            
            self.dataset_stats[name] = {
                'avg_length': np.mean(lengths),
                'std_length': np.std(lengths),
                'sample_fields': list(samples[0].keys()),
                'sample_count': len(samples)
            }
            
            print(f"  Average length: {self.dataset_stats[name]['avg_length']:.0f} words")
            print(f"  Fields: {self.dataset_stats[name]['sample_fields']}")
    
    def normalize_format(self, sample, dataset_name):
        """Convert all datasets to common format."""
        
        if dataset_name == 'openhermes':
            # Already in instruction-response format
            return {
                'text': f"### Instruction: {sample['instruction']}\n### Response: {sample['response']}",
                'source': 'openhermes',
                'type': 'instruction'
            }
            
        elif dataset_name == 'cosmopedia':
            # Educational content
            return {
                'text': sample['text'],
                'source': 'cosmopedia',
                'type': 'educational'
            }
            
        elif dataset_name == 'magpie':
            # Conversation format
            conversation = sample['conversation']
            text = '\n'.join([f"{turn['role']}: {turn['content']}" 
                            for turn in conversation])
            return {
                'text': text,
                'source': 'magpie',
                'type': 'conversation'
            }
            
        elif dataset_name == 'fineweb':
            # Web content
            return {
                'text': sample['text'],
                'source': 'fineweb',
                'type': 'web_educational'
            }
    
    def create_mixed_dataset(self, mixing_ratios: Dict[str, float], 
                           total_samples: int = 1000000):
        """Create mixed dataset according to ratios."""
        
        # Calculate samples per dataset
        samples_per_dataset = {
            name: int(ratio * total_samples)
            for name, ratio in mixing_ratios.items()
        }
        
        print(f"\nCreating mixed dataset with {total_samples} total samples:")
        for name, count in samples_per_dataset.items():
            print(f"  {name}: {count} samples ({mixing_ratios[name]*100:.0f}%)")
        
        # Collect samples
        mixed_samples = []
        
        for dataset_name, num_samples in samples_per_dataset.items():
            dataset = self.datasets[dataset_name]
            
            # Take samples
            samples = list(dataset.take(num_samples))
            
            # Normalize format
            normalized = [
                self.normalize_format(s, dataset_name) 
                for s in samples
            ]
            
            mixed_samples.extend(normalized)
            print(f"  Added {len(normalized)} samples from {dataset_name}")
        
        # Shuffle
        np.random.shuffle(mixed_samples)
        
        return mixed_samples
```

### Step 2: Training Configuration

```python
# mixing_experiments.py

def run_mixing_experiment(strategy_name: str, mixing_ratios: Dict[str, float]):
    """Run a complete mixing experiment."""
    
    print(f"\n{'='*60}")
    print(f"Running Mixing Strategy: {strategy_name}")
    print(f"{'='*60}")
    
    # 1. Create mixed dataset
    mixer = DatasetMixer()
    mixer.load_datasets()
    mixed_dataset = mixer.create_mixed_dataset(mixing_ratios)
    
    # 2. Save dataset info
    dataset_info = {
        'strategy': strategy_name,
        'mixing_ratios': mixing_ratios,
        'total_samples': len(mixed_dataset),
        'samples_per_source': dict(Counter([s['source'] for s in mixed_dataset]))
    }
    
    with open(f'data/exp3_{strategy_name}_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # 3. Prepare for training
    from datasets import Dataset
    dataset = Dataset.from_list(mixed_dataset)
    
    # Split
    dataset = dataset.train_test_split(test_size=0.05, seed=42)
    
    # 4. Initialize model
    model = GPT2LMHeadModel(GPT2Config(**config_500M))
    
    # 5. Training arguments
    training_args = TrainingArguments(
        output_dir=f"./models/exp3-{strategy_name}",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=1000,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_steps=100,
        save_steps=5000,
        eval_steps=1000,
        evaluation_strategy="steps",
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        report_to="wandb",
        run_name=f"exp3-{strategy_name}",
    )
    
    # 6. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    train_result = trainer.train()
    
    # 7. Save model
    model.save_pretrained(f"models/exp3-{strategy_name}")
    
    return model, train_result
```

### Step 3: Capability-Specific Evaluation

```python
# capability_evaluation.py

class CapabilityEvaluator:
    def __init__(self):
        self.test_suites = {
            'instruction_following': self.load_instruction_tests(),
            'factual_knowledge': self.load_knowledge_tests(),
            'reasoning': self.load_reasoning_tests(),
            'conversation': self.load_conversation_tests(),
            'creativity': self.load_creativity_tests()
        }
    
    def load_instruction_tests(self):
        """Tests for instruction-following capability."""
        return [
            {
                'prompt': "Write a Python function to calculate factorial.",
                'check': lambda r: 'def' in r and 'factorial' in r
            },
            {
                'prompt': "List 5 tips for better sleep.",
                'check': lambda r: r.count('\n') >= 4 or r.count('•') >= 4
            },
            {
                'prompt': "Explain photosynthesis in simple terms.",
                'check': lambda r: 'light' in r.lower() and 'plant' in r.lower()
            }
        ]
    
    def load_knowledge_tests(self):
        """Tests for factual knowledge."""
        return [
            {
                'prompt': "What is the capital of France?",
                'check': lambda r: 'paris' in r.lower()
            },
            {
                'prompt': "When did World War II end?",
                'check': lambda r: '1945' in r
            },
            {
                'prompt': "What is the speed of light?",
                'check': lambda r: '299' in r or '300' in r or '3×10' in r
            }
        ]
    
    def load_reasoning_tests(self):
        """Tests for reasoning ability."""
        return [
            {
                'prompt': "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",
                'check': lambda r: 'no' in r.lower() or 'cannot' in r.lower()
            },
            {
                'prompt': "I have 3 apples. I eat 1 and buy 2 more. How many do I have?",
                'check': lambda r: '4' in r
            }
        ]
    
    def evaluate_all_capabilities(self, model, tokenizer):
        """Evaluate model on all capability tests."""
        
        results = {}
        
        for capability, tests in self.test_suites.items():
            print(f"\nEvaluating {capability}...")
            
            correct = 0
            total = len(tests)
            
            for test in tests:
                # Generate response
                inputs = tokenizer(test['prompt'], return_tensors='pt')
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(test['prompt']):]  # Remove prompt
                
                # Check response
                if test['check'](response):
                    correct += 1
                
            results[capability] = correct / total
            print(f"  Score: {results[capability]:.2%}")
        
        return results

def compare_mixing_strategies(all_results):
    """Compare and visualize results across strategies."""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare data for visualization
    strategies = list(all_results.keys())
    capabilities = list(next(iter(all_results.values())).keys())
    
    # Create heatmap
    data = []
    for strategy in strategies:
        row = [all_results[strategy][cap] for cap in capabilities]
        data.append(row)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        data,
        xticklabels=capabilities,
        yticklabels=strategies,
        annot=True,
        fmt='.2f',
        cmap='YlOrRd',
        cbar_kws={'label': 'Performance Score'}
    )
    
    plt.title('Capability Scores by Mixing Strategy', fontsize=14)
    plt.xlabel('Capability', fontsize=12)
    plt.ylabel('Mixing Strategy', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('figures/exp3_capability_heatmap.png', dpi=300)
    
    # Radar chart for top strategies
    plot_radar_comparison(all_results)
```

### Step 4: Automated Mixing Optimization

```python
# mixing_optimizer.py

class MixingOptimizer:
    def __init__(self, datasets, eval_func):
        self.datasets = datasets
        self.eval_func = eval_func
        self.history = []
        
    def objective(self, ratios):
        """Objective function for optimization."""
        
        # Normalize ratios to sum to 1
        ratios = np.array(ratios)
        ratios = ratios / ratios.sum()
        
        # Create mixing dict
        mixing_ratios = {
            name: float(ratio) 
            for name, ratio in zip(self.datasets, ratios)
        }
        
        # Train small model with these ratios
        model = self.train_small_model(mixing_ratios, n_samples=10000)
        
        # Evaluate
        score = self.eval_func(model)
        
        # Store history
        self.history.append({
            'ratios': mixing_ratios,
            'score': score
        })
        
        return -score  # Minimize negative score
    
    def optimize(self, n_trials=20):
        """Find optimal mixing ratios."""
        
        from scipy.optimize import differential_evolution
        
        # Bounds: each ratio between 0.1 and 0.6
        bounds = [(0.1, 0.6) for _ in self.datasets]
        
        # Optimize
        result = differential_evolution(
            self.objective,
            bounds,
            maxiter=n_trials,
            seed=42
        )
        
        # Get best ratios
        best_ratios = result.x / result.x.sum()
        
        return {
            name: float(ratio)
            for name, ratio in zip(self.datasets, best_ratios)
        }
```

### Step 5: Analysis and Insights

```python
# analyze_mixing_results.py

def analyze_dataset_contributions():
    """Determine what each dataset contributes."""
    
    # Load all trained models
    models = {}
    for strategy in strategies:
        models[strategy] = load_model(f"models/exp3-{strategy}")
    
    # Ablation study: remove one dataset at a time
    ablation_results = {}
    
    for dataset_to_remove in ['openhermes', 'cosmopedia', 'magpie', 'fineweb']:
        # Create mixing without this dataset
        ablation_ratios = {
            name: 0.33 if name != dataset_to_remove else 0.0
            for name in all_datasets
        }
        
        # Train model
        model = train_with_mixing(ablation_ratios, name=f"ablation_no_{dataset_to_remove}")
        
        # Evaluate
        results = evaluate_all_capabilities(model)
        ablation_results[dataset_to_remove] = results
    
    # Analyze contribution
    contributions = {}
    baseline = all_results['equal_mix']
    
    for dataset, ablation in ablation_results.items():
        contribution = {}
        for capability in capabilities:
            # How much does performance drop without this dataset?
            drop = baseline[capability] - ablation[capability]
            contribution[capability] = drop
        
        contributions[dataset] = contribution
    
    return contributions

def generate_mixing_recommendations():
    """Generate practical recommendations."""
    
    recommendations = []
    
    # Find best overall strategy
    overall_scores = {
        strategy: np.mean(list(results.values()))
        for strategy, results in all_results.items()
    }
    
    best_strategy = max(overall_scores, key=overall_scores.get)
    recommendations.append(
        f"Best overall: {best_strategy} (score: {overall_scores[best_strategy]:.3f})"
    )
    
    # Find best strategy for each capability
    for capability in capabilities:
        cap_scores = {
            strategy: results[capability]
            for strategy, results in all_results.items()
        }
        best = max(cap_scores, key=cap_scores.get)
        recommendations.append(
            f"Best for {capability}: {best} (score: {cap_scores[best]:.3f})"
        )
    
    # Dataset-specific insights
    contributions = analyze_dataset_contributions()
    
    for dataset, contrib in contributions.items():
        main_contribution = max(contrib, key=contrib.get)
        recommendations.append(
            f"{dataset} mainly contributes to: {main_contribution}"
        )
    
    return recommendations
```

## Budget Allocation

- **Data preparation**: 0.5 GPU-hours
- **6 mixing strategies**: 6 × 0.5 = 3 GPU-hours
- **Optimization experiments**: 0.5 GPU-hours
- **Total**: 4 GPU-hours

## Expected Outcomes

1. **Optimal mixing ratios** for balanced performance
2. **Clear understanding** of dataset contributions
3. **Capability profiles** for each mixing strategy
4. **Practical guidelines** for dataset selection

## Success Criteria

1. **Identify clear winner** among mixing strategies
2. **>20% improvement** over single-dataset training
3. **Understand trade-offs** between capabilities
4. **Reproducible mixing recipe**

## Deliverables

1. **6 trained models** with different mixing strategies
2. **Capability heatmap** visualization
3. **Dataset contribution analysis**
4. **Optimal mixing recommendations**
5. **Code for automated mixing optimization**

## Key Files to Create

1. `data_mixer.py` - Dataset loading and mixing utilities
2. `mixing_experiments.py` - Main experimental pipeline
3. `capability_evaluation.py` - Multi-capability testing
4. `mixing_optimizer.py` - Automated optimization
5. `analyze_mixing_results.py` - Results analysis

## Common Challenges

1. **Dataset format differences**: Careful normalization needed
2. **Imbalanced lengths**: May need to truncate/pad
3. **Memory usage**: Use streaming and batching
4. **Fair comparison**: Ensure equal total training tokens

## Next Steps

1. Apply findings to create optimal training mixture
2. Test if mixing benefits scale to larger models
3. Investigate curriculum learning with dataset ordering
4. Consider dynamic mixing based on training progress 