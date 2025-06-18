# Experiment 2: Quality vs Quantity Trade-off - Instructions

## Overview
This experiment investigates the trade-off between data quality and quantity in synthetic data training. We'll train models on carefully filtered subsets to find the optimal quality threshold for small-scale LLMs.

## Hypothesis
High-quality synthetic data can achieve better performance with 10x-100x less data than medium-quality data, potentially revolutionizing our understanding of data requirements for LLM training.

## Objectives
1. Create quality-filtered subsets of synthetic data (10K, 100K, 1M samples)
2. Train identical models on each subset
3. Compare performance to identify quality thresholds
4. Develop zero-cost quality scoring methods

## Quality Scoring Framework

### Scoring Dimensions
1. **Instruction Clarity** (0-1)
   - Clear, specific instructions
   - No ambiguity or confusion
   - Appropriate complexity

2. **Response Quality** (0-1)
   - Accurate and helpful
   - Well-structured
   - Appropriate length

3. **Diversity Score** (0-1)
   - Unique vocabulary usage
   - Novel task types
   - Non-repetitive patterns

4. **Complexity Score** (0-1)
   - Reasoning depth
   - Multi-step solutions
   - Knowledge integration

## Implementation Steps

### Step 1: Quality Scoring Pipeline

```python
# quality_scorer.py

import numpy as np
from transformers import AutoTokenizer
import textstat
from collections import Counter
import re

class QualityScorer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.instruction_patterns = self.load_instruction_patterns()
        self.quality_cache = {}
        
    def score_sample(self, sample):
        """Score a single instruction-response pair."""
        instruction = sample['instruction']
        response = sample['response']
        
        scores = {
            'clarity': self.score_clarity(instruction),
            'response_quality': self.score_response(response),
            'diversity': self.score_diversity(instruction, response),
            'complexity': self.score_complexity(instruction, response)
        }
        
        # Weighted average
        weights = {'clarity': 0.25, 'response_quality': 0.35, 
                  'diversity': 0.2, 'complexity': 0.2}
        
        total_score = sum(scores[k] * weights[k] for k in scores)
        return total_score, scores
    
    def score_clarity(self, instruction):
        """Score instruction clarity."""
        # Length check (not too short or too long)
        word_count = len(instruction.split())
        if word_count < 5:
            length_score = 0.5
        elif word_count > 100:
            length_score = 0.7
        else:
            length_score = 1.0
            
        # Question mark or clear directive
        has_question = '?' in instruction
        has_directive = any(word in instruction.lower() for word in 
                          ['explain', 'describe', 'write', 'create', 'analyze'])
        structure_score = 1.0 if (has_question or has_directive) else 0.6
        
        # Readability
        try:
            readability = textstat.flesch_reading_ease(instruction)
            read_score = min(readability / 100, 1.0) if readability > 0 else 0.5
        except:
            read_score = 0.7
            
        return (length_score + structure_score + read_score) / 3
    
    def score_response(self, response):
        """Score response quality."""
        # Length appropriateness
        word_count = len(response.split())
        if word_count < 10:
            length_score = 0.3
        elif word_count > 500:
            length_score = 0.8
        else:
            length_score = 1.0
            
        # Structure indicators
        has_paragraphs = '\n\n' in response
        has_list = bool(re.search(r'^\d+\.|\-\s', response, re.MULTILINE))
        structure_score = 1.0 if (has_paragraphs or has_list) else 0.7
        
        # Coherence proxy (sentence length variation)
        sentences = response.split('.')
        if len(sentences) > 2:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            coherence_score = 1.0 if np.std(lengths) > 3 else 0.7
        else:
            coherence_score = 0.6
            
        return (length_score + structure_score + coherence_score) / 3
    
    def score_diversity(self, instruction, response):
        """Score uniqueness/diversity."""
        combined = instruction + " " + response
        tokens = self.tokenizer.tokenize(combined)
        
        # Vocabulary diversity
        unique_ratio = len(set(tokens)) / len(tokens)
        vocab_score = min(unique_ratio * 2, 1.0)  # Scale up
        
        # N-gram diversity
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        bigram_diversity = len(set(bigrams)) / len(bigrams) if bigrams else 0
        
        # Topic diversity (simple keyword check)
        common_topics = ['python', 'code', 'explain', 'write', 'create']
        has_common = sum(1 for topic in common_topics if topic in combined.lower())
        topic_score = 1.0 - (has_common * 0.1)  # Penalize common topics
        
        return (vocab_score + bigram_diversity + topic_score) / 3
    
    def score_complexity(self, instruction, response):
        """Score reasoning complexity."""
        # Multi-step indicators
        step_indicators = ['first', 'second', 'then', 'finally', 'step']
        has_steps = sum(1 for ind in step_indicators if ind in response.lower())
        step_score = min(has_steps * 0.3, 1.0)
        
        # Code or technical content
        has_code = '```' in response or 'def ' in response
        technical_score = 1.0 if has_code else 0.6
        
        # Question complexity
        question_words = ['why', 'how', 'analyze', 'compare', 'evaluate']
        complex_question = any(word in instruction.lower() for word in question_words)
        question_score = 1.0 if complex_question else 0.5
        
        return (step_score + technical_score + question_score) / 3
```

### Step 2: Dataset Filtering

```python
# filter_datasets.py

def create_quality_subsets(dataset, quality_scorer):
    """Create three quality-based subsets."""
    
    print("Scoring entire dataset...")
    scored_data = []
    
    for idx, sample in enumerate(tqdm(dataset)):
        score, components = quality_scorer.score_sample(sample)
        scored_data.append({
            'sample': sample,
            'score': score,
            'components': components,
            'idx': idx
        })
        
        # Save periodically
        if idx % 10000 == 0:
            save_checkpoint(scored_data, f'scored_data_{idx}.pkl')
    
    # Sort by quality score
    scored_data.sort(key=lambda x: x['score'], reverse=True)
    
    # Create subsets
    subsets = {
        'ultra_high_10k': scored_data[:10000],
        'high_100k': scored_data[:100000],
        'medium_1m': scored_data[:1000000]
    }
    
    # Analyze score distributions
    for name, subset in subsets.items():
        scores = [item['score'] for item in subset]
        print(f"\n{name}:")
        print(f"  Score range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"  Mean score: {np.mean(scores):.3f}")
        print(f"  Std dev: {np.std(scores):.3f}")
    
    return subsets

def save_filtered_datasets(subsets):
    """Save filtered datasets in HuggingFace format."""
    
    for name, data in subsets.items():
        # Extract just the samples
        samples = [item['sample'] for item in data]
        
        # Convert to HuggingFace dataset
        from datasets import Dataset
        dataset = Dataset.from_list(samples)
        
        # Save
        dataset.save_to_disk(f'data/{name}')
        
        # Also save score metadata
        scores_df = pd.DataFrame([
            {'idx': item['idx'], 'score': item['score'], **item['components']}
            for item in data
        ])
        scores_df.to_csv(f'data/{name}_scores.csv', index=False)
```

### Step 3: Training Configuration

```python
# training_configs.py

def get_training_config(subset_name, model_size='500M'):
    """Get training configuration for each subset."""
    
    # Base configuration
    base_config = {
        'model_size': model_size,
        'learning_rate': 2e-4,
        'warmup_steps': 1000,
        'weight_decay': 0.01,
        'fp16': True,
        'gradient_checkpointing': True,
    }
    
    # Subset-specific adjustments
    if subset_name == 'ultra_high_10k':
        return {
            **base_config,
            'num_train_epochs': 10,  # More epochs for small data
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': 4,
            'eval_steps': 500,
            'save_steps': 1000,
            'learning_rate': 1e-4,  # Lower LR for small data
        }
    
    elif subset_name == 'high_100k':
        return {
            **base_config,
            'num_train_epochs': 3,
            'per_device_train_batch_size': 8,
            'gradient_accumulation_steps': 4,
            'eval_steps': 1000,
            'save_steps': 5000,
        }
    
    else:  # medium_1m
        return {
            **base_config,
            'num_train_epochs': 1,
            'per_device_train_batch_size': 16,
            'gradient_accumulation_steps': 2,
            'eval_steps': 2000,
            'save_steps': 10000,
        }
```

### Step 4: Comparative Training Script

```python
# train_quality_comparison.py

def train_all_subsets():
    """Train models on all quality subsets."""
    
    results = {}
    
    for subset_name in ['ultra_high_10k', 'high_100k', 'medium_1m']:
        print(f"\n{'='*50}")
        print(f"Training on {subset_name}")
        print(f"{'='*50}")
        
        # Load dataset
        dataset = load_from_disk(f'data/{subset_name}')
        
        # Get config
        config = get_training_config(subset_name)
        
        # Initialize model (fresh for each subset)
        model = GPT2LMHeadModel(GPT2Config(**config_500M))
        
        # Train
        trainer = Trainer(
            model=model,
            args=TrainingArguments(**config),
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
        
        # Track time
        start_time = time.time()
        train_result = trainer.train()
        train_time = time.time() - start_time
        
        # Evaluate
        eval_results = trainer.evaluate()
        
        # Generate samples
        generation_results = generate_and_evaluate(model, tokenizer)
        
        # Store results
        results[subset_name] = {
            'train_time': train_time,
            'final_loss': train_result.training_loss,
            'eval_perplexity': np.exp(eval_results['eval_loss']),
            'generation_quality': generation_results,
            'config': config,
            'data_size': len(dataset['train']),
        }
        
        # Save model
        model.save_pretrained(f'models/exp2-{subset_name}')
        
    return results
```

### Step 5: Evaluation Framework

```python
# evaluate_quality_tradeoff.py

def comprehensive_evaluation(results):
    """Analyze quality vs quantity trade-offs."""
    
    analysis = {
        'performance_vs_size': {},
        'efficiency_metrics': {},
        'quality_threshold': None,
        'recommendations': []
    }
    
    # 1. Performance comparison
    for subset, metrics in results.items():
        size = metrics['data_size']
        perplexity = metrics['eval_perplexity']
        
        analysis['performance_vs_size'][subset] = {
            'size': size,
            'perplexity': perplexity,
            'performance_per_sample': 1 / (perplexity * size),
            'training_efficiency': metrics['train_time'] / size
        }
    
    # 2. Find quality threshold
    # Where does ultra-high quality stop being worth it?
    perplexities = [results[s]['eval_perplexity'] for s in 
                   ['ultra_high_10k', 'high_100k', 'medium_1m']]
    
    if perplexities[0] < perplexities[1] * 0.9:  # 10% better
        analysis['quality_threshold'] = 'ultra_high'
        analysis['recommendations'].append(
            "Ultra-high quality (top 1%) provides significant benefits"
        )
    elif perplexities[1] < perplexities[2] * 0.9:
        analysis['quality_threshold'] = 'high'
        analysis['recommendations'].append(
            "High quality (top 10%) is optimal balance"
        )
    else:
        analysis['quality_threshold'] = 'medium'
        analysis['recommendations'].append(
            "Quantity matters more than extreme quality filtering"
        )
    
    # 3. Efficiency analysis
    for subset in results:
        train_time = results[subset]['train_time']
        perplexity = results[subset]['eval_perplexity']
        
        # Time to reach perplexity of 20
        time_to_threshold = train_time * (20 / perplexity)
        
        analysis['efficiency_metrics'][subset] = {
            'time_to_threshold': time_to_threshold,
            'samples_per_point_perplexity': 
                results[subset]['data_size'] / (100 - perplexity)
        }
    
    return analysis

def create_visualizations(results, analysis):
    """Create plots for the paper."""
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 1. Performance vs Data Size
    plt.figure(figsize=(10, 6))
    
    subsets = ['ultra_high_10k', 'high_100k', 'medium_1m']
    sizes = [results[s]['data_size'] for s in subsets]
    perplexities = [results[s]['eval_perplexity'] for s in subsets]
    
    plt.semilogx(sizes, perplexities, 'o-', markersize=10, linewidth=2)
    plt.xlabel('Dataset Size', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title('Quality vs Quantity Trade-off', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    for i, subset in enumerate(subsets):
        plt.annotate(subset, (sizes[i], perplexities[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.savefig('figures/exp2_quality_vs_quantity.png', dpi=300, bbox_inches='tight')
    
    # 2. Training Efficiency
    plt.figure(figsize=(10, 6))
    
    efficiency = [analysis['efficiency_metrics'][s]['time_to_threshold'] 
                 for s in subsets]
    
    plt.bar(subsets, efficiency, color=['gold', 'silver', 'bronze'])
    plt.ylabel('Time to Perplexity=20 (hours)', fontsize=12)
    plt.title('Training Efficiency by Quality Level', fontsize=14)
    plt.xticks(rotation=45)
    
    plt.savefig('figures/exp2_training_efficiency.png', dpi=300, bbox_inches='tight')
```

### Step 6: Generation Quality Analysis

```python
# generation_analysis.py

def analyze_generation_quality(model, tokenizer, test_prompts):
    """Detailed generation quality analysis."""
    
    quality_metrics = {
        'coherence': [],
        'diversity': [],
        'instruction_following': [],
        'length_appropriateness': []
    }
    
    for prompt in test_prompts:
        # Generate multiple completions
        completions = []
        for _ in range(5):
            output = model.generate(
                tokenizer(prompt, return_tensors='pt').input_ids,
                max_length=200,
                temperature=0.8,
                do_sample=True,
                top_p=0.9
            )
            completion = tokenizer.decode(output[0], skip_special_tokens=True)
            completions.append(completion)
        
        # Analyze completions
        metrics = analyze_completions(completions, prompt)
        for key in quality_metrics:
            quality_metrics[key].append(metrics[key])
    
    # Aggregate results
    return {k: np.mean(v) for k, v in quality_metrics.items()}
```

## Budget Allocation

- **Data Scoring**: 0.5 GPU-hours (one-time)
- **Ultra-high 10K**: 0.5 GPU-hours
- **High 100K**: 1 GPU-hour  
- **Medium 1M**: 1.5 GPU-hours
- **Total**: 3.5 GPU-hours

## Expected Outcomes

1. **Clear quality threshold** identification
2. **Efficiency curves** showing diminishing returns
3. **Practical guidelines** for data filtering
4. **Cost-benefit analysis** of quality vs compute

## Key Insights to Extract

1. **Minimum quality threshold** for viable training
2. **Optimal quality/quantity balance** for budget-constrained training
3. **Quality scoring methods** that correlate with model performance
4. **Scaling laws** for synthetic data quality

## Deliverables

1. **Trained models** for each quality tier
2. **Quality scoring code** (reusable)
3. **Performance comparison** table
4. **Visualizations** of trade-offs
5. **Practical recommendations** for practitioners

## Next Steps

1. Use findings to inform data selection for Experiment 3
2. Apply quality scoring to dataset mixing decisions
3. Consider quality-aware curriculum learning
4. Document quality thresholds for different model sizes 