# Experiment 2: Quality vs Quantity

## Overview
This experiment investigates whether 10K ultra-high quality samples can outperform 1M medium quality samples, finding the optimal quality-quantity threshold for small LLMs.

## Quick Start

1. **Score and filter the dataset:**
   ```bash
   # Test run with small dataset
   python filter_datasets.py --max_samples 10000
   
   # Full dataset scoring and filtering
   python filter_datasets.py
   ```

2. **Run experiments:**
   ```bash
   # Run a single experiment
   python train_quality_comparison.py ultra_10k_500M
   
   # Run all experiments sequentially
   python train_quality_comparison.py all --skip_existing
   ```

3. **Analyze results:**
   ```bash
   # Compare experiment results
   python analyze_results.py ./results/
   ```

## Scripts

- **quality_scorer.py**: Zero-cost quality scoring system (clarity, response quality, diversity, complexity)
- **filter_datasets.py**: Creates quality-filtered subsets (ultra-high 10K, high 100K, medium 1M)
- **training_configs.py**: Experiment configurations for different quality/quantity combinations
- **train_quality_comparison.py**: Main training script with quality-aware evaluation
- **evaluate_quality_tradeoff.py**: Analyzes model performance across quality dimensions
- **generation_analysis.py**: Deep analysis of generation quality

## Experiments

### Main Experiments (3.5 GPU-hours total)
1. **ultra_10k_500M**: 10K highest quality samples, 500M model
2. **ultra_10k_1B**: 10K highest quality samples, 1B model
3. **high_100k_500M**: 100K high quality samples, 500M model
4. **high_100k_1B**: 100K high quality samples, 1B model
5. **medium_1M_500M**: 1M medium quality samples, 500M model
6. **medium_1M_1B**: 1M medium quality samples, 1B model

### Quality Scoring Dimensions
- **Clarity (25%)**: Instruction clarity, readability, specificity
- **Response Quality (35%)**: Length, structure, coherence, completeness
- **Diversity (20%)**: Vocabulary diversity, topic coverage, non-templated
- **Complexity (20%)**: Reasoning depth, technical content, knowledge integration

## Expected Results

- **Hypothesis**: Ultra-high quality (10K) will achieve 80%+ of medium quality (1M) performance
- **Key Metrics**: Perplexity, generation quality, task completion rate
- **Quality Correlation**: Expect negative correlation between data quality and required quantity

## Running All Experiments

```bash
# Step 1: Filter datasets (1-2 hours)
python filter_datasets.py

# Step 2: Run all training experiments (3.5 GPU-hours)
python train_quality_comparison.py all --skip_existing

# Step 3: Generate comparison report
python analyze_results.py ./results/ --output_dir ./analysis/
```

## Ablation Studies

Additional experiments in `ABLATION_CONFIGS`:
- **ultra_5k_500M**: Extreme quality filtering (5K samples)
- **low_100k_500M**: Low quality baseline

## Output Structure

```
results/
├── ultra_10k_500M/
│   ├── experiment_config.json
│   ├── train_results.json
│   ├── evaluation/
│   ├── quality_analysis/
│   └── generation_analysis/
├── high_100k_500M/
│   └── ...
└── experiment_summary.json
```

## Memory Requirements

- 500M models: ~20GB VRAM with gradient checkpointing
- 1B models: ~35GB VRAM with gradient checkpointing
- Use `--fp16` for reduced memory usage 