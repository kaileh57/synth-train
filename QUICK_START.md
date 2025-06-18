# Quick Start Guide (5 Minutes)

## 1. Initial Setup (2 minutes)

```bash
# Clone and enter directory
git clone <repo-url>
cd synth-train

# Run automated setup
bash setup.sh

# Activate environment (if created)
source venv/bin/activate  # or conda activate synth-train
```

## 2. Test Your Setup (1 minute)

```bash
# Verify installation
python test_setup.py

# Expected output:
# ✓ PyTorch installed
# ✓ CUDA available (or CPU mode)
# ✓ Transformers working
# ✓ Can load GPT-2 tokenizer
```

## 3. Run Your First Experiment (2 minutes)

### Option A: Test Run (Recommended for first time)
```bash
# Small test with 1000 samples
cd experiments/exp1-pure-synthetic
python data_preparation.py --max_samples 1000
python train_pure_synthetic.py --model_size 125M --num_train_epochs 1 --fp16
```

### Option B: Run Experiment 1 (Pure Synthetic)
```bash
cd experiments/exp1-pure-synthetic

# Prepare data
python data_preparation.py --analyze  # View dataset statistics

# Train 500M model
python train_pure_synthetic.py \
  --model_size 500M \
  --num_train_epochs 1 \
  --fp16 \
  --gradient_checkpointing
```

### Option C: Run Experiment 2 (Quality vs Quantity)
```bash
cd experiments/exp2-quality-vs-quantity

# Filter datasets by quality
python filter_datasets.py --max_samples 10000  # Quick test

# Train ultra-high quality model
python train_quality_comparison.py ultra_10k_500M
```

## 4. Monitor Progress

- **Console**: Training progress, loss, and metrics
- **TensorBoard**: `tensorboard --logdir ./models/`
- **Weights & Biases**: Add `--use_wandb` flag

## Common Issues

1. **Out of Memory**: 
   - Add `--gradient_checkpointing`
   - Reduce batch size in training script
   - Use smaller model (125M instead of 500M)

2. **Slow Data Loading**:
   - First run downloads datasets (one-time)
   - Use `--streaming` for large datasets

3. **CUDA Not Found**:
   - CPU training works but is slower
   - Check GPU with `nvidia-smi`

## Next Steps

- Read experiment READMEs in each folder
- Check `MASTER_PLAN.md` for research overview
- Join Discord/Discussions for help

---

**Time to first model**: ~10 minutes on GPU (125M model, 1K samples)

## Overview
This project aims to prove that small LLMs (500M-1B params) can be effectively trained on synthetic data alone with just $50 and consumer GPUs.

## Project Structure
```
synth-train/
├── MASTER_PLAN.md              # Read this first for the full plan
├── experiments/                # All experiments with detailed instructions
│   ├── exp1-pure-synthetic/    # Start here!
│   ├── exp2-quality-vs-quantity/
│   ├── exp3-dataset-mixing/
│   └── exp4-zero-cost-eval/
└── logs/                       # Background planning discussions
```

## Getting Started in 5 Minutes

1. **Read the Master Plan**
   - Open `MASTER_PLAN.md` for the comprehensive research overview
   - Understand the 4 experiments and their goals

2. **Start with Experiment 1**
   - Navigate to `experiments/exp1-pure-synthetic/`
   - Read `instructions.md` for detailed steps
   - This establishes baseline performance with pure synthetic data

3. **Key Experiments Flow**
   - **Exp 1**: Can synthetic data alone work? (Baseline)
   - **Exp 2**: How much does quality matter? (Optimization)
   - **Exp 3**: How to mix datasets? (Enhancement)
   - **Exp 4**: How to evaluate for free? (Infrastructure)

## Critical Success Factors

1. **Budget Management**: Total 10 GPU-hours ($50)
   - Exp 1: 2 hours
   - Exp 2: 3.5 hours
   - Exp 3: 4 hours
   - Exp 4: 0.5 hours

2. **Key Datasets**
   - OpenHermes-2.5 (highest quality)
   - Cosmopedia (knowledge)
   - Magpie-Pro (conversations)
   - FineWeb-Edu (diversity)

3. **Model Sizes**
   - 500M params (primary focus)
   - 1B params (stretch goal)

## Next Steps

1. Set up your environment (see README.md)
2. Read experiment 1 instructions thoroughly
3. Start with a small test run (100 steps)
4. Scale up based on initial results

## Questions to Answer

- Can 10K high-quality samples beat 1M medium-quality?
- What's the optimal dataset mixing ratio?
- Can we evaluate models without expensive APIs?
- Is synthetic data viable for production models?

## Remember

- This is groundbreaking research in democratizing LLM training
- Every experiment builds on the previous one
- Document everything for reproducibility
- The constraints (budget, hardware) drive innovation

Happy researching! 🚀 