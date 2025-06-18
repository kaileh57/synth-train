# Experiment 1: Pure Synthetic Excellence

## Overview
This experiment tests whether high-quality synthetic data (GPT-4 generated) alone can train capable small-scale LLMs (500M-1B parameters).

## Quick Start

1. **Prepare the data:**
   ```bash
   python data_preparation.py --analyze  # Check dataset statistics
   python data_preparation.py --max_samples 1000  # Small test
   python data_preparation.py  # Full dataset
   ```

2. **Train the model:**
   ```bash
   # Test run (small dataset)
   python train_pure_synthetic.py \
     --model_size 125M \
     --max_samples 10000 \
     --num_train_epochs 1 \
     --eval_steps 100 \
     --fp16 \
     --gradient_checkpointing
   
   # Full 500M model
   python train_pure_synthetic.py \
     --model_size 500M \
     --num_train_epochs 1 \
     --fp16 \
     --gradient_checkpointing \
     --use_wandb
   
   # Full 1B model (requires more memory)
   python train_pure_synthetic.py \
     --model_size 1B \
     --num_train_epochs 1 \
     --fp16 \
     --gradient_checkpointing \
     --use_wandb
   ```

3. **Evaluate the model:**
   ```bash
   python evaluate.py \
     --model_path ./models/exp1-500M-*/  \
     --dataset_path ./data/openhermes_processed \
     --num_samples 1000
   ```

4. **Generate samples:**
   ```bash
   python generate_samples.py \
     --model_path ./models/exp1-500M-*/ \
     --num_samples 50
   ```

## Scripts

- **data_preparation.py**: Downloads and preprocesses OpenHermes-2.5 dataset
- **model_configs.py**: Model architecture configurations (125M to 1.5B)
- **train_pure_synthetic.py**: Main training script with HuggingFace Trainer
- **evaluate.py**: Comprehensive evaluation (perplexity, ROUGE, BLEU, diversity)
- **generate_samples.py**: Generate and analyze sample outputs

## Key Features

1. **Automatic Mixed Precision**: Uses FP16 to reduce memory usage
2. **Gradient Checkpointing**: Trades compute for memory
3. **Multi-GPU Support**: Automatically uses all available GPUs
4. **Weights & Biases Integration**: Optional experiment tracking
5. **Resume Training**: Can resume from checkpoints

## Expected Results

- **Perplexity**: < 20 on validation set
- **ROUGE-L**: > 0.3 on instruction-following
- **Training Time**: ~1 hour for 500M model on 8x RTX 4090
- **Memory Usage**: ~20GB VRAM for 500M model with gradient checkpointing

## Common Issues

1. **OOM Errors**: Reduce batch size or enable gradient checkpointing
2. **Slow Training**: Check data loading, use more GPUs
3. **Poor Quality**: Increase training epochs or learning rate

## Budget

- 500M model: ~1 GPU-hour
- 1B model: ~1 GPU-hour
- Total: 2 GPU-hours 