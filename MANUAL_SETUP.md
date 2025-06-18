# Manual Setup Guide

This guide covers setup steps that cannot be automated and important configuration details for optimal performance.

## Table of Contents
1. [System Requirements](#system-requirements)
2. [CUDA Setup](#cuda-setup)
3. [API Keys and Authentication](#api-keys-and-authentication)
4. [Multi-GPU Configuration](#multi-gpu-configuration)
5. [Performance Optimization](#performance-optimization)
6. [Storage Considerations](#storage-considerations)
7. [Troubleshooting](#troubleshooting)
8. [Security Best Practices](#security-best-practices)

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ or similar Linux distribution
- **Python**: 3.8 or higher
- **RAM**: 32GB (minimum), 64GB recommended
- **Storage**: 500GB SSD (1TB+ recommended)
- **GPU**: 1x NVIDIA GPU with 24GB VRAM (RTX 4090, A5000, etc.)

### Recommended Setup
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.10
- **RAM**: 128GB
- **Storage**: 2TB NVMe SSD
- **GPU**: 8x RTX 4090 or 4x A100 40GB

## CUDA Setup

### 1. Install NVIDIA Drivers
```bash
# Check current driver
nvidia-smi

# If not installed, install drivers (Ubuntu)
sudo apt update
sudo apt install nvidia-driver-535  # or latest version

# Reboot after installation
sudo reboot
```

### 2. Install CUDA Toolkit
```bash
# For CUDA 11.8 (recommended for broad compatibility)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# Add to PATH in ~/.bashrc
echo 'export PATH=/usr/local/cuda-11.8/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvcc --version
```

### 3. Install cuDNN (for optimal performance)
1. Download cuDNN from [NVIDIA Developer](https://developer.nvidia.com/cudnn)
2. Extract and copy files:
```bash
tar -xvf cudnn-linux-x86_64-8.x.x.x_cuda11-archive.tar.xz
sudo cp cudnn-*-archive/include/cudnn*.h /usr/local/cuda/include
sudo cp cudnn-*-archive/lib/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

## API Keys and Authentication

### 1. Weights & Biases (Recommended)
```bash
# Sign up at https://wandb.ai/
# Get API key from https://wandb.ai/settings

# Add to .env file
echo "WANDB_API_KEY=your_key_here" >> .env

# Or set globally
wandb login
```

### 2. HuggingFace Hub
```bash
# Create account at https://huggingface.co/
# Get token from https://huggingface.co/settings/tokens

# Add to .env file
echo "HF_TOKEN=your_token_here" >> .env

# Or login via CLI
huggingface-cli login
```

### 3. Environment Variables
Create `.env` file from template:
```bash
cp .env.template .env
# Edit .env with your favorite editor
nano .env
```

## Multi-GPU Configuration

### 1. Check GPU Visibility
```bash
# List all GPUs
nvidia-smi -L

# Set visible GPUs (example for GPUs 0,1,2,3)
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 2. Configure for Distributed Training
```bash
# For PyTorch distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# For Accelerate
accelerate config
# Follow prompts to set up multi-GPU training
```

### 3. NCCL Optimization (for multi-GPU)
```bash
# Add to ~/.bashrc for optimal multi-GPU communication
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # If not using InfiniBand
export NCCL_P2P_DISABLE=1  # If P2P causes issues
```

## Performance Optimization

### 1. System Settings
```bash
# Disable CPU frequency scaling for consistent performance
sudo apt install cpufrequtils
sudo cpufreq-set -g performance

# Increase file descriptor limits
echo "* soft nofile 1000000" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 1000000" | sudo tee -a /etc/security/limits.conf

# Set GPU to persistence mode
sudo nvidia-smi -pm 1

# Set GPU power limit (optional, for thermal management)
sudo nvidia-smi -pl 350  # Watts, adjust based on cooling
```

### 2. PyTorch Optimization
```bash
# Add to ~/.bashrc
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"  # For RTX 4090
```

### 3. Data Loading Optimization
```bash
# Install faster data loading libraries
pip install --upgrade pillow-simd  # Faster image processing
pip install nvtx  # For profiling

# Set optimal number of workers
export DATALOADER_NUM_WORKERS=4  # Adjust based on CPU cores
```

## Storage Considerations

### 1. Dataset Storage
```bash
# Create fast storage directory (on NVMe SSD)
sudo mkdir -p /nvme/datasets
sudo chown $USER:$USER /nvme/datasets

# Symlink to project
ln -s /nvme/datasets ./data/datasets
```

### 2. Cache Configuration
```bash
# Set HuggingFace cache location
export HF_HOME=/nvme/cache/huggingface
export TRANSFORMERS_CACHE=/nvme/cache/transformers
export HF_DATASETS_CACHE=/nvme/cache/datasets

# Create directories
mkdir -p $HF_HOME $TRANSFORMERS_CACHE $HF_DATASETS_CACHE
```

### 3. Model Checkpoints
```bash
# Use fast storage for checkpoints
mkdir -p /nvme/checkpoints
ln -s /nvme/checkpoints ./checkpoints
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA Out of Memory
```python
# Add to training script
import torch
torch.cuda.empty_cache()

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Reduce batch size or use gradient accumulation
```

#### 2. Slow Data Loading
```bash
# Check I/O bottlenecks
iostat -x 1

# Use faster storage or increase workers
# Consider using datasets streaming mode
```

#### 3. Multi-GPU Hanging
```bash
# Debug NCCL issues
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL

# Try different backend
export TORCH_DISTRIBUTED_BACKEND=gloo  # Instead of nccl
```

#### 4. Import Errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall problematic package
pip install --force-reinstall package_name
```

## Security Best Practices

### 1. API Key Management
- Never commit `.env` files to git
- Use environment-specific `.env` files
- Rotate keys regularly
- Use read-only tokens when possible

### 2. Model Security
```bash
# Scan models before loading
pip install pickle-scanning
# Use the tool to scan downloaded models

# Only load models from trusted sources
```

### 3. Network Security
```bash
# If using wandb or uploading results
# Use VPN or secure connection
# Configure firewall rules
sudo ufw allow 29500  # If using distributed training
```

## Monitoring and Logging

### 1. GPU Monitoring
```bash
# Real-time GPU monitoring
watch -n 1 nvidia-smi

# Or use nvtop for better interface
sudo apt install nvtop
nvtop
```

### 2. Training Monitoring
```bash
# TensorBoard
tensorboard --logdir=./logs --host=0.0.0.0

# Weights & Biases (automatic if configured)
# View at https://wandb.ai/your-username/project-name
```

### 3. System Monitoring
```bash
# Install monitoring tools
sudo apt install htop iotop nethogs

# Monitor everything
htop  # CPU and memory
iotop  # Disk I/O
nethogs  # Network usage
```

## Advanced Configuration

### 1. Custom CUDA Kernels
If using custom CUDA operations:
```bash
# Install ninja for faster compilation
pip install ninja

# Set architecture-specific flags
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9"
```

### 2. Mixed Precision Training
```python
# Automatic mixed precision settings
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
```

### 3. Distributed Training Script
Create `launch_distributed.sh`:
```bash
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WORLD_SIZE=8

python -m torch.distributed.launch \
    --nproc_per_node=$WORLD_SIZE \
    --master_port=29500 \
    train_script.py \
    --distributed
```

## Final Checklist

Before starting experiments:

- [ ] CUDA and cuDNN properly installed
- [ ] All GPUs visible and functional
- [ ] Virtual environment activated
- [ ] API keys configured in `.env`
- [ ] Fast storage configured for datasets/checkpoints
- [ ] System optimizations applied
- [ ] Monitoring tools ready
- [ ] Backup strategy in place

## Getting Help

1. Check experiment-specific instructions in `experiments/*/instructions.md`
2. Review error logs in `logs/` directory
3. Common solutions in Troubleshooting section above
4. Project issues: Create an issue on GitHub
5. Framework issues: Check PyTorch/HuggingFace documentation

Remember: Start with small test runs before committing to full experiments! 