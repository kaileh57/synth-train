#!/bin/bash

# Synthetic Data LLM Training - Setup Script
# This script installs all dependencies for experiments 1-4
# Tested on Ubuntu 20.04/22.04 with CUDA 11.8/12.1

set -e  # Exit on error

echo "=========================================="
echo "Synthetic Data LLM Training Setup"
echo "=========================================="

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    case "$OSTYPE" in
        linux*)   echo "linux" ;;
        darwin*)  echo "macos" ;;
        msys*|cygwin*|mingw*) echo "windows" ;;
        *)        echo "unknown" ;;
    esac
}

# Function to detect CUDA version
detect_cuda() {
    if command_exists nvidia-smi; then
        cuda_version=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
        echo "$cuda_version"
    else
        echo "none"
    fi
}

# Function to detect Python version
detect_python() {
    if command_exists python3; then
        echo "python3"
    elif command_exists python; then
        echo "python"
    else
        echo "none"
    fi
}

OS=$(detect_os)
CUDA_VERSION=$(detect_cuda)
PYTHON_CMD=$(detect_python)

echo "Detected OS: $OS"
echo "Detected CUDA: $CUDA_VERSION"
echo "Python command: $PYTHON_CMD"

# Check Python installation
if [ "$PYTHON_CMD" = "none" ]; then
    echo "ERROR: Python not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Check for required system packages (for matplotlib backend)
echo ""
echo "Checking system dependencies..."
if [ "$OS" = "linux" ]; then
    if ! dpkg -l | grep -q python3-tk; then
        echo "WARNING: python3-tk not found. This is needed for matplotlib."
        echo "Install with: sudo apt-get install python3-tk"
    fi
fi

# Create project directory structure
print_status "Creating project directories..."
mkdir -p models
mkdir -p data
mkdir -p results
mkdir -p figures
mkdir -p checkpoints
mkdir -p judges
mkdir -p cache

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
else
    echo "Virtual environment already exists."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip setuptools wheel

# Install PyTorch based on CUDA availability
print_status "Installing PyTorch..."

if [ "$CUDA_VERSION" != "none" ]; then
    # Extract major.minor version
    CUDA_MAJOR_MINOR=$(echo $CUDA_VERSION | cut -d'.' -f1,2)
    
    case "$CUDA_MAJOR_MINOR" in
        "11.7")
            print_status "Installing PyTorch for CUDA 11.7..."
            $PYTHON_CMD -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
            ;;
        "11.8")
            print_status "Installing PyTorch for CUDA 11.8..."
            $PYTHON_CMD -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            ;;
        "12.1"|"12.2"|"12.3"|"12.4")
            print_status "Installing PyTorch for CUDA 12.1+..."
            $PYTHON_CMD -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
            ;;
        *)
            print_warning "CUDA $CUDA_VERSION detected but no specific PyTorch build available."
            print_warning "Installing default PyTorch (may not use GPU)..."
            $PYTHON_CMD -m pip install torch torchvision torchaudio
            ;;
    esac
else
    print_warning "No CUDA detected. Installing CPU-only PyTorch..."
    $PYTHON_CMD -m pip install torch torchvision torchaudio
fi

# Install core ML libraries
print_status "Installing core ML libraries..."
$PYTHON_CMD -m pip install transformers==4.36.2
$PYTHON_CMD -m pip install datasets==2.16.1
$PYTHON_CMD -m pip install accelerate==0.25.0
$PYTHON_CMD -m pip install tokenizers==0.15.0
$PYTHON_CMD -m pip install sentencepiece==0.1.99
$PYTHON_CMD -m pip install safetensors==0.4.1

# Install training and logging utilities
print_status "Installing training utilities..."
$PYTHON_CMD -m pip install wandb==0.16.2
$PYTHON_CMD -m pip install tensorboard==2.15.1
$PYTHON_CMD -m pip install tqdm==4.66.1
$PYTHON_CMD -m pip install rich==13.7.0

# Install data processing libraries
print_status "Installing data processing libraries..."
$PYTHON_CMD -m pip install numpy==1.24.3
$PYTHON_CMD -m pip install pandas==2.0.3
$PYTHON_CMD -m pip install scipy==1.11.4
$PYTHON_CMD -m pip install scikit-learn==1.3.2

# Install visualization libraries
print_status "Installing visualization libraries..."
$PYTHON_CMD -m pip install matplotlib==3.7.3
$PYTHON_CMD -m pip install seaborn==0.13.0
$PYTHON_CMD -m pip install plotly==5.18.0

# Install text analysis libraries (for Experiment 2)
print_status "Installing text analysis libraries..."
$PYTHON_CMD -m pip install textstat==0.7.3
$PYTHON_CMD -m pip install nltk==3.8.1
$PYTHON_CMD -m pip install rouge-score==0.1.2
$PYTHON_CMD -m pip install bert-score==0.3.13

# Install evaluation libraries (for Experiment 4)
print_status "Installing evaluation libraries..."
$PYTHON_CMD -m pip install sacrebleu==2.4.0
$PYTHON_CMD -m pip install evaluate==0.4.1

# Install missing dependencies for evaluation framework
print_status "Installing additional evaluation dependencies..."
$PYTHON_CMD -m pip install packaging==23.2
$PYTHON_CMD -m pip install py-cpuinfo==9.0.0

# Install missing dependencies for mixing experiments  
print_status "Installing dataset mixing dependencies..."
$PYTHON_CMD -m pip install jaxtyping==0.2.24

# Install utility libraries
print_status "Installing utility libraries..."
$PYTHON_CMD -m pip install python-dotenv==1.0.0
$PYTHON_CMD -m pip install pyyaml==6.0.1
$PYTHON_CMD -m pip install jsonlines==4.0.0
$PYTHON_CMD -m pip install fire==0.5.0
$PYTHON_CMD -m pip install omegaconf==2.3.0

# Install development tools
print_status "Installing development tools..."
$PYTHON_CMD -m pip install ipython==8.18.1
$PYTHON_CMD -m pip install jupyter==1.0.0
$PYTHON_CMD -m pip install notebook==7.0.6
$PYTHON_CMD -m pip install ipywidgets==8.1.1
$PYTHON_CMD -m pip install black==23.12.1
$PYTHON_CMD -m pip install isort==5.13.2
$PYTHON_CMD -m pip install flake8==7.0.0

# Download NLTK data
print_status "Downloading NLTK data..."
$PYTHON_CMD -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Create requirements.txt for reproducibility
print_status "Creating requirements.txt..."
$PYTHON_CMD -m pip freeze > requirements.txt

# Create .env file template
print_status "Creating .env template..."
cat > .env.template << 'EOF'
# Weights & Biases API key (optional but recommended)
WANDB_API_KEY=your_api_key_here

# HuggingFace token (for private datasets/models)
HF_TOKEN=your_token_here

# Project settings
PROJECT_NAME=synth-train
CACHE_DIR=./cache
DATA_DIR=./data
MODEL_DIR=./models
RESULTS_DIR=./results

# Training settings
MIXED_PRECISION=fp16
GRADIENT_CHECKPOINTING=true
DATALOADER_NUM_WORKERS=4

# Experiment tracking
EXPERIMENT_TRACKER=wandb  # or 'tensorboard' or 'none'
LOG_LEVEL=INFO
EOF

# Create a simple test script
print_status "Creating test script..."
cat > test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test script to verify installation"""

import sys
import torch
import transformers
import datasets
import accelerate
import numpy as np
import pandas as pd
from packaging import version

def check_package(name, min_version=None):
    try:
        module = __import__(name.replace('-', '_'))
        installed_version = getattr(module, '__version__', 'Unknown')
        
        if min_version and hasattr(module, '__version__'):
            if version.parse(installed_version) < version.parse(min_version):
                print(f"⚠️  {name}: {installed_version} (minimum {min_version} required)")
                return False
        
        print(f"✓ {name}: {installed_version}")
        return True
    except ImportError:
        print(f"✗ {name}: Not installed")
        return False

def main():
    print("Testing Synthetic Data LLM Training Setup")
    print("=" * 50)
    
    # Check Python version
    print(f"Python: {sys.version}")
    print()
    
    # Check CUDA
    print("CUDA Information:")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠️  CUDA not available - CPU only mode")
    print()
    
    # Check core packages
    print("Core ML Packages:")
    packages = [
        ('torch', '2.0.0'),
        ('transformers', '4.30.0'),
        ('datasets', '2.10.0'),
        ('accelerate', '0.20.0'),
        ('wandb', None),
        ('numpy', None),
        ('pandas', None),
        ('scipy', None),
        ('matplotlib', None),
        ('textstat', None),
        ('nltk', None),
        ('evaluate', None),
    ]
    
    all_good = True
    for package, min_ver in packages:
        if not check_package(package, min_ver):
            all_good = False
    
    print()
    
    # Test a simple model load
    print("Testing model loading:")
    try:
        from transformers import GPT2Config, GPT2Model
        config = GPT2Config(n_embd=128, n_layer=2, n_head=2)
        model = GPT2Model(config)
        print(f"✓ Successfully created test model with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        print(f"✗ Failed to create test model: {e}")
        all_good = False
    
    print()
    print("=" * 50)
    if all_good:
        print("✅ All checks passed! Setup is complete.")
    else:
        print("⚠️  Some issues detected. Please review the output above.")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
EOF

chmod +x test_setup.py

# Create a utility script for dataset downloading
print_status "Creating dataset download script..."
cat > download_datasets.py << 'EOF'
#!/usr/bin/env python3
"""Download and cache datasets for experiments"""

import os
from datasets import load_dataset
from huggingface_hub import snapshot_download
import argparse

def download_openhermes():
    print("Downloading OpenHermes-2.5...")
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)
    # Just iterate through first item to trigger download
    next(iter(dataset))
    print("✓ OpenHermes-2.5 cached")

def download_cosmopedia_sample():
    print("Downloading Cosmopedia sample...")
    # Download a smaller subset for testing
    dataset = load_dataset("HuggingFaceTB/cosmopedia", "auto_math_text", split="train", streaming=True)
    next(iter(dataset))
    print("✓ Cosmopedia sample cached")

def download_magpie():
    print("Downloading Magpie-Pro...")
    dataset = load_dataset("Magpie-Align/MagpieLM-Pro-300K-v0.1", split="train", streaming=True)
    next(iter(dataset))
    print("✓ Magpie-Pro cached")

def download_fineweb_sample():
    print("Downloading FineWeb-Edu sample...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu", split="train", streaming=True)
    next(iter(dataset))
    print("✓ FineWeb-Edu sample cached")

def main():
    parser = argparse.ArgumentParser(description="Download datasets for experiments")
    parser.add_argument("--exp1", action="store_true", help="Download datasets for Experiment 1")
    parser.add_argument("--exp2", action="store_true", help="Download datasets for Experiment 2")
    parser.add_argument("--exp3", action="store_true", help="Download datasets for Experiment 3")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    
    args = parser.parse_args()
    
    if args.all or args.exp1 or args.exp2:
        download_openhermes()
    
    if args.all or args.exp3:
        download_cosmopedia_sample()
        download_magpie()
        download_fineweb_sample()
    
    if not any([args.exp1, args.exp2, args.exp3, args.all]):
        print("No datasets specified. Use --all to download everything.")
        parser.print_help()

if __name__ == "__main__":
    main()
EOF

chmod +x download_datasets.py

print_status "Running installation test..."
python test_setup.py

echo
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo
print_status "Next steps:"
echo "1. Copy .env.template to .env and fill in your API keys"
echo "2. Run ./test_setup.py to verify installation"
echo "3. Run ./download_datasets.py --all to pre-download datasets"
echo "4. Read MANUAL_SETUP.md for additional configuration"
echo
print_warning "Remember to activate the virtual environment:"
echo "source venv/bin/activate"
echo

# Create activation reminder
cat > activate.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
echo "Virtual environment activated. You're ready to start experimenting!"
EOF
chmod +x activate.sh 