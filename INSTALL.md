# Installation Guide

## Quick Start (Automated)

1. **Make setup script executable and run it:**
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

2. **Activate the environment:**
   ```bash
   source venv/bin/activate
   # or use the helper script:
   ./activate.sh
   ```

3. **Configure API keys:**
   ```bash
   cp .env.template .env
   nano .env  # Add your API keys
   ```

4. **Test installation:**
   ```bash
   ./test_setup.py
   ```

## Manual Installation

If you prefer manual installation or the script fails:

1. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install PyTorch (choose based on your CUDA version):**
   ```bash
   # CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # CPU only (for development)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Install all other dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data:**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

## System Setup

For GPU training, you must also:

1. **Install NVIDIA drivers and CUDA** - See [MANUAL_SETUP.md](MANUAL_SETUP.md#cuda-setup)
2. **Configure multi-GPU settings** - See [MANUAL_SETUP.md](MANUAL_SETUP.md#multi-gpu-configuration)
3. **Optimize performance** - See [MANUAL_SETUP.md](MANUAL_SETUP.md#performance-optimization)

## Verify Installation

Run the test script to ensure everything is working:

```bash
./test_setup.py
```

Expected output:
- ✅ Python 3.8+ detected
- ✅ CUDA available (if on GPU machine)
- ✅ All packages installed
- ✅ Test model created successfully

## Pre-download Datasets (Optional)

To avoid download delays during experiments:

```bash
# Download all datasets
./download_datasets.py --all

# Or download for specific experiments
./download_datasets.py --exp1  # Just OpenHermes-2.5
./download_datasets.py --exp3  # All mixing datasets
```

## Troubleshooting

### Common Issues:

1. **"CUDA not available"**
   - Normal on development machines
   - Ensure CUDA is properly installed on training machine
   - See [MANUAL_SETUP.md](MANUAL_SETUP.md#cuda-setup)

2. **"Package version conflicts"**
   - Try: `pip install --upgrade pip`
   - Then: `pip install -r requirements.txt --force-reinstall`

3. **"Permission denied"**
   - Make scripts executable: `chmod +x *.sh *.py`

4. **Out of disk space**
   - Models and datasets can be large (100GB+)
   - See storage setup in [MANUAL_SETUP.md](MANUAL_SETUP.md#storage-considerations)

## Next Steps

1. ✅ Installation complete
2. 📖 Read [QUICK_START.md](QUICK_START.md) for project overview
3. 🚀 Start with [Experiment 1](experiments/exp1-pure-synthetic/instructions.md)
4. 📊 Track progress with W&B or TensorBoard

Happy experimenting! 🎉 