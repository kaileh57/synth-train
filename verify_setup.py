#!/usr/bin/env python3
"""
Verification script to check if the project is properly set up
"""

import os
import sys
from pathlib import Path
import importlib
import json

# Color codes for output
RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
NC = '\033[0m'  # No Color

def print_status(message, status="info"):
    """Print colored status message"""
    if status == "success":
        print(f"{GREEN}✓{NC} {message}")
    elif status == "error":
        print(f"{RED}✗{NC} {message}")
    elif status == "warning":
        print(f"{YELLOW}⚠{NC} {message}")
    else:
        print(f"{BLUE}→{NC} {message}")

def check_python_version():
    """Check Python version"""
    print("\n1. Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_status(f"Python {version.major}.{version.minor}.{version.micro} - OK", "success")
        return True
    else:
        print_status(f"Python {version.major}.{version.minor} found. Requires 3.8+", "error")
        return False

def check_imports():
    """Check if all required packages can be imported"""
    print("\n2. Checking required packages...")
    
    required_packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("datasets", "Datasets"),
        ("accelerate", "Accelerate"),
        ("wandb", "Weights & Biases"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("textstat", "Textstat"),
        ("nltk", "NLTK"),
        ("evaluate", "Evaluate"),
        ("matplotlib", "Matplotlib"),
        ("seaborn", "Seaborn"),
        ("tqdm", "TQDM"),
    ]
    
    all_good = True
    for package, name in required_packages:
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            print_status(f"{name} ({package}) v{version}", "success")
        except ImportError:
            print_status(f"{name} ({package}) - NOT INSTALLED", "error")
            all_good = False
    
    return all_good

def check_cuda():
    """Check CUDA availability"""
    print("\n3. Checking CUDA/GPU...")
    try:
        import torch
        if torch.cuda.is_available():
            print_status(f"CUDA available: {torch.cuda.get_device_name(0)}", "success")
            print_status(f"CUDA version: {torch.version.cuda}", "success")
            print_status(f"Number of GPUs: {torch.cuda.device_count()}", "success")
            return True
        else:
            print_status("CUDA not available - CPU only mode", "warning")
            return True  # CPU mode is acceptable
    except Exception as e:
        print_status(f"Error checking CUDA: {e}", "error")
        return False

def check_directories():
    """Check if required directories exist"""
    print("\n4. Checking project structure...")
    
    required_dirs = [
        "experiments/exp1-pure-synthetic",
        "experiments/exp2-quality-vs-quantity",
        "experiments/exp3-dataset-mixing",
        "experiments/exp4-zero-cost-eval",
        "logs",
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print_status(f"{dir_path}/", "success")
        else:
            print_status(f"{dir_path}/ - MISSING", "error")
            all_good = False
    
    return all_good

def check_experiment_files():
    """Check if experiment files exist"""
    print("\n5. Checking experiment files...")
    
    exp1_files = [
        "experiments/exp1-pure-synthetic/data_preparation.py",
        "experiments/exp1-pure-synthetic/model_configs.py",
        "experiments/exp1-pure-synthetic/train_pure_synthetic.py",
        "experiments/exp1-pure-synthetic/evaluate.py",
        "experiments/exp1-pure-synthetic/generate_samples.py",
        "experiments/exp1-pure-synthetic/README.md",
    ]
    
    exp2_files = [
        "experiments/exp2-quality-vs-quantity/quality_scorer.py",
        "experiments/exp2-quality-vs-quantity/filter_datasets.py",
        "experiments/exp2-quality-vs-quantity/training_configs.py",
        "experiments/exp2-quality-vs-quantity/train_quality_comparison.py",
        "experiments/exp2-quality-vs-quantity/evaluate_quality_tradeoff.py",
        "experiments/exp2-quality-vs-quantity/generation_analysis.py",
        "experiments/exp2-quality-vs-quantity/README.md",
    ]
    
    all_files = exp1_files + exp2_files
    all_good = True
    
    for file_path in all_files:
        if Path(file_path).exists():
            print_status(f"{file_path}", "success")
        else:
            print_status(f"{file_path} - MISSING", "error")
            all_good = False
    
    return all_good

def check_nltk_data():
    """Check if NLTK data is downloaded"""
    print("\n6. Checking NLTK data...")
    try:
        import nltk
        required_data = ['punkt', 'stopwords']
        all_good = True
        
        for data_name in required_data:
            try:
                nltk.data.find(f'tokenizers/{data_name}')
                print_status(f"NLTK {data_name}", "success")
            except LookupError:
                print_status(f"NLTK {data_name} - NOT DOWNLOADED", "warning")
                print_status(f"  Run: python -c \"import nltk; nltk.download('{data_name}')\"", "info")
                all_good = False
        
        return all_good
    except ImportError:
        print_status("NLTK not installed", "error")
        return False

def test_experiment_imports():
    """Test if experiment modules can be imported"""
    print("\n7. Testing experiment imports...")
    
    # Save current directory
    original_dir = os.getcwd()
    all_good = True
    
    try:
        # Test Experiment 1 imports
        os.chdir("experiments/exp1-pure-synthetic")
        try:
            import model_configs
            import data_preparation
            print_status("Experiment 1 modules import correctly", "success")
        except Exception as e:
            print_status(f"Experiment 1 import error: {e}", "error")
            all_good = False
        
        os.chdir(original_dir)
        
        # Test Experiment 2 imports
        os.chdir("experiments/exp2-quality-vs-quantity")
        try:
            import quality_scorer
            import training_configs
            print_status("Experiment 2 modules import correctly", "success")
        except Exception as e:
            print_status(f"Experiment 2 import error: {e}", "error")
            all_good = False
            
    finally:
        os.chdir(original_dir)
    
    return all_good

def check_memory():
    """Check available memory"""
    print("\n8. Checking system resources...")
    try:
        import psutil
        
        # RAM
        ram = psutil.virtual_memory()
        ram_gb = ram.total / (1024**3)
        if ram_gb >= 32:
            print_status(f"RAM: {ram_gb:.1f} GB - OK", "success")
        else:
            print_status(f"RAM: {ram_gb:.1f} GB - May be insufficient (32GB recommended)", "warning")
        
        # Disk space
        disk = psutil.disk_usage('.')
        disk_gb = disk.free / (1024**3)
        if disk_gb >= 50:
            print_status(f"Free disk space: {disk_gb:.1f} GB - OK", "success")
        else:
            print_status(f"Free disk space: {disk_gb:.1f} GB - May be insufficient (50GB recommended)", "warning")
            
    except ImportError:
        print_status("psutil not installed - cannot check system resources", "warning")

def main():
    """Run all checks"""
    print("="*60)
    print("Synthetic Data LLM Training - Setup Verification")
    print("="*60)
    
    checks = [
        ("Python Version", check_python_version),
        ("Package Imports", check_imports),
        ("CUDA/GPU", check_cuda),
        ("Directory Structure", check_directories),
        ("Experiment Files", check_experiment_files),
        ("NLTK Data", check_nltk_data),
        ("Experiment Imports", test_experiment_imports),
        ("System Resources", check_memory),
    ]
    
    results = {}
    for name, check_func in checks:
        results[name] = check_func()
    
    # Summary
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    failed_checks = [name for name, passed in results.items() if not passed]
    
    if all_passed:
        print_status("All checks passed! ✨", "success")
        print("\nYou're ready to start experimenting!")
        print("\nNext steps:")
        print("1. cd experiments/exp1-pure-synthetic")
        print("2. python data_preparation.py --max_samples 1000")
        print("3. python train_pure_synthetic.py --model_size 125M --fp16")
    else:
        print_status(f"{len(failed_checks)} checks failed:", "error")
        for check in failed_checks:
            print(f"  - {check}")
        print("\nPlease run 'bash setup.sh' to fix missing dependencies")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 