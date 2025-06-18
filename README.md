# Synthetic Data LLM Training Research

## Overview

This project explores training small LLMs (500M-1B parameters) using only synthetic data with a $50 budget constraint. Through 4 carefully designed experiments, we investigate critical questions about synthetic data quality, mixing strategies, and evaluation methods.

## Quick Start

```bash
# Setup (one-time)
git clone <repo-url>
cd synth-train
bash setup.sh

# Test setup
python test_setup.py

# Run first experiment
cd experiments/exp1-pure-synthetic
python data_preparation.py --max_samples 1000  # Quick test
python train_pure_synthetic.py --model_size 125M --fp16
```

See [QUICK_START.md](QUICK_START.md) for detailed 5-minute guide.

## Project Structure

```
synth-train/
├── experiments/
│   ├── exp1-pure-synthetic/      # Test pure synthetic data
│   ├── exp2-quality-vs-quantity/ # Quality threshold analysis
│   ├── exp3-dataset-mixing/      # Optimal mixing strategies
│   └── exp4-zero-cost-eval/      # Evaluation without APIs
├── logs/                          # Research planning documents
├── setup.sh                       # Automated setup script
├── requirements.txt               # Python dependencies
├── MASTER_PLAN.md                # Detailed research plan
├── QUICK_START.md                # 5-minute quickstart
└── LICENSE                       # MIT License
```

## Experiments

### Experiment 1: Pure Synthetic Excellence (2 GPU-hours)
**Question**: Can high-quality synthetic data alone train capable LLMs?
- Dataset: OpenHermes-2.5 (1M GPT-4 samples)
- Models: 500M & 1B parameters
- Focus: Baseline performance with premium synthetic data

### Experiment 2: Quality vs Quantity (3.5 GPU-hours)
**Question**: Is 10K ultra-high quality better than 1M medium quality?
- Quality scoring system (no APIs)
- Compare: 10K vs 100K vs 1M samples
- Find optimal quality-quantity threshold

### Experiment 3: Dataset Mixing (4 GPU-hours)
**Question**: What's the optimal mix of synthetic datasets?
- Datasets: OpenHermes, Cosmopedia, Magpie-Pro, FineWeb-Edu
- Test 6 mixing strategies
- Capability-based evaluation

### Experiment 4: Zero-Cost Evaluation (0.5 GPU-hours)
**Question**: How to evaluate without expensive API calls?
- Statistical metrics & perplexity
- Micro-judges (tiny specialized models)
- Behavioral testing framework

## Key Features

- **Budget-Conscious**: Total 10 GPU-hours (~$50)
- **Reproducible**: Fixed seeds, versioned dependencies
- **Practical**: Consumer GPUs (RTX 4090), standard frameworks
- **Scientific**: Systematic ablations, statistical validation
- **Open**: All code, data, and results publicly available

## Installation

### Requirements
- Linux (Ubuntu 20.04/22.04 recommended)
- Python 3.8+
- CUDA 11.7+ (optional but recommended)
- 50GB disk space
- 32GB RAM minimum

### Automated Setup
```bash
bash setup.sh
```

### Manual Setup
See [MANUAL_SETUP.md](MANUAL_SETUP.md) for detailed instructions.

## Results

*Results will be added as experiments complete*

Expected outcomes:
- Baseline perplexity < 20 on validation
- Quality threshold: ~50K samples for 80% performance
- Optimal mix: 40% instruction, 30% knowledge, 30% web
- Zero-cost eval correlation > 0.8 with human eval

## Contributing

We welcome contributions! Please:
1. Fork the repository
2. Create a feature branch
3. Run experiments with small samples first
4. Document your findings
5. Submit a pull request

## Citation

If you use this research, please cite:
```bibtex
@misc{synth-train-2024,
  title={Training Small LLMs on Synthetic Data: A $50 Budget Approach},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/synth-train}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## Acknowledgments

- OpenHermes-2.5 by Teknium
- HuggingFace for infrastructure
- Open source ML community

---

**Status**: 🚧 Active Development | **Budget Used**: $0/$50 | **Next**: Experiment 1
