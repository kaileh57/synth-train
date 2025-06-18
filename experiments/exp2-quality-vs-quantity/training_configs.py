#!/usr/bin/env python3
"""
Training configurations for quality vs quantity experiments
Defines different training setups for comparison
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
import json
from pathlib import Path


@dataclass
class ExperimentConfig:
    """Configuration for a single training experiment"""
    name: str
    dataset_path: str
    model_size: str
    num_train_samples: int
    training_steps: Optional[int] = None
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_steps: int = 500
    eval_steps: int = 500
    save_steps: int = 2000
    max_epochs: float = 1.0
    weight_decay: float = 0.01
    gradient_checkpointing: bool = True
    fp16: bool = True
    expected_gpu_hours: float = 0.5
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size"""
        return self.batch_size * self.gradient_accumulation_steps
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


# Define experiment configurations
EXPERIMENTS = {
    # Ultra-high quality experiments (10K samples)
    "ultra_10k_500M": ExperimentConfig(
        name="ultra_10k_500M",
        dataset_path="./data/ultra_high_10k",
        model_size="500M",
        num_train_samples=10000,
        batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,  # Higher LR for small data
        warmup_steps=200,
        eval_steps=100,
        save_steps=500,
        max_epochs=3.0,  # More epochs for small data
        expected_gpu_hours=0.5
    ),
    
    "ultra_10k_1B": ExperimentConfig(
        name="ultra_10k_1B",
        dataset_path="./data/ultra_high_10k",
        model_size="1B",
        num_train_samples=10000,
        batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        warmup_steps=200,
        eval_steps=100,
        save_steps=500,
        max_epochs=3.0,
        expected_gpu_hours=0.75
    ),
    
    # High quality experiments (100K samples)
    "high_100k_500M": ExperimentConfig(
        name="high_100k_500M",
        dataset_path="./data/high_100k",
        model_size="500M",
        num_train_samples=100000,
        batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=500,
        eval_steps=500,
        save_steps=2000,
        max_epochs=1.5,
        expected_gpu_hours=0.75
    ),
    
    "high_100k_1B": ExperimentConfig(
        name="high_100k_1B",
        dataset_path="./data/high_100k",
        model_size="1B",
        num_train_samples=100000,
        batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1.5e-4,
        warmup_steps=500,
        eval_steps=500,
        save_steps=2000,
        max_epochs=1.5,
        expected_gpu_hours=1.0
    ),
    
    # Medium quality experiments (1M samples)
    "medium_1M_500M": ExperimentConfig(
        name="medium_1M_500M",
        dataset_path="./data/medium_1m",
        model_size="500M",
        num_train_samples=1000000,
        batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=1000,
        eval_steps=1000,
        save_steps=5000,
        max_epochs=1.0,
        expected_gpu_hours=0.75
    ),
    
    "medium_1M_1B": ExperimentConfig(
        name="medium_1M_1B",
        dataset_path="./data/medium_1m",
        model_size="1B",
        num_train_samples=1000000,
        batch_size=2,
        gradient_accumulation_steps=16,
        learning_rate=1.5e-4,
        warmup_steps=1000,
        eval_steps=1000,
        save_steps=5000,
        max_epochs=1.0,
        expected_gpu_hours=1.0
    ),
}


# Ablation study configurations
ABLATION_CONFIGS = {
    # Test extreme quality filtering
    "ultra_5k_500M": ExperimentConfig(
        name="ultra_5k_500M",
        dataset_path="./data/ultra_high_10k",
        model_size="500M",
        num_train_samples=5000,
        batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=3e-4,
        warmup_steps=100,
        eval_steps=50,
        save_steps=250,
        max_epochs=5.0,  # Even more epochs
        expected_gpu_hours=0.25
    ),
    
    # Test with bottom quality samples
    "low_100k_500M": ExperimentConfig(
        name="low_100k_500M",
        dataset_path="./data/low_100k",  # Need to create this
        model_size="500M",
        num_train_samples=100000,
        batch_size=4,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        warmup_steps=500,
        eval_steps=500,
        save_steps=2000,
        max_epochs=1.5,
        expected_gpu_hours=0.75
    ),
}


# Learning rate schedules for different data sizes
LR_SCHEDULES = {
    "small": {  # < 50k samples
        "scheduler_type": "cosine",
        "num_cycles": 0.5,
        "final_lr_ratio": 0.1
    },
    "medium": {  # 50k - 500k samples
        "scheduler_type": "linear",
        "final_lr_ratio": 0.0
    },
    "large": {  # > 500k samples
        "scheduler_type": "linear",
        "final_lr_ratio": 0.0
    }
}


def get_lr_schedule(num_samples: int) -> Dict:
    """Get appropriate learning rate schedule based on dataset size"""
    if num_samples < 50000:
        return LR_SCHEDULES["small"]
    elif num_samples < 500000:
        return LR_SCHEDULES["medium"]
    else:
        return LR_SCHEDULES["large"]


def save_all_configs(output_dir: str = "./configs"):
    """Save all configurations to JSON files"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save main experiments
    main_configs = {}
    for name, config in EXPERIMENTS.items():
        main_configs[name] = config.to_dict()
        # Add LR schedule
        main_configs[name]["lr_schedule"] = get_lr_schedule(config.num_train_samples)
    
    with open(output_path / "experiment_configs.json", "w") as f:
        json.dump(main_configs, f, indent=2)
    
    # Save ablation configs
    ablation_configs = {}
    for name, config in ABLATION_CONFIGS.items():
        ablation_configs[name] = config.to_dict()
        ablation_configs[name]["lr_schedule"] = get_lr_schedule(config.num_train_samples)
    
    with open(output_path / "ablation_configs.json", "w") as f:
        json.dump(ablation_configs, f, indent=2)
    
    # Save summary
    summary = {
        "total_experiments": len(EXPERIMENTS) + len(ABLATION_CONFIGS),
        "main_experiments": list(EXPERIMENTS.keys()),
        "ablation_experiments": list(ABLATION_CONFIGS.keys()),
        "total_gpu_hours": sum(c.expected_gpu_hours for c in EXPERIMENTS.values()) + 
                          sum(c.expected_gpu_hours for c in ABLATION_CONFIGS.values()),
        "model_sizes": list(set(c.model_size for c in EXPERIMENTS.values())),
        "dataset_sizes": sorted(list(set(c.num_train_samples for c in EXPERIMENTS.values())))
    }
    
    with open(output_path / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Saved configurations to {output_path}")
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Expected GPU hours: {summary['total_gpu_hours']:.1f}")


def get_config_by_name(name: str) -> Optional[ExperimentConfig]:
    """Get configuration by experiment name"""
    if name in EXPERIMENTS:
        return EXPERIMENTS[name]
    elif name in ABLATION_CONFIGS:
        return ABLATION_CONFIGS[name]
    else:
        return None


def estimate_training_time(config: ExperimentConfig, gpu_type: str = "RTX_4090") -> Dict:
    """Estimate training time for a configuration"""
    
    # GPU throughput estimates (tokens/second)
    gpu_throughput = {
        "RTX_4090": {
            "500M": 3000,  # tokens/sec with fp16
            "1B": 1500
        },
        "A100": {
            "500M": 5000,
            "1B": 2500
        },
        "A6000": {
            "500M": 4500,
            "1B": 2200
        }
    }
    
    if gpu_type not in gpu_throughput:
        gpu_type = "RTX_4090"
    
    # Calculate total tokens
    avg_seq_length = 512  # Average sequence length
    total_tokens = config.num_train_samples * avg_seq_length * config.max_epochs
    
    # Get throughput
    throughput = gpu_throughput[gpu_type].get(config.model_size, 2000)
    
    # Account for gradient accumulation
    effective_throughput = throughput * 0.9  # 90% efficiency
    
    # Calculate time
    training_seconds = total_tokens / effective_throughput
    training_hours = training_seconds / 3600
    
    return {
        "total_tokens": total_tokens,
        "throughput": effective_throughput,
        "training_hours": training_hours,
        "training_steps": int(config.num_train_samples * config.max_epochs / config.get_effective_batch_size())
    }


def print_experiment_summary():
    """Print a summary of all experiments"""
    print("Quality vs Quantity Experiment Summary")
    print("=" * 80)
    
    # Group by model size
    for model_size in ["500M", "1B"]:
        print(f"\n{model_size} Model Experiments:")
        print("-" * 40)
        
        size_experiments = {k: v for k, v in EXPERIMENTS.items() if v.model_size == model_size}
        
        for name, config in sorted(size_experiments.items(), key=lambda x: x[1].num_train_samples):
            time_est = estimate_training_time(config)
            print(f"\n{name}:")
            print(f"  Dataset: {config.dataset_path}")
            print(f"  Samples: {config.num_train_samples:,}")
            print(f"  Epochs: {config.max_epochs}")
            print(f"  Batch size: {config.get_effective_batch_size()} (effective)")
            print(f"  Learning rate: {config.learning_rate}")
            print(f"  Est. time: {time_est['training_hours']:.2f} hours")
            print(f"  Est. steps: {time_est['training_steps']:,}")
    
    print("\n" + "=" * 80)
    print(f"Total GPU hours (conservative): {sum(c.expected_gpu_hours for c in EXPERIMENTS.values()):.1f}")


if __name__ == "__main__":
    # Save all configurations
    save_all_configs()
    
    # Print summary
    print_experiment_summary()
    
    # Test time estimation
    print("\n\nDetailed time estimates for RTX 4090:")
    print("-" * 60)
    for name, config in EXPERIMENTS.items():
        est = estimate_training_time(config, "RTX_4090")
        print(f"{name:20} | {est['training_hours']:6.2f} hours | {est['training_steps']:8,} steps") 