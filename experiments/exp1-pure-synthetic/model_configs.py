#!/usr/bin/env python3
"""
Model configurations for Experiment 1: Pure Synthetic Excellence
Defines GPT-2 style models of 500M and 1B parameters
"""

from dataclasses import dataclass
from typing import Optional
from transformers import GPT2Config


@dataclass
class ModelSize:
    """Model size configuration"""
    name: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    num_parameters: int  # Approximate


# Predefined model sizes
MODEL_SIZES = {
    "125M": ModelSize(
        name="125M",
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        num_parameters=125_000_000
    ),
    "350M": ModelSize(
        name="350M",
        hidden_size=1024,
        num_hidden_layers=20,
        num_attention_heads=16,
        intermediate_size=4096,
        num_parameters=350_000_000
    ),
    "500M": ModelSize(
        name="500M",
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        intermediate_size=4096,
        num_parameters=500_000_000
    ),
    "1B": ModelSize(
        name="1B",
        hidden_size=1536,
        num_hidden_layers=24,
        num_attention_heads=24,
        intermediate_size=6144,
        num_parameters=1_000_000_000
    ),
    "1.5B": ModelSize(
        name="1.5B",
        hidden_size=1600,
        num_hidden_layers=32,
        num_attention_heads=25,
        intermediate_size=6400,
        num_parameters=1_500_000_000
    )
}


def create_model_config(
    model_size: str = "500M",
    vocab_size: int = 50257,
    max_position_embeddings: int = 2048,
    dropout: float = 0.1,
    attention_dropout: float = 0.1,
    activation_function: str = "gelu_new",
    layer_norm_epsilon: float = 1e-5,
    initializer_range: float = 0.02,
    use_cache: bool = True,
    gradient_checkpointing: bool = False,
    scale_attn_by_inverse_layer_idx: bool = False,
    reorder_and_upcast_attn: bool = False,
) -> GPT2Config:
    """
    Create a GPT2Config for the specified model size
    
    Args:
        model_size: Size of the model (125M, 350M, 500M, 1B, 1.5B)
        vocab_size: Size of the vocabulary
        max_position_embeddings: Maximum sequence length
        dropout: Dropout probability
        attention_dropout: Attention dropout probability
        activation_function: Activation function type
        layer_norm_epsilon: Layer normalization epsilon
        initializer_range: Weight initialization range
        use_cache: Whether to use KV cache
        gradient_checkpointing: Whether to use gradient checkpointing
        scale_attn_by_inverse_layer_idx: Scale attention by inverse layer index
        reorder_and_upcast_attn: Reorder and upcast attention computations
        
    Returns:
        GPT2Config object
    """
    
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Model size {model_size} not found. Available sizes: {list(MODEL_SIZES.keys())}")
    
    size_config = MODEL_SIZES[model_size]
    
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=max_position_embeddings,
        n_embd=size_config.hidden_size,
        n_layer=size_config.num_hidden_layers,
        n_head=size_config.num_attention_heads,
        n_inner=size_config.intermediate_size,
        activation_function=activation_function,
        resid_pdrop=dropout,
        embd_pdrop=dropout,
        attn_pdrop=attention_dropout,
        layer_norm_epsilon=layer_norm_epsilon,
        initializer_range=initializer_range,
        use_cache=use_cache,
        gradient_checkpointing=gradient_checkpointing,
        scale_attn_by_inverse_layer_idx=scale_attn_by_inverse_layer_idx,
        reorder_and_upcast_attn=reorder_and_upcast_attn,
    )
    
    return config


def get_model_param_count(config: GPT2Config) -> int:
    """
    Calculate the approximate number of parameters for a GPT2 model
    
    Args:
        config: GPT2Config object
        
    Returns:
        Approximate number of parameters
    """
    # Embedding parameters
    embedding_params = config.vocab_size * config.n_embd  # Token embeddings
    position_params = config.n_positions * config.n_embd  # Position embeddings
    
    # Transformer block parameters (per layer)
    # Attention: Q, K, V projections + output projection
    attn_params_per_layer = 4 * config.n_embd * config.n_embd
    
    # MLP: 2 linear layers
    mlp_params_per_layer = 2 * config.n_embd * config.n_inner
    
    # Layer norms: 2 per transformer block
    ln_params_per_layer = 4 * config.n_embd  # 2 layer norms with weight and bias
    
    # Total transformer parameters
    transformer_params = config.n_layer * (attn_params_per_layer + mlp_params_per_layer + ln_params_per_layer)
    
    # Final layer norm
    final_ln_params = 2 * config.n_embd
    
    # Language modeling head (shares embeddings, so no additional params)
    
    total_params = embedding_params + position_params + transformer_params + final_ln_params
    
    return total_params


def print_model_info(config: GPT2Config, model_size: str = "Custom"):
    """Print model configuration and statistics"""
    param_count = get_model_param_count(config)
    
    print(f"\nModel Configuration ({model_size}):")
    print(f"  Hidden size: {config.n_embd}")
    print(f"  Number of layers: {config.n_layer}")
    print(f"  Number of heads: {config.n_head}")
    print(f"  Head dimension: {config.n_embd // config.n_head}")
    print(f"  Intermediate size: {config.n_inner}")
    print(f"  Vocabulary size: {config.vocab_size}")
    print(f"  Max position embeddings: {config.n_positions}")
    print(f"  Approximate parameters: {param_count:,} ({param_count / 1e9:.2f}B)")
    
    # Memory estimates (rough)
    # FP32: 4 bytes per param, FP16: 2 bytes per param
    fp32_memory_gb = (param_count * 4) / (1024**3)
    fp16_memory_gb = (param_count * 2) / (1024**3)
    
    print(f"\nMemory estimates:")
    print(f"  Model weights (FP32): {fp32_memory_gb:.2f} GB")
    print(f"  Model weights (FP16): {fp16_memory_gb:.2f} GB")
    print(f"  Estimated training memory (FP16 + optimizer): ~{fp16_memory_gb * 4:.2f} GB")


def get_training_config(model_size: str = "500M") -> dict:
    """
    Get recommended training configuration for a model size
    
    Args:
        model_size: Size of the model
        
    Returns:
        Dictionary with training recommendations
    """
    
    base_config = {
        "learning_rate": 2e-4,
        "warmup_steps": 1000,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
        "fp16": True,
        "gradient_checkpointing": True,
    }
    
    # Size-specific adjustments
    if model_size == "125M":
        base_config.update({
            "per_device_train_batch_size": 16,
            "gradient_accumulation_steps": 2,
            "gradient_checkpointing": False,  # Not needed for small model
        })
    elif model_size == "350M":
        base_config.update({
            "per_device_train_batch_size": 8,
            "gradient_accumulation_steps": 4,
        })
    elif model_size == "500M":
        base_config.update({
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 8,
        })
    elif model_size == "1B":
        base_config.update({
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 16,
            "learning_rate": 1.5e-4,  # Slightly lower for larger model
        })
    elif model_size == "1.5B":
        base_config.update({
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 32,
            "learning_rate": 1e-4,
        })
    
    return base_config


if __name__ == "__main__":
    # Test configurations
    print("Available model sizes:")
    for size_name in MODEL_SIZES:
        config = create_model_config(size_name)
        print_model_info(config, size_name)
        
        # Show training config
        train_config = get_training_config(size_name)
        print(f"\nRecommended training config for {size_name}:")
        print(f"  Batch size per device: {train_config['per_device_train_batch_size']}")
        print(f"  Gradient accumulation: {train_config['gradient_accumulation_steps']}")
        print(f"  Effective batch size: {train_config['per_device_train_batch_size'] * train_config['gradient_accumulation_steps']}")
        print(f"  Learning rate: {train_config['learning_rate']}")
        print("-" * 80) 