"""
Weight loading from HuggingFace format (Safetensors)
"""

import torch
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import re


def load_safetensors_metadata(path: str) -> Dict:
    """Load metadata from safetensors file without loading weights"""
    from safetensors import safe_open
    
    metadata = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    return metadata


def load_safetensors_weights(path: str, device: str = "cpu") -> Dict[str, torch.Tensor]:
    """Load all weights from safetensors file"""
    from safetensors import safe_open
    
    weights = {}
    with safe_open(path, framework="pt", device=device) as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def convert_hf_to_minillm(hf_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Convert HuggingFace weight names to MiniLLM format
    
    HF format: model.layers.0.self_attn.q_proj.weight
    MiniLLM: layers.0.self_attn.q_proj.weight
    """
    minillm_weights = {}
    
    for hf_key, tensor in hf_weights.items():
        # Remove 'model.' prefix if present
        if hf_key.startswith("model."):
            minillm_key = hf_key[6:]  # Remove 'model.'
        else:
            minillm_key = hf_key
        
        minillm_weights[minillm_key] = tensor
    
    return minillm_weights


def load_model_weights(
    model_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> Dict[str, torch.Tensor]:
    """
    Load model weights from HuggingFace format
    
    Supports:
    - Safetensors format (.safetensors files)
    - PyTorch format (.bin files)
    
    Args:
        model_path: Path to model directory or file
        device: Device to load weights to
        dtype: Data type for weights
    
    Returns:
        Dictionary of weight tensors
    """
    model_path = Path(model_path)
    
    # Find weight files
    if model_path.is_dir():
        # Look for safetensors files first
        safetensor_files = list(model_path.glob("*.safetensors"))
        pytorch_files = list(model_path.glob("pytorch_model*.bin"))
        
        if safetensor_files:
            weight_files = sorted(safetensor_files)
            print(f"Found {len(weight_files)} safetensors file(s)")
        elif pytorch_files:
            weight_files = sorted(pytorch_files)
            print(f"Found {len(weight_files)} pytorch file(s)")
        else:
            raise FileNotFoundError(f"No weight files found in {model_path}")
    else:
        weight_files = [model_path]
    
    # Load all weights
    all_weights = {}
    for file_path in weight_files:
        print(f"Loading {file_path.name}...")
        
        if file_path.suffix == ".safetensors":
            weights = load_safetensors_weights(str(file_path), device="cpu")
        else:
            weights = torch.load(file_path, map_location="cpu")
        
        all_weights.update(weights)
    
    # Convert to MiniLLM format
    all_weights = convert_hf_to_minillm(all_weights)
    
    # Convert dtype and move to device
    for key in all_weights:
        if isinstance(all_weights[key], torch.Tensor):
            all_weights[key] = all_weights[key].to(dtype).to(device)
    
    print(f"Loaded {len(all_weights)} tensors")
    
    # Print model size
    total_params = sum(t.numel() for t in all_weights.values() if isinstance(t, torch.Tensor))
    print(f"Total parameters: {total_params / 1e9:.2f}B")
    
    return all_weights


def load_config(model_path: str) -> Dict:
    """Load model config.json"""
    config_path = Path(model_path) / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config


def estimate_memory_usage(config: Dict, dtype: torch.dtype = torch.float16) -> Dict[str, float]:
    """Estimate memory usage for a model"""
    vocab_size = config.get("vocab_size", 32000)
    hidden_size = config.get("hidden_size", 4096)
    num_layers = config.get("num_hidden_layers", 32)
    intermediate_size = config.get("intermediate_size", 11008)
    num_heads = config.get("num_attention_heads", 32)
    num_kv_heads = config.get("num_key_value_heads", num_heads)
    
    bytes_per_param = 2 if dtype == torch.float16 else 4
    
    # Embedding
    embedding_params = vocab_size * hidden_size
    
    # Per layer
    # Attention: q_proj, k_proj, v_proj, o_proj
    q_proj_params = hidden_size * hidden_size
    k_proj_params = hidden_size * (hidden_size * num_kv_heads // num_heads)
    v_proj_params = hidden_size * (hidden_size * num_kv_heads // num_heads)
    o_proj_params = hidden_size * hidden_size
    
    # MLP: gate_proj, up_proj, down_proj
    gate_proj_params = hidden_size * intermediate_size
    up_proj_params = hidden_size * intermediate_size
    down_proj_params = intermediate_size * hidden_size
    
    # Norms
    norm_params = hidden_size * 2  # input_layernorm + post_attention_layernorm
    
    layer_params = q_proj_params + k_proj_params + v_proj_params + o_proj_params
    layer_params += gate_proj_params + up_proj_params + down_proj_params + norm_params
    
    # Final norm + lm_head (tied with embedding)
    other_params = hidden_size  # final norm
    
    total_params = embedding_params + (layer_params * num_layers) + other_params
    
    # Memory estimates
    model_memory_gb = (total_params * bytes_per_param) / 1e9
    
    # KV cache for 4096 context
    kv_cache_per_token = 2 * num_kv_heads * (hidden_size // num_heads) * num_layers * bytes_per_param
    kv_cache_gb = (4096 * kv_cache_per_token) / 1e9
    
    # Activation memory (rough estimate)
    activation_gb = (batch_size := 1) * 4096 * hidden_size * bytes_per_param * 4 / 1e9
    
    return {
        "total_params_B": total_params / 1e9,
        "model_memory_gb": model_memory_gb,
        "kv_cache_4k_gb": kv_cache_gb,
        "activation_gb": activation_gb,
        "total_inference_gb": model_memory_gb + kv_cache_gb + activation_gb,
    }
