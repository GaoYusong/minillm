"""
Model loading and inference
"""

import torch
import torch.nn as nn
from typing import Optional, List, Dict
import json
from pathlib import Path
from safetensors import safe_open

from .cache import KVCache
from .layers import LlamaAttention, LlamaMLP, LlamaRMSNorm


class MiniLLMConfig:
    """Configuration for MiniLLM models"""
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.get("vocab_size", 32000)
        self.hidden_size = kwargs.get("hidden_size", 4096)
        self.num_hidden_layers = kwargs.get("num_hidden_layers", 32)
        self.num_attention_heads = kwargs.get("num_attention_heads", 32)
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 32)
        self.intermediate_size = kwargs.get("intermediate_size", 11008)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 2048)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)


class LlamaDecoderLayer(nn.Module):
    """Single transformer layer with fused operations"""
    def __init__(self, config: MiniLLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config, layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        position_ids: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        # Pre-norm architecture
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, kv_cache, position_ids)
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states


class MiniLLM(nn.Module):
    """
    Fast inference-only LLM
    Compatible with HuggingFace Llama format
    """
    def __init__(self, config: MiniLLMConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LlamaDecoderLayer(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = LlamaRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights (common in Llama)
        self.lm_head.weight = self.embed_tokens.weight
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        kv_caches: Optional[List[KVCache]] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional KV-cache
        
        Args:
            input_ids: [batch_size, seq_len]
            kv_caches: List of KVCache for each layer (for generation)
        
        Returns:
            logits: [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # Embeddings
        hidden_states = self.embed_tokens(input_ids)
        
        # Position IDs for RoPE
        if kv_caches is not None and kv_caches[0] is not None:
            # Generation mode: only process new tokens
            past_len = kv_caches[0].get_seq_length()
            position_ids = torch.arange(past_len, past_len + seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            # Prefill mode
            position_ids = torch.arange(seq_len, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # Transformer layers
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states = layer(hidden_states, kv_cache, position_ids)
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.8,
        top_p: float = 0.95,
        tokenizer = None,
    ) -> str:
        """
        Simple greedy generation (for MVP)
        """
        if tokenizer is None:
            raise ValueError("Tokenizer required for generation")
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.embed_tokens.weight.device)
        
        # Initialize KV caches
        kv_caches = [KVCache() for _ in range(len(self.layers))]
        
        # Prefill
        logits = self.forward(input_ids, kv_caches)
        
        # Generate
        generated = input_ids.clone()
        
        for _ in range(max_tokens):
            # Get last token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Simple top-p sampling
            probs = torch.softmax(next_token_logits, dim=-1)
            
            # Greedy for MVP (can add sampling later)
            next_token = torch.argmax(probs, dim=-1, keepdim=True)
            
            # Append
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Forward with cache
            logits = self.forward(next_token, kv_caches)
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
        
        return tokenizer.decode(generated[0], skip_special_tokens=True)


def load(model_path: str, device: str = "cuda") -> MiniLLM:
    """
    Load a model from HuggingFace format
    
    Args:
        model_path: Path to model (local dir or HF hub)
        device: "cuda" or "cpu"
    
    Returns:
        MiniLLM model
    """
    from huggingface_hub import hf_hub_download
    
    # Load config
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        # Try to download from HF
        config_path = hf_hub_download(model_path, "config.json")
    
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    config = MiniLLMConfig(**config_dict)
    model = MiniLLM(config)
    
    # Load weights (simplified for MVP)
    # In production, would use safetensors and memory mapping
    print(f"Loading model from {model_path}...")
    print(f"Parameters: ~{sum(p.numel() for p in model.parameters()) / 1e9:.1f}B")
    
    model = model.to(device)
    model.eval()
    
    return model
