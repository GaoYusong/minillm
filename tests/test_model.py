import pytest
import torch
import sys
sys.path.insert(0, "..")

from minillm.model import MiniLLMConfig, MiniLLM
from minillm.cache import KVCache
from minillm.layers import LlamaRMSNorm, RotaryEmbedding


def test_config():
    config = MiniLLMConfig(
        vocab_size=32000,
        hidden_size=4096,
        num_hidden_layers=32,
    )
    assert config.vocab_size == 32000
    assert config.hidden_size == 4096


def test_model_forward():
    config = MiniLLMConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=256,
    )
    model = MiniLLM(config)
    
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, config.vocab_size)


def test_kv_cache():
    cache = KVCache()
    
    batch = 1
    num_heads = 4
    head_dim = 32
    
    # First update
    key1 = torch.randn(batch, num_heads, 5, head_dim)
    value1 = torch.randn(batch, num_heads, 5, head_dim)
    k, v = cache.update(key1, value1, layer_idx=0)
    
    assert k.shape == (batch, num_heads, 5, head_dim)
    assert cache.get_seq_length() == 5
    
    # Second update
    key2 = torch.randn(batch, num_heads, 3, head_dim)
    value2 = torch.randn(batch, num_heads, 3, head_dim)
    k, v = cache.update(key2, value2, layer_idx=0)
    
    assert k.shape == (batch, num_heads, 8, head_dim)
    assert cache.get_seq_length() == 8


def test_rms_norm():
    norm = LlamaRMSNorm(hidden_size=64, eps=1e-6)
    x = torch.randn(2, 10, 64)
    output = norm(x)
    assert output.shape == x.shape


def test_rotary_embedding():
    rope = RotaryEmbedding(dim=32)
    x = torch.randn(1, 4, 10, 32)  # [batch, heads, seq, dim]
    position_ids = torch.arange(10).unsqueeze(0)
    
    cos, sin = rope(x, position_ids)
    assert cos.shape == (1, 1, 10, 32)
    assert sin.shape == (1, 1, 10, 32)


if __name__ == "__main__":
    test_config()
    test_model_forward()
    test_kv_cache()
    test_rms_norm()
    test_rotary_embedding()
    print("All tests passed!")
