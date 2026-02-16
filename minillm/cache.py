"""
KV-cache for efficient autoregressive generation
"""

import torch
from typing import Tuple, Optional


class KVCache:
    """
    Key-Value cache for transformer attention
    Reduces computation from O(n^2) to O(n) per token during generation
    """
    def __init__(self):
        self.key_cache: Optional[torch.Tensor] = None
        self.value_cache: Optional[torch.Tensor] = None
        self._seen_tokens = 0
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key/value states and return full cache
        
        Args:
            key_states: [batch, num_heads, seq_len, head_dim]
            value_states: [batch, num_heads, seq_len, head_dim]
            layer_idx: layer index (for debug)
        
        Returns:
            Full key and value cache tensors
        """
        batch_size = key_states.shape[0]
        
        if self.key_cache is None:
            # First call - initialize cache
            self.key_cache = key_states
            self.value_cache = value_states
        else:
            # Append new keys/values
            self.key_cache = torch.cat([self.key_cache, key_states], dim=2)
            self.value_cache = torch.cat([self.value_cache, value_states], dim=2)
        
        self._seen_tokens += key_states.shape[2]
        return self.key_cache, self.value_cache
    
    def get_seq_length(self) -> int:
        """Get the sequence length of the cache"""
        if self.key_cache is None:
            return 0
        return self.key_cache.shape[2]
    
    def clear(self):
        """Clear the cache"""
        self.key_cache = None
        self.value_cache = None
        self._seen_tokens = 0


class PagedKVCache:
    """
    PagedAttention-style KV cache for vLLM-like efficiency
    (Future optimization)
    """
    def __init__(self, block_size: int = 16, num_blocks: int = 1000):
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.block_table = {}  # seq_id -> list of block indices
        self.free_blocks = list(range(num_blocks))
    
    def allocate(self, seq_id: int, num_tokens: int):
        """Allocate blocks for a sequence"""
        num_blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        blocks = [self.free_blocks.pop() for _ in range(num_blocks_needed)]
        self.block_table[seq_id] = blocks
        return blocks
    
    def get_block_table(self, seq_id: int):
        return self.block_table.get(seq_id, [])
