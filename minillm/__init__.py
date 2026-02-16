"""
MiniLLM: Fast inference engine for transformer models
"""

from .model import MiniLLM, load
from .cache import KVCache
from .quant import quantize_model

__version__ = "0.1.0"
__all__ = ["MiniLLM", "load", "KVCache", "quantize_model"]

# Try to import Rust extensions
try:
    from . import minillm_core
    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False
    import warnings
    warnings.warn("Rust extensions not available, falling back to PyTorch")
