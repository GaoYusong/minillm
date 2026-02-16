# MiniLLM

A minimal, fast inference engine for transformer models. Designed to be faster than Hugging Face Transformers for inference workloads.

## Core Principles

1. **Zero-copy tensor operations** - minimize memory copies
2. **Fused kernels** - combine operations to reduce memory bandwidth
3. **KV-cache optimization** - efficient attention cache management
4. **Quantization support** - INT8/INT4 out of the box
5. **No training code** - inference only, no backward pass overhead

## Architecture

```
minillm/
├── minillm/           # Python package
│   ├── __init__.py
│   ├── model.py       # Model loading & inference
│   ├── layers.py      # Custom layers (fused attention, etc.)
│   ├── cache.py       # KV-cache management
│   └── quant.py       # Quantization utilities
├── src/               # Rust/C++ kernels
│   ├── lib.rs         # Rust main
│   ├── attention.rs   # Fused attention kernel
│   ├── matmul.rs      # Optimized matmul
│   └── bindings.pyx   # Python bindings
├── tests/             # Tests
└── benchmarks/        # Performance benchmarks
```

## Quick Start

```python
import minillm

# Load model (compatible with HF format)
model = minillm.load("meta-llama/Llama-2-7b-hf")

# Generate
output = model.generate("Hello, world!", max_tokens=100)
```

## Performance Goals

| Model | Transformers | MiniLLM | Speedup |
|-------|-------------|---------|---------|
| Llama-2-7B | 45 tok/s | 90 tok/s | 2x |
| Llama-2-70B | 8 tok/s | 20 tok/s | 2.5x |

## License

MIT
