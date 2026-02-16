# MiniLLM Development Guide

## Quick Start

```bash
# Clone and setup
cd minillm

# Install Rust (for native kernels)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install Python dependencies
pip install -e ".[dev]"

# Build Rust extensions
maturin develop  # or: pip install -e .

# Run tests
pytest tests/

# Run benchmarks
python benchmarks/benchmark.py
```

## Architecture

### Core Optimizations

1. **Fused Attention** (`src/attention.rs`)
   - Online softmax to avoid materializing full attention matrix
   - Chunked processing for memory efficiency
   - FlashAttention-style memory access pattern

2. **Optimized Matmul** (`src/matmul.rs`)
   - Tiled matrix multiplication
   - Parallelized with Rayon
   - Can be extended with SIMD/BLAS

3. **Efficient KV Cache** (`minillm/cache.py`)
   - O(n) per token instead of O(n²)
   - PagedAttention support (future)

4. **Quantization** (`minillm/quant.py`)
   - INT8 weights reduce memory by 4x
   - On-the-fly dequantization (can optimize with INT8 GEMM)

### Model Support

Currently supports:
- Llama-2/Llama-3 architecture
- Grouped Query Attention (GQA)
- Rotary Position Embedding (RoPE)
- SwiGLU MLP
- RMSNorm

Planned:
- Mistral
- Qwen
- DeepSeek
- Mixtral (MoE)

## Benchmarking

Expected speedups vs Transformers:

| Model | Transformers | MiniLLM | Speedup |
|-------|-------------|---------|---------|
| Llama-2-7B | 45 tok/s | 90 tok/s | 2x |
| Llama-2-70B | 8 tok/s | 20 tok/s | 2.5x |

## Contributing

1. Fork the repo
2. Create a branch: `git checkout -b feature/my-feature`
3. Make changes
4. Run tests: `pytest tests/`
5. Submit PR

## License

MIT
