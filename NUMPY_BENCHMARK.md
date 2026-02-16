# MiniLLM NumPy Benchmark

**Date**: 2026-02-17T03:04:52.215464
**Total Parameters**: 2,877,696 (2.88M)

## Configuration

- vocab_size: 1000
- hidden_size: 256
- num_layers: 4
- num_heads: 4
- intermediate_size: 512

## Results

| Seq Len | Attention (ms) | MLP (ms) | Total (ms) | Throughput (tok/s) | GFLOPS |
|---------|----------------|----------|------------|-------------------|--------|
| 64 | 0.27 | 0.42 | 2.77 | 23144.2 | 7.81 |
| 128 | 0.95 | 0.71 | 6.65 | 19257.0 | 8.79 |
| 256 | 3.74 | 4.68 | 33.67 | 7602.9 | 8.97 |
| 512 | 13.55 | 4.67 | 72.90 | 7023.7 | 9.90 |

## Notes

- This benchmark uses NumPy to simulate MiniLLM operations
- Actual PyTorch/CUDA performance will be significantly higher
- NumPy is single-threaded CPU-only, while MiniLLM uses optimized CUDA kernels
