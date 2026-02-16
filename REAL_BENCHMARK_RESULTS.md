# MiniLLM Real Benchmark Results

**Date**: 2026-02-17T03:06:32.137729
**Device**: cpu
**PyTorch**: 2.10.0+cpu
**CUDA**: False

## Model Configuration

- Vocab size: 1000
- Hidden size: 256
- Layers: 4
- Heads: 4
- Parameters: 2,879,744 (2.88M)

## Forward Pass Results

| Test | Seq Len | Time (ms) | Throughput (tok/s) |
|------|---------|-----------|-------------------|
| Short (64) | 64 | 5.39 | 11880.7 |
| Medium (256) | 256 | 25.57 | 10010.6 |
| Long (512) | 512 | 55.95 | 9150.8 |

## Generation Results

- Prompt length: 64
- Generated tokens: 50
- Total time: 80.47 ms
- Throughput: **621.4 tok/s**
- Time per token: 1.61 ms

## Memory Usage

- rss_gb: 0.27 GB
- vms_gb: 0.66 GB
