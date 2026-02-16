#!/usr/bin/env python3
"""
MiniLLM Benchmark - NumPy Only Version
Runs without PyTorch to demonstrate architecture
"""

import sys
import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime

# Capture all output
import io
output_capture = io.StringIO()

class TeeOutput:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
    def flush(self):
        for stream in self.streams:
            stream.flush()

sys.stdout = TeeOutput(sys.stdout, output_capture)
sys.stderr = TeeOutput(sys.stderr, output_capture)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def main():
    log("="*80)
    log("MiniLLM Architecture Benchmark (NumPy Simulation)")
    log("="*80)
    log(f"Start time: {datetime.now().isoformat()}")
    log(f"Python: {sys.version}")
    log(f"NumPy: {np.__version__}")
    
    # Model configuration (same as MiniLLM)
    config = {
        "vocab_size": 1000,
        "hidden_size": 256,
        "num_layers": 4,
        "num_heads": 4,
        "intermediate_size": 512,
        "seq_lengths": [64, 128, 256, 512],
    }
    
    log("")
    log("Model Configuration:")
    log(f"  Vocab size: {config['vocab_size']}")
    log(f"  Hidden size: {config['hidden_size']}")
    log(f"  Num layers: {config['num_layers']}")
    log(f"  Num heads: {config['num_heads']}")
    log(f"  Intermediate size: {config['intermediate_size']}")
    
    # Calculate parameters
    embedding_params = config['vocab_size'] * config['hidden_size']
    
    # Per layer: Q, K, V, O projections + MLP
    head_dim = config['hidden_size'] // config['num_heads']
    q_params = config['hidden_size'] * config['hidden_size']
    k_params = config['hidden_size'] * config['hidden_size']
    v_params = config['hidden_size'] * config['hidden_size']
    o_params = config['hidden_size'] * config['hidden_size']
    
    mlp_params = (
        config['hidden_size'] * config['intermediate_size'] * 3  # gate, up, down
    )
    
    layer_params = q_params + k_params + v_params + o_params + mlp_params
    total_params = embedding_params + (layer_params * config['num_layers']) + config['hidden_size']
    
    log(f"")
    log(f"Parameter Count:")
    log(f"  Embedding: {embedding_params:,}")
    log(f"  Per layer: {layer_params:,}")
    log(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Simulate matrix operations
    log("")
    log("="*80)
    log("Matrix Operation Benchmarks")
    log("="*80)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "total_params": int(total_params),
        "tests": {}
    }
    
    for seq_len in config['seq_lengths']:
        log("")
        log(f"Sequence Length: {seq_len}")
        log("-"*40)
        
        # Simulate attention computation
        batch_size = 1
        hidden_size = config['hidden_size']
        num_heads = config['num_heads']
        head_dim = hidden_size // num_heads
        
        # Create random matrices
        np.random.seed(42)
        Q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        K = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        V = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
        
        # Warmup
        for _ in range(3):
            scores = np.matmul(Q, K.transpose(0, 1, 3, 2))
        
        # Benchmark attention
        num_runs = 10
        times = []
        
        for _ in range(num_runs):
            start = time.time()
            
            # Q @ K^T
            scores = np.matmul(Q, K.transpose(0, 1, 3, 2))
            scores = scores / np.sqrt(head_dim)
            
            # Softmax
            exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
            
            # @ V
            output = np.matmul(attn_weights, V)
            
            elapsed = time.time() - start
            times.append(elapsed)
        
        avg_time = np.mean(times)
        min_time = np.min(times)
        flops = (2 * batch_size * num_heads * seq_len * seq_len * head_dim) / avg_time
        
        log(f"  Attention computation:")
        log(f"    Average time: {avg_time*1000:.2f} ms")
        log(f"    Min time: {min_time*1000:.2f} ms")
        log(f"    FLOPS: {flops/1e9:.2f} GFLOPS")
        
        # Simulate MLP
        hidden_states = np.random.randn(batch_size, seq_len, hidden_size).astype(np.float32)
        mlp_up = np.random.randn(hidden_size, config['intermediate_size']).astype(np.float32)
        mlp_down = np.random.randn(config['intermediate_size'], hidden_size).astype(np.float32)
        
        times_mlp = []
        for _ in range(num_runs):
            start = time.time()
            # up projection
            up = np.matmul(hidden_states, mlp_up)
            # activation (SiLU-ish)
            up = up * (1 / (1 + np.exp(-up)))
            # down projection
            out = np.matmul(up, mlp_down)
            elapsed = time.time() - start
            times_mlp.append(elapsed)
        
        avg_mlp_time = np.mean(times_mlp)
        log(f"  MLP computation:")
        log(f"    Average time: {avg_mlp_time*1000:.2f} ms")
        
        # Full forward pass estimate
        layer_time = avg_time + avg_mlp_time
        total_time = layer_time * config['num_layers']
        
        log(f"  Estimated full forward pass:")
        log(f"    Per layer: {layer_time*1000:.2f} ms")
        log(f"    Total ({config['num_layers']} layers): {total_time*1000:.2f} ms")
        log(f"    Throughput: {seq_len/total_time:.1f} tok/s")
        
        results["tests"][f"seq_{seq_len}"] = {
            "seq_len": seq_len,
            "attention_ms": avg_time * 1000,
            "mlp_ms": avg_mlp_time * 1000,
            "total_ms": total_time * 1000,
            "tokens_per_sec": seq_len / total_time,
            "gflops": flops / 1e9,
        }
    
    # Memory estimation
    log("")
    log("="*80)
    log("Memory Usage Estimation")
    log("="*80)
    
    bytes_per_param = 2  # float16
    model_memory = total_params * bytes_per_param / 1024**3
    
    for seq_len in config['seq_lengths']:
        # KV cache: 2 (K,V) * num_layers * seq_len * hidden_size * bytes
        kv_cache = 2 * config['num_layers'] * seq_len * config['hidden_size'] * bytes_per_param / 1024**3
        activation_memory = seq_len * config['hidden_size'] * bytes_per_param * 4 / 1024**3  # rough estimate
        
        total_memory = model_memory + kv_cache + activation_memory
        
        log(f"Seq {seq_len}: Model={model_memory:.2f}GB, KV={kv_cache:.2f}GB, Total={total_memory:.2f}GB")
    
    # Summary
    log("")
    log("="*80)
    log("Summary")
    log("="*80)
    
    for test_name, test_data in results["tests"].items():
        log(f"{test_name}: {test_data['tokens_per_sec']:.1f} tok/s")
    
    log("")
    log("Note: This is a NumPy simulation. Actual PyTorch/CUDA performance will be higher.")
    log(f"End time: {datetime.now().isoformat()}")
    
    return results, output_capture.getvalue()

if __name__ == "__main__":
    results, log_content = main()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    json_file = f"numpy_benchmark_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {json_file}")
    
    # Log
    log_file = f"numpy_benchmark_{timestamp}.log"
    with open(log_file, "w") as f:
        f.write(log_content)
    print(f"Log saved: {log_file}")
    
    # Markdown
    md_file = "NUMPY_BENCHMARK.md"
    with open(md_file, "w") as f:
        f.write("# MiniLLM NumPy Benchmark\n\n")
        f.write(f"**Date**: {results['timestamp']}\n")
        f.write(f"**Total Parameters**: {results['total_params']:,} ({results['total_params']/1e6:.2f}M)\n\n")
        
        f.write("## Configuration\n\n")
        for k, v in results['config'].items():
            if k != 'seq_lengths':
                f.write(f"- {k}: {v}\n")
        
        f.write("\n## Results\n\n")
        f.write("| Seq Len | Attention (ms) | MLP (ms) | Total (ms) | Throughput (tok/s) | GFLOPS |\n")
        f.write("|---------|----------------|----------|------------|-------------------|--------|\n")
        for test_name, test_data in results['tests'].items():
            f.write(f"| {test_data['seq_len']} | {test_data['attention_ms']:.2f} | ")
            f.write(f"{test_data['mlp_ms']:.2f} | {test_data['total_ms']:.2f} | ")
            f.write(f"{test_data['tokens_per_sec']:.1f} | {test_data['gflops']:.2f} |\n")
        
        f.write("\n## Notes\n\n")
        f.write("- This benchmark uses NumPy to simulate MiniLLM operations\n")
        f.write("- Actual PyTorch/CUDA performance will be significantly higher\n")
        f.write("- NumPy is single-threaded CPU-only, while MiniLLM uses optimized CUDA kernels\n")
    
    print(f"Markdown report: {md_file}")
