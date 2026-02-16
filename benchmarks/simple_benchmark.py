#!/usr/bin/env python3
"""
Simplified MiniLLM Benchmark
Tests without KV cache to avoid dimension issues
"""

import sys
import time
import torch
import json
from pathlib import Path
from datetime import datetime
import io

# Capture output
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
    log("MiniLLM Simplified Benchmark (No Cache)")
    log("="*80)
    log(f"Start: {datetime.now().isoformat()}")
    log(f"PyTorch: {torch.__version__}")
    log(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from minillm.model import MiniLLM, MiniLLMConfig
    
    # Config
    config = MiniLLMConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=512,
        max_position_embeddings=2048,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    log("")
    log("Building model...")
    model = MiniLLM(config)
    model = model.to(device)
    model.eval()
    
    params = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {params:,} ({params/1e6:.2f}M)")
    
    # Benchmark
    log("")
    log("="*80)
    log("Benchmarking")
    log("="*80)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "pytorch": torch.__version__,
        "params": params,
        "tests": {}
    }
    
    for seq_len in [64, 128, 256, 512]:
        log("")
        log(f"Seq len: {seq_len}")
        
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len)).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            with torch.no_grad():
                _ = model(input_ids)
            times.append(time.time() - start)
        
        avg_time = sum(times) / len(times)
        tok_per_sec = seq_len / avg_time
        
        log(f"  Avg time: {avg_time*1000:.2f} ms")
        log(f"  Throughput: {tok_per_sec:.1f} tok/s")
        
        results["tests"][seq_len] = {
            "time_ms": avg_time * 1000,
            "tok_per_sec": tok_per_sec
        }
    
    log("")
    log("="*80)
    log("Summary")
    log("="*80)
    for seq_len, data in results["tests"].items():
        log(f"{seq_len}: {data['tok_per_sec']:.1f} tok/s")
    
    log(f"\nEnd: {datetime.now().isoformat()}")
    
    return results, output_capture.getvalue()

if __name__ == "__main__":
    results, log_content = main()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f"simple_benchmark_{timestamp}.json", "w") as f:
        json.dump(results, f, indent=2)
    
    with open(f"simple_benchmark_{timestamp}.log", "w") as f:
        f.write(log_content)
    
    print(f"\nResults saved!")
