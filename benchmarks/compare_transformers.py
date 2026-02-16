#!/usr/bin/env python3
"""
MiniLLM vs Transformers Benchmark
Direct comparison between MiniLLM and HuggingFace Transformers
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

def benchmark_transformers_vs_minillm():
    """Compare MiniLLM with Transformers"""
    
    log("="*80)
    log("MiniLLM vs Transformers Benchmark")
    log("="*80)
    log(f"Start: {datetime.now().isoformat()}")
    log(f"PyTorch: {torch.__version__}")
    log(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    # Test configuration
    config = {
        "vocab_size": 1000,
        "hidden_size": 256,
        "num_hidden_layers": 4,
        "num_attention_heads": 4,
        "intermediate_size": 512,
        "max_position_embeddings": 2048,
    }
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # ========== MiniLLM ==========
    log("")
    log("="*80)
    log("Testing MiniLLM")
    log("="*80)
    
    from minillm.model import MiniLLM, MiniLLMConfig
    
    minillm_config = MiniLLMConfig(**config)
    minillm_model = MiniLLM(minillm_config)
    minillm_model = minillm_model.to(device)
    minillm_model.eval()
    
    minillm_params = sum(p.numel() for p in minillm_model.parameters())
    log(f"Parameters: {minillm_params:,}")
    
    # ========== Transformers ==========
    log("")
    log("="*80)
    log("Testing Transformers (LlamaForCausalLM)")
    log("="*80)
    
    try:
        from transformers import LlamaForCausalLM, LlamaConfig
        
        hf_config = LlamaConfig(
            vocab_size=config["vocab_size"],
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_attention_heads=config["num_attention_heads"],
            intermediate_size=config["intermediate_size"],
            max_position_embeddings=config["max_position_embeddings"],
        )
        
        hf_model = LlamaForCausalLM(hf_config)
        hf_model = hf_model.to(device)
        hf_model.eval()
        
        hf_params = sum(p.numel() for p in hf_model.parameters())
        log(f"Parameters: {hf_params:,}")
        
    except Exception as e:
        log(f"Error loading Transformers: {e}")
        hf_model = None
    
    # ========== Benchmark ==========
    log("")
    log("="*80)
    log("Running Benchmarks")
    log("="*80)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "pytorch_version": torch.__version__,
        "config": config,
        "minillm_params": minillm_params,
        "hf_params": hf_params if hf_model else 0,
        "tests": {}
    }
    
    test_lengths = [64, 128, 256]
    
    for seq_len in test_lengths:
        log("")
        log(f"Sequence Length: {seq_len}")
        log("-"*40)
        
        input_ids = torch.randint(0, config["vocab_size"], (1, seq_len)).to(device)
        
        # MiniLLM benchmark
        log("MiniLLM:")
        with torch.no_grad():
            # Warmup
            for _ in range(3):
                _ = minillm_model(input_ids)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            times = []
            for _ in range(10):
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
                _ = minillm_model(input_ids)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.time() - start)
            
            avg_time = sum(times) / len(times)
            tok_per_sec = seq_len / avg_time
            
            log(f"  Time: {avg_time*1000:.2f} ms")
            log(f"  Throughput: {tok_per_sec:.1f} tok/s")
            
            results["tests"][f"seq_{seq_len}"] = {
                "minillm_ms": avg_time * 1000,
                "minillm_tok_per_sec": tok_per_sec,
            }
        
        # Transformers benchmark
        if hf_model:
            log("Transformers:")
            with torch.no_grad():
                # Warmup
                for _ in range(3):
                    _ = hf_model(input_ids)
                
                if device == "cuda":
                    torch.cuda.synchronize()
                
                times = []
                for _ in range(10):
                    if device == "cuda":
                        torch.cuda.synchronize()
                    start = time.time()
                    _ = hf_model(input_ids)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    times.append(time.time() - start)
                
                avg_time_hf = sum(times) / len(times)
                tok_per_sec_hf = seq_len / avg_time_hf
                speedup = tok_per_sec / tok_per_sec_hf
                
                log(f"  Time: {avg_time_hf*1000:.2f} ms")
                log(f"  Throughput: {tok_per_sec_hf:.1f} tok/s")
                log(f"  Speedup: {speedup:.2f}x")
                
                results["tests"][f"seq_{seq_len}"]["hf_ms"] = avg_time_hf * 1000
                results["tests"][f"seq_{seq_len}"]["hf_tok_per_sec"] = tok_per_sec_hf
                results["tests"][f"seq_{seq_len}"]["speedup"] = speedup
    
    # Summary
    log("")
    log("="*80)
    log("Summary")
    log("="*80)
    
    for test_name, test_data in results["tests"].items():
        log(f"{test_name}:")
        log(f"  MiniLLM: {test_data['minillm_tok_per_sec']:.1f} tok/s")
        if 'hf_tok_per_sec' in test_data:
            log(f"  Transformers: {test_data['hf_tok_per_sec']:.1f} tok/s")
            log(f"  Speedup: {test_data['speedup']:.2f}x")
    
    log(f"\nEnd: {datetime.now().isoformat()}")
    
    return results, output_capture.getvalue()

if __name__ == "__main__":
    results, log_content = benchmark_transformers_vs_minillm()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_file = f"comparison_benchmark_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    
    log_file = f"comparison_benchmark_{timestamp}.log"
    with open(log_file, "w") as f:
        f.write(log_content)
    
    md_file = "COMPARISON_BENCHMARK.md"
    with open(md_file, "w") as f:
        f.write("# MiniLLM vs Transformers Benchmark\n\n")
        f.write(f"**Date**: {results['timestamp']}\n")
        f.write(f"**Device**: {results['device']}\n")
        f.write(f"**PyTorch**: {results['pytorch_version']}\n\n")
        
        f.write("## Results\n\n")
        f.write("| Seq Len | MiniLLM (tok/s) | Transformers (tok/s) | Speedup |\n")
        f.write("|---------|-----------------|---------------------|---------|\n")
        
        for test_name, test_data in results['tests'].items():
            seq_len = test_name.split('_')[1]
            ml_tps = test_data['minillm_tok_per_sec']
            hf_tps = test_data.get('hf_tok_per_sec', 'N/A')
            speedup = test_data.get('speedup', 'N/A')
            
            if isinstance(hf_tps, float):
                hf_tps = f"{hf_tps:.1f}"
            if isinstance(speedup, float):
                speedup = f"{speedup:.2f}x"
            
            f.write(f"| {seq_len} | {ml_tps:.1f} | {hf_tps} | {speedup} |\n")
    
    print(f"\n{'='*80}")
    print("Benchmark complete!")
    print(f"{'='*80}")
    print(f"JSON: {json_file}")
    print(f"Log: {log_file}")
    print(f"Report: {md_file}")
