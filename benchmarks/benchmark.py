"""
Benchmarks comparing MiniLLM vs Transformers
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
sys.path.insert(0, "..")

from minillm import load


def benchmark_prefill(model, tokenizer, prompt_len: int, device: str = "cuda"):
    """Benchmark prefill (first token generation)"""
    prompt = "Hello world " * (prompt_len // 2)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Warmup
    for _ in range(3):
        _ = model(**inputs)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(10):
        _ = model(**inputs)
    
    torch.cuda.synchronize()
    elapsed = (time.time() - start) / 10
    
    tokens_per_sec = prompt_len / elapsed
    return elapsed, tokens_per_sec


def benchmark_generate(model, tokenizer, prompt: str, max_tokens: int, device: str = "cuda"):
    """Benchmark autoregressive generation"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model.generate(inputs.input_ids, max_new_tokens=10)
    
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        output = model.generate(inputs.input_ids, max_new_tokens=max_tokens)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    tokens_generated = output.shape[1] - inputs.input_ids.shape[1]
    tokens_per_sec = tokens_generated / elapsed
    
    return elapsed, tokens_per_sec


def main():
    model_name = "meta-llama/Llama-2-7b-hf"  # Example
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Benchmarking on {device}")
    print("=" * 50)
    
    # Load MiniLLM
    print("Loading MiniLLM...")
    minillm_model = load(model_name, device=device)
    
    # Load Transformers
    print("Loading Transformers...")
    hf_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Benchmark
    prompt_lens = [128, 512, 1024]
    
    print("\nPrefill Benchmark:")
    print("-" * 50)
    for prompt_len in prompt_lens:
        minillm_time, minillm_tps = benchmark_prefill(minillm_model, tokenizer, prompt_len, device)
        hf_time, hf_tps = benchmark_prefill(hf_model, tokenizer, prompt_len, device)
        
        speedup = hf_time / minillm_time
        print(f"Prompt len {prompt_len}: MiniLLM {minillm_tps:.1f} tok/s vs HF {hf_tps:.1f} tok/s ({speedup:.2f}x speedup)")
    
    print("\nGeneration Benchmark:")
    print("-" * 50)
    prompt = "The future of AI is"
    max_tokens = 100
    
    minillm_time, minillm_tps = benchmark_generate(minillm_model, tokenizer, prompt, max_tokens, device)
    hf_time, hf_tps = benchmark_generate(hf_model, tokenizer, prompt, max_tokens, device)
    
    speedup = hf_time / minillm_time
    print(f"Generate {max_tokens} tokens: MiniLLM {minillm_tps:.1f} tok/s vs HF {hf_tps:.1f} tok/s ({speedup:.2f}x speedup)")


if __name__ == "__main__":
    main()
