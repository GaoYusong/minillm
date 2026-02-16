#!/usr/bin/env python3
"""
Complete MiniLLM Real Benchmark with Full Logging
Records entire benchmark process to log file
"""

import sys
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from io import StringIO

# Setup logging to capture everything
log_capture = StringIO()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(log_capture)
    ]
)
logger = logging.getLogger(__name__)

def log_section(title):
    """Log a section header"""
    logger.info("="*80)
    logger.info(title)
    logger.info("="*80)

def run_benchmark():
    """Run complete benchmark with logging"""
    
    log_section("MiniLLM Real Benchmark - Full Test")
    logger.info(f"Start Time: {datetime.now().isoformat()}")
    logger.info(f"Python Version: {sys.version}")
    
    # Test 1: Import dependencies
    log_section("Step 1: Importing Dependencies")
    
    try:
        import torch
        logger.info(f"✓ PyTorch imported: {torch.__version__}")
        logger.info(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"  CUDA version: {torch.version.cuda}")
            logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        logger.error(f"✗ PyTorch import failed: {e}")
        return None
    
    try:
        from safetensors import safe_open
        logger.info("✓ Safetensors imported")
    except ImportError:
        logger.warning("⚠ Safetensors not available")
    
    # Test 2: Import MiniLLM
    log_section("Step 2: Importing MiniLLM")
    
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    try:
        from minillm.model import MiniLLM, MiniLLMConfig
        logger.info("✓ MiniLLM model imported")
    except ImportError as e:
        logger.error(f"✗ MiniLLM import failed: {e}")
        return None
    
    try:
        from minillm.cache import KVCache
        logger.info("✓ KVCache imported")
    except ImportError as e:
        logger.error(f"✗ KVCache import failed: {e}")
    
    try:
        from minillm.weights import estimate_memory_usage
        logger.info("✓ Weight utilities imported")
    except ImportError as e:
        logger.error(f"✗ Weight utilities import failed: {e}")
    
    # Test 3: Create model
    log_section("Step 3: Creating Test Model")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Small model for testing
    config = MiniLLMConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=512,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
    )
    
    logger.info(f"Model config:")
    logger.info(f"  Vocab size: {config.vocab_size}")
    logger.info(f"  Hidden size: {config.hidden_size}")
    logger.info(f"  Num layers: {config.num_hidden_layers}")
    logger.info(f"  Num heads: {config.num_attention_heads}")
    logger.info(f"  Intermediate size: {config.intermediate_size}")
    
    # Memory estimation
    try:
        config_dict = {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "intermediate_size": config.intermediate_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
        }
        memory = estimate_memory_usage(config_dict)
        logger.info(f"\nEstimated memory:")
        logger.info(f"  Model: {memory['model_memory_gb']:.4f} GB")
        logger.info(f"  KV cache (4K): {memory['kv_cache_4k_gb']:.4f} GB")
        logger.info(f"  Total: {memory['total_inference_gb']:.4f} GB")
    except Exception as e:
        logger.error(f"Memory estimation failed: {e}")
    
    # Build model
    logger.info("\nBuilding MiniLLM model...")
    start_time = time.time()
    
    try:
        model = MiniLLM(config)
        model = model.to(device)
        model.eval()
        
        build_time = time.time() - start_time
        logger.info(f"✓ Model built in {build_time:.2f}s")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"  Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        
    except Exception as e:
        logger.error(f"✗ Model build failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
    
    # Test 4: Forward pass benchmark
    log_section("Step 4: Forward Pass Benchmark")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "model_config": {
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_layers": config.num_hidden_layers,
            "num_heads": config.num_attention_heads,
            "total_params": total_params,
        },
        "tests": {}
    }
    
    test_cases = [
        ("Short (64)", 64),
        ("Medium (256)", 256),
        ("Long (512)", 512),
    ]
    
    for test_name, seq_len in test_cases:
        logger.info(f"\nTest: {test_name}")
        logger.info(f"  Sequence length: {seq_len}")
        
        # Create input
        input_ids = torch.randint(0, config.vocab_size, (1, seq_len)).to(device)
        
        # Warmup
        logger.info("  Warming up...")
        with torch.no_grad():
            for _ in range(3):
                _ = model(input_ids)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        logger.info("  Running benchmark...")
        num_runs = 10
        times = []
        
        for i in range(num_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            
            start = time.time()
            with torch.no_grad():
                output = model(input_ids)
            
            if device == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.time() - start
            times.append(elapsed)
            
            if i == 0:
                logger.info(f"    First run: {elapsed*1000:.2f} ms")
        
        # Statistics
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        tokens_per_sec = seq_len / avg_time
        
        logger.info(f"  Results ({num_runs} runs):")
        logger.info(f"    Average: {avg_time*1000:.2f} ms")
        logger.info(f"    Min: {min_time*1000:.2f} ms")
        logger.info(f"    Max: {max_time*1000:.2f} ms")
        logger.info(f"    Throughput: {tokens_per_sec:.1f} tok/s")
        
        results["tests"][test_name] = {
            "seq_len": seq_len,
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "tokens_per_sec": tokens_per_sec,
        }
    
    # Test 5: Generation benchmark
    log_section("Step 5: Generation Benchmark")
    
    try:
        prompt_len = 64
        gen_tokens = 50
        
        logger.info(f"Prompt length: {prompt_len}")
        logger.info(f"Tokens to generate: {gen_tokens}")
        
        input_ids = torch.randint(0, config.vocab_size, (1, prompt_len)).to(device)
        
        # Initialize KV caches
        kv_caches = [KVCache() for _ in range(len(model.layers))]
        
        # Prefill
        logger.info("Prefilling...")
        with torch.no_grad():
            logits = model(input_ids, kv_caches)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        # Generate
        logger.info("Generating...")
        start = time.time()
        
        generated_tokens = []
        with torch.no_grad():
            for i in range(gen_tokens):
                next_token_logits = logits[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_tokens.append(next_token.item())
                
                logits = model(next_token, kv_caches)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        gen_time = time.time() - start
        gen_tok_per_sec = gen_tokens / gen_time
        
        logger.info(f"Generated {len(generated_tokens)} tokens in {gen_time*1000:.2f} ms")
        logger.info(f"Generation speed: {gen_tok_per_sec:.1f} tok/s")
        logger.info(f"Time per token: {gen_time/gen_tokens*1000:.2f} ms")
        
        results["generation"] = {
            "prompt_len": prompt_len,
            "gen_tokens": gen_tokens,
            "total_time_ms": gen_time * 1000,
            "tokens_per_sec": gen_tok_per_sec,
            "ms_per_token": gen_time / gen_tokens * 1000,
        }
        
    except Exception as e:
        logger.error(f"Generation benchmark failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Test 6: Memory usage
    log_section("Step 6: Memory Usage")
    
    if device == "cuda":
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"CUDA Memory:")
        logger.info(f"  Allocated: {allocated:.2f} GB")
        logger.info(f"  Reserved: {reserved:.2f} GB")
        results["memory"] = {
            "allocated_gb": allocated,
            "reserved_gb": reserved,
        }
    else:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        logger.info(f"CPU Memory:")
        logger.info(f"  RSS: {mem_info.rss / 1024**3:.2f} GB")
        logger.info(f"  VMS: {mem_info.vms / 1024**3:.2f} GB")
        results["memory"] = {
            "rss_gb": mem_info.rss / 1024**3,
            "vms_gb": mem_info.vms / 1024**3,
        }
    
    # Summary
    log_section("Benchmark Summary")
    
    logger.info("Forward Pass:")
    for test_name, test_data in results["tests"].items():
        logger.info(f"  {test_name}: {test_data['tokens_per_sec']:.1f} tok/s")
    
    if "generation" in results:
        logger.info(f"\nGeneration: {results['generation']['tokens_per_sec']:.1f} tok/s")
    
    logger.info(f"\nBenchmark completed at {datetime.now().isoformat()}")
    
    return results, log_capture.getvalue()


def save_results(results, log_content):
    """Save results and log"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON results
    json_file = f"real_benchmark_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to: {json_file}")
    
    # Save log
    log_file = f"benchmark_log_{timestamp}.txt"
    with open(log_file, "w") as f:
        f.write(log_content)
    logger.info(f"Log saved to: {log_file}")
    
    # Generate markdown report
    md_file = "REAL_BENCHMARK_RESULTS.md"
    with open(md_file, "w") as f:
        f.write("# MiniLLM Real Benchmark Results\n\n")
        f.write(f"**Date**: {results['timestamp']}\n")
        f.write(f"**Device**: {results['device']}\n")
        f.write(f"**PyTorch**: {results['torch_version']}\n")
        f.write(f"**CUDA**: {results['cuda_available']}\n\n")
        
        f.write("## Model Configuration\n\n")
        f.write(f"- Vocab size: {results['model_config']['vocab_size']}\n")
        f.write(f"- Hidden size: {results['model_config']['hidden_size']}\n")
        f.write(f"- Layers: {results['model_config']['num_layers']}\n")
        f.write(f"- Heads: {results['model_config']['num_heads']}\n")
        f.write(f"- Parameters: {results['model_config']['total_params']:,} ({results['model_config']['total_params']/1e6:.2f}M)\n\n")
        
        f.write("## Forward Pass Results\n\n")
        f.write("| Test | Seq Len | Time (ms) | Throughput (tok/s) |\n")
        f.write("|------|---------|-----------|-------------------|\n")
        for test_name, test_data in results['tests'].items():
            f.write(f"| {test_name} | {test_data['seq_len']} | {test_data['avg_time_ms']:.2f} | {test_data['tokens_per_sec']:.1f} |\n")
        
        if "generation" in results:
            f.write("\n## Generation Results\n\n")
            gen = results['generation']
            f.write(f"- Prompt length: {gen['prompt_len']}\n")
            f.write(f"- Generated tokens: {gen['gen_tokens']}\n")
            f.write(f"- Total time: {gen['total_time_ms']:.2f} ms\n")
            f.write(f"- Throughput: **{gen['tokens_per_sec']:.1f} tok/s**\n")
            f.write(f"- Time per token: {gen['ms_per_token']:.2f} ms\n")
        
        if "memory" in results:
            f.write("\n## Memory Usage\n\n")
            for key, value in results['memory'].items():
                f.write(f"- {key}: {value:.2f} GB\n")
    
    logger.info(f"Markdown report: {md_file}")
    
    return json_file, log_file, md_file


if __name__ == "__main__":
    results, log_content = run_benchmark()
    
    if results:
        json_file, log_file, md_file = save_results(results, log_content)
        print(f"\n{'='*80}")
        print("Benchmark completed successfully!")
        print(f"{'='*80}")
        print(f"JSON: {json_file}")
        print(f"Log: {log_file}")
        print(f"Report: {md_file}")
        sys.exit(0)
    else:
        print(f"\n{'='*80}")
        print("Benchmark failed!")
        print(f"{'='*80}")
        sys.exit(1)
