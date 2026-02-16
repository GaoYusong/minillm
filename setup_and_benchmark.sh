#!/bin/bash
# MiniLLM Setup and Benchmark Script
# Records entire process to log file

LOGFILE="setup_and_benchmark_$(date +%Y%m%d_%H%M%S).log"

exec 1> >(tee -a "$LOGFILE")
exec 2>&1

echo "================================================================================"
echo "MiniLLM Setup and Benchmark"
echo "Started: $(date)"
echo "Log file: $LOGFILE"
echo "================================================================================"

echo ""
echo "Step 1: System Information"
echo "================================================================================"
echo "OS: $(uname -a)"
echo "Python: $(python3 --version)"
echo "CPU: $(nproc) cores"
echo "Memory: $(free -h | grep Mem | awk '{print $2}')"

echo ""
echo "Step 2: Check Rust"
echo "================================================================================"
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
    echo "Rust: $(rustc --version)"
    echo "Cargo: $(cargo --version)"
else
    echo "Rust not found, installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

echo ""
echo "Step 3: Install Python Dependencies"
echo "================================================================================"

# Install numpy first
echo "Installing numpy..."
pip install numpy --break-system-packages

# Install safetensors
echo "Installing safetensors..."
pip install safetensors --break-system-packages

# Install psutil for memory monitoring
echo "Installing psutil..."
pip install psutil --break-system-packages

# Install PyTorch (CPU version for compatibility)
echo "Installing PyTorch (this may take a while)..."
pip install torch --index-url https://download.pytorch.org/whl/cpu --break-system-packages

echo ""
echo "Step 4: Verify Installation"
echo "================================================================================"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python3 -c "import safetensors; print(f'Safetensors: OK')"

echo ""
echo "Step 5: Build Rust Extensions"
echo "================================================================================"
cd /root/.openclaw/workspace/minillm

# Install maturin
echo "Installing maturin..."
pip install maturin --break-system-packages

# Build extensions
echo "Building Rust extensions..."
maturin develop --release

echo ""
echo "Step 6: Run Benchmark"
echo "================================================================================"
python3 benchmarks/run_full_benchmark.py

echo ""
echo "================================================================================"
echo "Setup and Benchmark Complete"
echo "Finished: $(date)"
echo "Log file: $LOGFILE"
echo "================================================================================"
