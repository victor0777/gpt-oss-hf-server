#!/bin/bash
# GPT-OSS HF Server Startup Script with Memory Optimization

# GPU Memory Management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=0

# Clear any existing Python processes
echo "🔧 Cleaning up existing processes..."
pkill -f "python src/server.py" 2>/dev/null || true
sleep 2

# Check GPU status
echo "📊 GPU Status:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader

# Clear GPU cache
echo "🧹 Clearing GPU cache..."
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# Start server with optimized settings
echo "🚀 Starting server with optimized memory settings..."
python src/server.py \
    --model 20b \
    --profile latency_first \
    --port 8000 \
    --auto-port \
    $@