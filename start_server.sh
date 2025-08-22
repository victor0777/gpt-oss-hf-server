#!/bin/bash

# GPT-OSS HF Server v4.6.0 Startup Script
# Optimized for memory management and OOM prevention

echo "=================================="
echo "GPT-OSS HF Server v4.6.0 Launcher"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check GPU status
echo "🔍 Checking GPU status..."
nvidia-smi --query-gpu=index,name,memory.total,memory.free,memory.used --format=csv,noheader,nounits | while read line; do
    IFS=',' read -r gpu_id gpu_name mem_total mem_free mem_used <<< "$line"
    mem_total_gb=$((mem_total / 1024))
    mem_free_gb=$((mem_free / 1024))
    mem_used_gb=$((mem_used / 1024))
    usage_percent=$((mem_used * 100 / mem_total))
    
    echo "  GPU $gpu_id: $gpu_name"
    echo "    Memory: ${mem_used_gb}/${mem_total_gb} GB used (${usage_percent}%)"
    echo "    Free: ${mem_free_gb} GB"
    
    if [ $mem_free -lt 30000 ]; then
        echo -e "    ${YELLOW}⚠️ Low free memory! Clearing cache...${NC}"
        # Try to clear GPU cache
        python -c "import torch; torch.cuda.empty_cache(); print('    Cache cleared')" 2>/dev/null || true
    fi
done

echo ""

# Kill any existing server on port 8000
echo "🔍 Checking for existing server..."
if lsof -i:8000 > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️ Found existing server on port 8000${NC}"
    echo "Killing existing process..."
    kill -9 $(lsof -t -i:8000) 2>/dev/null || true
    sleep 2
fi

# Environment variables for better memory management
echo "⚙️ Setting environment variables..."
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.0;8.6"
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "  PYTORCH_CUDA_ALLOC_CONF: expandable_segments enabled"
echo "  Memory fragmentation mitigation: enabled"
echo ""

# Parse command line arguments
MODEL_SIZE="${1:-20b}"
PROFILE="${2:-latency_first}"

echo "📋 Configuration:"
echo "  Model size: $MODEL_SIZE"
echo "  Profile: $PROFILE"
echo ""

# Start the server
echo -e "${GREEN}🚀 Starting GPT-OSS HF Server v4.6.0...${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Run with proper Python buffering
python -u src/server.py \
    --model $MODEL_SIZE \
    --profile $PROFILE \
    2>&1 | tee server_v460.log

# Server stopped
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${RED}Server stopped${NC}"

# Check if it was OOM
if grep -q "CUDA out of memory" server_v460.log; then
    echo ""
    echo -e "${RED}❌ Server crashed due to OOM${NC}"
    echo ""
    echo "Suggestions:"
    echo "1. Restart with cleared GPU cache:"
    echo "   python -c 'import torch; torch.cuda.empty_cache()'"
    echo ""
    echo "2. Use 4-GPU mode (when implemented):"
    echo "   ./start_server_v460.sh 20b quality_first --4gpu"
    echo ""
    echo "3. Check GPU memory usage:"
    echo "   nvidia-smi"
    echo ""
fi