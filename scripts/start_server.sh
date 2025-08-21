#!/bin/bash
# Start script for GPT-OSS v4.4 with CLI parameters

MODEL_SIZE=${1:-20b}
GPU_MODE=${2:-auto}
PORT=${3:-8000}

echo "=========================================="
echo "Starting GPT-OSS Server v4.4"
echo "Model Size: $MODEL_SIZE"
echo "GPU Mode: $GPU_MODE"
echo "Port: $PORT"
echo "=========================================="

# Set model path based on size
if [ "$MODEL_SIZE" == "120b" ]; then
    export MODEL_NAME="openai/gpt-oss-120b"
    echo "Loading 120b model..."
else
    export MODEL_NAME="openai/gpt-oss-20b"
    echo "Loading 20b model..."
fi

# Set environment
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MODEL_SIZE=$MODEL_SIZE
export GPU_MODE=$GPU_MODE

# Kill any existing server
pkill -f hf_server_v4

# Start server
python hf_server_v4.4.py $MODEL_SIZE --gpu-mode $GPU_MODE --port $PORT