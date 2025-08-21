#!/bin/bash

# Start vLLM server for GPT-OSS models
# Usage: ./start_vllm.sh [model_size] [port] [tensor_parallel_size]

MODEL_SIZE=${1:-"20b"}
PORT=${2:-8001}
TP_SIZE=${3:-4}

# Model paths
if [ "$MODEL_SIZE" == "20b" ]; then
    MODEL_PATH="openai/gpt-oss-20b"
    MAX_MODEL_LEN=16384
elif [ "$MODEL_SIZE" == "120b" ]; then
    MODEL_PATH="openai/gpt-oss-120b"
    MAX_MODEL_LEN=8192
else
    echo "Invalid model size. Use '20b' or '120b'"
    exit 1
fi

echo "=========================================="
echo "Starting vLLM Server"
echo "Model: $MODEL_PATH"
echo "Port: $PORT"
echo "Tensor Parallel Size: $TP_SIZE"
echo "=========================================="

# Check if vLLM is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo "vLLM not installed. Installing..."
    pip install vllm
fi

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Start vLLM with optimized settings
python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_PATH \
    --port $PORT \
    --host 0.0.0.0 \
    --tensor-parallel-size $TP_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-seqs 256 \
    --max-num-batched-tokens 65536 \
    --disable-log-stats \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.95 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-parallel-loading-workers 4 \
    --api-key "" \
    --served-model-name "gpt-oss-${MODEL_SIZE}" \
    2>&1 | tee vllm_server_${MODEL_SIZE}.log