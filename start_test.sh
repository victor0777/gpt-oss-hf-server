#!/bin/bash
# Start script for v4.5.1 testing without backend engine

# Disable engine endpoint to prevent connection attempts
export ENGINE_ENDPOINT=""
export VLLM_ENDPOINT=""
export TRTLLM_ENDPOINT=""

# Parse arguments
MODEL=${1:-20b}
PROFILE=${2:-latency_first}
PORT=${3:-8000}

echo "Starting GPT-OSS v4.5.1 Server"
echo "================================"
echo "Model: $MODEL"
echo "Profile: $PROFILE"
echo "Port: $PORT"
echo "================================"

# Start server
python src/server_v451.py \
    --engine custom \
    --profile $PROFILE \
    --model $MODEL \
    --port $PORT