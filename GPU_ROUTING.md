# GPU Routing in v4.7.0

## Overview

The GPT-OSS HF Server v4.7.0 introduces intelligent GPU routing (PR-MG01) that works alongside the existing HuggingFace device_map mechanism to optimize resource utilization for large requests.

## Key Concepts

### 1. Static GPU Distribution (Existing)
The server already supports GPU distribution at startup via the `--gpu-mode` argument:
- `single`: Use single GPU (default)
- `auto`: HuggingFace automatic device_map distribution
- `tensor`: Tensor parallelism across GPUs
- `pipeline`: Pipeline parallelism

This distribution is set at model load time and remains static during server operation.

### 2. Dynamic Routing Decisions (New in v4.7.0)
The GPU Router makes intelligent routing **decisions** based on request characteristics:
- Tracks which requests would benefit from multi-GPU processing
- Provides recommendations when single-GPU mode is insufficient
- Records statistics for observability

## How It Works

### Routing Triggers
The router triggers multi-GPU recommendations when:
1. **Large Input**: `input_tokens > 8000`
2. **Large KV Cache**: `predicted_kv_mb > 6000`
3. **Memory Pressure**: GPU memory >85% AND (input_tokens > 4000 OR kv_mb > 3000)

### Integration with Existing System

```python
# Server startup with multi-GPU mode
python src/server.py --model 120b --gpu-mode auto

# The router then:
1. Detects current GPU mode (single/auto/tensor/pipeline)
2. Makes routing decisions based on request size
3. If already in multi-GPU mode: logs routing statistics
4. If in single-GPU mode: warns and recommends multi-GPU mode
```

## Important Limitations

### What GPU Router v2 DOES:
✅ Tracks routing decisions and statistics
✅ Identifies requests that need multi-GPU resources
✅ Works with existing HuggingFace device_map
✅ Provides observability via metrics
✅ Recommends optimal GPU configuration

### What GPU Router v2 DOES NOT:
❌ Dynamically redistribute models at runtime
❌ Change GPU allocation after model loading
❌ Move model layers between GPUs during operation
❌ Override the static device_map configuration

## Configuration

### NCCL Environment (for multi-GPU mode)
When 4+ GPUs are available, the router configures NCCL:
```bash
NCCL_ASYNC_ERROR_HANDLING=1
NCCL_MIN_NCHANNELS=4
NCCL_BUFFSIZE=8388608  # 8MB
NCCL_P2P_DISABLE=0      # Enable P2P
```

### Routing Configuration
```python
RoutingConfig(
    large_input_tokens=8000,    # Trigger threshold
    large_kv_mb=6000,           # KV cache threshold
    min_gpu_util_percent=60.0,  # Target utilization
    max_p95_e2e_seconds=20.0    # Performance target
)
```

## Usage Examples

### Single GPU Mode (Default)
```bash
# Start server in single GPU mode
python src/server.py --model 120b

# Large requests will trigger warnings:
# "Large request detected but model in single GPU mode"
# "Recommendation: restart with --gpu-mode auto or tensor"
```

### Multi-GPU Mode (Recommended for Large Models)
```bash
# Start server with automatic GPU distribution
python src/server.py --model 120b --gpu-mode auto

# Large requests will be tracked:
# "Multi-GPU active: large_input (10000 > 8000)"
# Statistics show route4 percentage and triggers
```

## Metrics and Observability

The router provides statistics via `/stats` endpoint:
```json
{
  "gpu_routing": {
    "total_requests": 100,
    "single_gpu_requests": 60,
    "route4_gpu_requests": 40,
    "route4_percentage": 40.0,
    "route4_triggers": {
      "large_input": 30,
      "large_kv": 8,
      "memory_pressure": 2
    },
    "gpu_balance": {
      "balanced": true,
      "gpu_utilization": {
        "0": 65.0,
        "1": 62.0,
        "2": 63.0,
        "3": 64.0
      }
    }
  }
}
```

## Future Enhancements

Potential improvements for dynamic GPU redistribution:
1. Integration with DeepSpeed for dynamic parallelism
2. FairScale for flexible model sharding
3. Runtime device_map updates (requires model reload)
4. Request-based GPU allocation pools

## Troubleshooting

### "CPU tensor" Error
If you see: `Pointer argument cannot be accessed from Triton (cpu tensor?)`
- This occurs when trying to redistribute an already-loaded model
- Solution: Use `--gpu-mode auto` at startup instead of runtime redistribution

### Low GPU Utilization
If GPUs show unbalanced utilization:
- Check if model is in `auto` mode: `--gpu-mode auto`
- Verify NCCL environment variables are set
- Monitor with `nvidia-smi` or `/stats` endpoint

### Memory Issues with Large Requests
If large requests cause OOM:
- Ensure multi-GPU mode is enabled: `--gpu-mode auto`
- Check memory guard settings in server configuration
- Monitor KV cache usage via `/memory_stats`