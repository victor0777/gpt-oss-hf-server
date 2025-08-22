# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPT-OSS HuggingFace Server v4.5.3 - A production-ready inference server for GPT-OSS models (20B and 120B) with BF16 support, NumPy 2.x compatibility, and personal use optimization.

## Core Architecture

### Server Architecture
- **server.py**: Main server with all v4.5.4 features
- **prompt_builder.py**: Deterministic prompt generation with caching
- **port_manager.py**: Smart port management and process control

### Key Components
1. **PromptBuilder** (`src/prompt_builder.py`): Deterministic prompt generation with caching
2. **Profile System**: Three operational modes - LATENCY_FIRST, QUALITY_FIRST, BALANCED
3. **Model Support**: 20B (pipeline parallelism) and 120B (tensor parallelism) models
4. **GPU Management**: Automatic BF16/FP16 selection based on GPU capabilities

## Development Commands

### Running the Server
```bash
# Activate virtual environment
source .venv/bin/activate

# Run with 20B model (fast responses)
python src/server.py --model 20b --profile latency_first --port 8000

# Run with 120B model (high quality)
python src/server.py --model 120b --profile quality_first --port 8000

# With automatic port management
python src/server.py --model 20b --profile latency_first --port 8000 --auto-port
```

### Testing
```bash
# Run complete test suite
./run_tests.sh 20b all

# Run specific test categories
./run_tests.sh 20b p0          # P0 feature tests
./run_tests.sh 20b performance  # Performance tests only
./run_tests.sh 120b all        # Test with 120B model

# Run individual tests
python tests/p0/test_1_prompt_determinism.py  # Prompt builder tests
python tests/p0/test_2_sse_streaming.py       # SSE streaming tests
python tests/p0/test_3_model_tagging.py       # Model tagging tests
python tests/p0/test_4_performance.py         # Performance benchmarks
```

### Monitoring
```bash
# Check server health
curl http://localhost:8000/health | python -m json.tool

# View statistics
curl http://localhost:8000/stats | python -m json.tool

# Monitor GPU usage
nvidia-smi -l 1

# View server logs
tail -f server.log
tail -f server_p0.log
```

## API Endpoints

### Core Endpoints
- `POST /v1/chat/completions`: OpenAI-compatible chat completion API
- `GET /health`: Server health status with model information
- `GET /stats`: Detailed performance statistics
- `GET /metrics`: Prometheus-compatible metrics

### Request Format
```python
{
    "model": "gpt-oss-20b",  # or "gpt-oss-120b"
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false  # or true for SSE streaming
}
```

## Performance Profiles

| Profile | Model | Response Time | Use Case |
|---------|-------|--------------|----------|
| `latency_first` | 20B | 0.5-2s | Daily development, quick responses |
| `quality_first` | 120B | 3-8s | Complex tasks, high-quality output |
| `balanced` | 20B | 1-4s | Mixed workloads |

## NumPy 2.x Compatibility

The server includes a sklearn bypass mechanism for NumPy 2.x compatibility. This is implemented through monkey-patching in the server initialization to avoid dependency conflicts.

## GPU Configuration

### Supported GPUs
- **A100/H100**: BF16 enabled automatically
- **V100**: Falls back to FP16
- **RTX 4090**: BF16 support
- **RTX 3090**: FP16 only

### GPU Modes
- `single`: Single GPU operation
- `pipeline`: Pipeline parallelism (recommended for 20B)
- `tensor`: Tensor parallelism (required for 120B)
- `auto`: Automatic selection based on model and available GPUs

## Important Files

### Configuration
- `configs/server_config.yaml`: Server configuration parameters
- `requirements.txt`: Python dependencies

### Test Infrastructure
- `run_tests.sh`: Unified test runner for all test suites
- `tests/p0/`: P0 feature tests directory
- Individual test modules for specific features

### Documentation
- `ARCHITECTURE.md`: Architecture decisions and evolution
- `TEST_RESULTS_V453_FINAL.md`: Latest test results
- `CHANGELOG.md`: Version history and changes

## Environment Variables

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Select specific GPUs
export HF_HOME=/path/to/models       # Model cache directory
export TORCH_DTYPE=bfloat16         # Force specific dtype (auto-detect by default)
```

## Known Issues & Limitations

1. **Streaming Stability**: SSE streaming implementation needs minor improvements
2. **120B Model**: Requires 4x GPUs with tensor parallelism
3. **Memory Usage**: 120B model uses ~14GB per GPU (4-bit quantized)
4. **Prompt Format**: 120B model may need prompt format optimization

## Development Workflow

1. Make changes to server files in `src/`
2. Test locally with appropriate profile
3. Run P0 test suite to verify functionality
4. Check performance metrics against SLA targets
5. Monitor GPU memory and utilization
6. Review logs for errors or warnings

## Critical Performance Targets

- **TTFT (Time to First Token)**: p95 ≤ 7s
- **E2E Latency**: p95 ≤ 20s
- **Error Rate**: < 0.5%
- **Cache Hit Rate**: ≥ 30%
- **Memory Usage**: < 15GB per GPU