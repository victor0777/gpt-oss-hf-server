# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPT-OSS HuggingFace Server v4.8.0 - Enterprise-ready inference server for GPT-OSS models (20B and 120B) with comprehensive observability foundation pack, multi-GPU intelligence, and production-grade telemetry.

## Core Architecture

### Server Architecture v4.8.0
- **server.py**: Main server with comprehensive observability integration
- **observability.py**: ObservabilityManager with OpenTelemetry and Prometheus
- **structured_logging.py**: StructuredLogger for JSON event-based logging
- **prompt_builder.py**: Deterministic prompt generation with caching

### Key Components
1. **ObservabilityManager** (`src/observability.py`): Complete telemetry with metrics, tracing, sampling
2. **StructuredLogger** (`src/structured_logging.py`): JSON logging with event types and correlation
3. **GPU Router**: Intelligent multi-GPU routing with observability integration
4. **Memory Guard**: Session-based KV cache management with telemetry
5. **Profile System**: LATENCY_FIRST, QUALITY_FIRST operational modes
6. **Model Support**: 20B (pipeline) and 120B (tensor) with automatic routing

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

### Testing v4.8.0
```bash
# Run P0 observability tests (v4.8.0 features)
python tests/p0/test_v48x_observability.py        # P0: 7/7 tests (100%)
python tests/p0/test_v48x_p1_observability.py     # P1: 8/9 tests (88%)

# Run complete P0 test suite (11 modules)
for test in tests/p0/test_*.py; do echo "Running $test"; python "$test"; done

# Core functionality tests
python tests/p0/test_1_prompt_determinism.py      # Cache & determinism
python tests/p0/test_2_sse_streaming.py          # SSE streaming
python tests/p0/test_3_model_tagging.py          # Model metadata
python tests/p0/test_4_performance.py            # Performance SLA

# Feature evolution tests
python tests/p0/test_v46x_improvements.py        # v4.6.x features
python tests/p0/test_v47x_gpu_routing.py         # v4.7.x GPU routing

# Integration tests
python tests/p0/test_integration.py              # End-to-end
python tests/p0/test_integration_p0.py           # P0 scenarios
python tests/p0/test_memory_management.py        # Memory guard
```

### Monitoring v4.8.0
```bash
# Health check with v4.8.0 observability
curl http://localhost:8000/health | jq .

# Prometheus metrics (LLM-specific)
curl http://localhost:8000/metrics | grep llm_

# Debug bundle for comprehensive diagnostics
curl http://localhost:8000/admin/debug/bundle | jq .

# Statistics with observability data
curl http://localhost:8000/stats | jq .

# Memory management with pressure monitoring
curl http://localhost:8000/memory_stats | jq .memory_pressure

# GPU routing statistics
curl http://localhost:8000/stats | jq .gpu_routing

# Real-time monitoring
watch -n 1 'curl -s http://localhost:8000/metrics | grep -E "llm_(ttft|e2e|cache)"'
```

## API Endpoints v4.8.0

### Core Endpoints
- `POST /v1/chat/completions`: OpenAI-compatible chat completion API
- `GET /health`: Server health status with observability info
- `GET /stats`: Detailed performance statistics with observability data
- `GET /memory_stats`: Memory management statistics with pressure monitoring

### Observability Endpoints (NEW in v4.8.0)
- `GET /metrics`: Prometheus metrics with LLM-specific histograms and counters
- `GET /admin/debug/bundle`: Comprehensive diagnostics for issue reporting

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

### Observability Configuration
```bash
# Enable OpenTelemetry tracing (optional)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
export OTEL_SERVICE_NAME=gpt-oss-hf-server

# Start server with observability
python src/server.py --model 120b --profile latency_first
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

### Documentation v4.8.0
- `ARCHITECTURE.md`: v4.8.0 enterprise architecture with observability foundation
- `DEPLOYMENT_GUIDE.md`: Production deployment with observability stack
- `RELEASE_NOTES.md`: v4.8.0 comprehensive release documentation
- `GPU_ROUTING.md`: GPU routing with v4.8.0 observability integration
- `CHANGELOG.md`: Complete version history with v4.8.0 features
- `README.md`: Comprehensive documentation with test inventory

## Environment Variables

```bash
# GPU Configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Select specific GPUs
export HF_HOME=/path/to/models       # Model cache directory
export TORCH_DTYPE=bfloat16         # Force specific dtype (auto-detect by default)

# Observability Configuration (v4.8.0)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317  # OpenTelemetry endpoint
export OTEL_SERVICE_NAME=gpt-oss-hf-server               # Service name for tracing
```

## v4.8.0 Features & Limitations

### New Capabilities
1. **Complete Observability**: OpenTelemetry tracing, Prometheus metrics, structured logging
2. **Production Ready**: <1% performance overhead with full telemetry
3. **Debug Capabilities**: Comprehensive debug bundles for issue analysis
4. **Intelligent Sampling**: 100% errors/slow requests, 3% normal requests

### Limitations
1. **120B Model**: Requires 4x GPUs with tensor parallelism
2. **Memory Usage**: 120B model uses ~14GB per GPU + observability overhead
3. **OTLP Dependency**: Tracing requires external OTLP endpoint configuration
4. **Log Volume**: Structured JSON logging may increase disk usage

## Development Workflow v4.8.0

1. Make changes to server files in `src/`
2. Test locally with observability enabled
3. Run P0 observability test suite (v4.8.0 features)
4. Run complete P0 test suite (11 modules)
5. Verify metrics collection and structured logging
6. Check performance impact (<1% overhead target)
7. Monitor observability endpoints for data quality
8. Review debug bundle for comprehensive system info

## v4.8.0 Performance Targets

### SLA Targets (with Observability)
- **TTFT (Time to First Token)**: p95 ≤ 7s (20B), ≤ 10s (120B)
- **E2E Latency**: p95 ≤ 20s (20B), ≤ 30s (120B)
- **Error Rate**: < 0.5% → achieved ~0.1%
- **Cache Hit Rate**: ≥ 30% → achieved ~85%
- **Memory Usage**: < 15GB per GPU + observability overhead
- **Observability Overhead**: < 1% latency impact

### Production Metrics (Achieved)
- **QPS**: 0.3-0.8 (with full observability)
- **Availability**: >99.9% with health monitoring
- **Memory Pressure**: <85% with automated alerts
- **GPU Routing**: 25-33% large requests use multi-GPU

### Test Success Rates
- **P0 Observability**: 7/7 tests (100%)
- **P1 Observability**: 8/9 tests (88%)
- **Core Functionality**: >95% success across all test suites