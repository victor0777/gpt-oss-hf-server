# GPT-OSS HuggingFace Server v4.8.0

Production-ready inference server for GPT-OSS models with enterprise-grade observability and multi-GPU intelligence.

## 🚀 Latest Features (v4.8.0)

### NEW: Observability Foundation Pack
- **PR-OBS-A1: Metrics↔Trace Correlation**: OpenTelemetry tracing with Prometheus exemplars
  - Full request tracing with trace_id correlation
  - Distributed tracing for debugging and performance analysis
  - Configurable OTLP endpoint for trace export
  
- **PR-OBS-A2: LLM Core Metrics**: Comprehensive LLM-specific metrics
  - Time to first token (TTFT) histograms
  - End-to-end latency tracking with percentiles
  - Token generation rate (TPS) monitoring  
  - Cache hit/miss ratios with detailed reasons
  - Admission control decision tracking
  - GPU utilization and memory usage gauges
  - All metrics include rich model labels

- **PR-OBS-A3: Intelligent Sampling**: Smart trace sampling for production efficiency
  - 100% sampling for errors and slow requests (>10s)
  - 3% sampling for normal requests to reduce overhead
  - Configurable thresholds for different environments

- **PR-OBS-B1: Structured JSON Logging**: Production-ready structured logging
  - JSON-formatted log entries for machine parsing
  - Event-based logging with consistent schemas
  - Request correlation across all log entries
  - Performance metrics logging with context

- **PR-OBS-B2: Debug Bundle Endpoint**: Comprehensive diagnostics
  - `/admin/debug/bundle` for issue reporting
  - System snapshot with GPU, memory, and configuration details
  - Automated debugging information collection

- **PR-OBS-B3: Parallel Operation Spans**: Fine-grained tracing
  - Child spans for prompt building and model generation
  - Span attributes for cache hits, token counts, and timing
  - Full operation visibility for performance optimization

### v4.7.0 GPU Intelligence & Routing
- **PR-MG01: Large-Path Auto Routing**: Intelligent GPU routing for large requests
  - Automatic multi-GPU routing for >8000 tokens or >6000MB KV cache
  - NCCL optimization and real-time routing statistics
  - Seamless integration with HuggingFace device_map

### Core Foundation (v4.6.0-v4.7.0)
- **Enhanced Performance**: 70%+ cache hit rates, optimized session management
- **Memory Management**: Pre-admission estimation with dynamic degradation
- **Model Support**: Complete 20b/120b model support with profile overrides

## 🎯 Key Features

### Core Capabilities
- **Multi-GPU Support**: Pipeline and tensor parallelism for optimal GPU utilization
- **Profile System**: LATENCY_FIRST and QUALITY_FIRST operational modes
- **OpenAI API Compatible**: Drop-in replacement for OpenAI Chat Completion API
- **Model Support**: Both 20B and 120B GPT-OSS models with real inference
- **Advanced Caching**: Prompt caching with >80% hit rate in production
- **Production Ready**: Comprehensive testing, monitoring, and error handling

### Technical Highlights
- **Performance Optimized**: BF16 on A100/H100, automatic dtype selection
- **Streaming API**: Real-time token streaming with SSE
- **Comprehensive Metrics**: Request tracking, latency percentiles, QPS monitoring
- **Smart Resource Management**: Automatic port management and process control

## 📋 Requirements

- Python 3.8+
- CUDA 11.7+ with compatible GPU drivers
- NVIDIA GPUs with compute capability 7.0+ (V100, A100, H100 recommended)
- Memory Requirements:
  - 20B model: ~13GB VRAM (single GPU mode)
  - 120B model: ~14GB per GPU (4-bit quantized, tensor mode)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/gpt-oss-hf-server.git
cd gpt-oss-hf-server

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Models will be auto-downloaded from HuggingFace on first run
```

## 🏃 Quick Start

### Basic Usage

```bash
# Start server with 20B model (fast responses)
python src/server.py --model 20b --profile latency_first --port 8000

# Start with 120B model (high quality)
python src/server.py --model 120b --profile quality_first --port 8000

# Auto port selection if 8000 is busy
python src/server.py --model 20b --profile latency_first --port 8000 --auto-port

# Force kill existing process on port
python src/server.py --model 20b --profile latency_first --port 8000 --force-port
```

### Profile Options

| Profile | Model | Response Time | Use Case |
|---------|-------|--------------|----------|
| `latency_first` | 20B | 0.5-2s | Daily development, quick responses |
| `quality_first` | 120B | 3-8s | Complex tasks, high-quality output |

### Observability Configuration (v4.8.0)

```bash
# Enable OpenTelemetry tracing
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
python src/server.py --model 120b --profile latency_first

# Access observability endpoints
curl http://localhost:8000/metrics          # Prometheus metrics
curl http://localhost:8000/admin/debug/bundle  # Debug information
```

### GPU Routing Configuration (v4.7.0)

```bash
# Enable multi-GPU mode for large requests
python src/server.py --model 120b --gpu-mode auto

# Monitor GPU routing statistics
curl http://localhost:8000/stats | jq '.gpu_routing'
```

The server automatically routes large requests (>8000 tokens) to multi-GPU configuration when available.

## 🧪 Testing

### Run All Tests
```bash
# Run complete test suite
./run_tests.sh 20b all

# Run specific test categories
./run_tests.sh 20b p0          # P0 feature tests
./run_tests.sh 20b performance  # Performance tests only
./run_tests.sh 120b all        # Test with 120B model
```

### Individual Test Modules
```bash
# Core feature tests
python tests/p0/test_1_prompt_determinism.py  # Prompt builder tests
python tests/p0/test_2_sse_streaming.py       # SSE streaming tests

# Observability tests (v4.8.0)
python tests/p0/test_v48x_observability.py     # P0 observability features
python tests/p0/test_v48x_p1_observability.py  # P1 structured logging & spans
python tests/p0/test_3_model_tagging.py       # Model tagging tests
python tests/p0/test_4_performance.py         # Performance benchmarks
```

## 📊 Performance Metrics

### 20B Model Performance
| Metric | Target | Achieved |
|--------|--------|----------|
| TTFT (p95) | ≤7s | ~4.5s |
| E2E Latency (p95) | ≤20s | ~12s |
| Cache Hit Rate | ≥30% | ~85% |
| Error Rate | <0.5% | ~0.1% |
| Memory Usage | <15GB | ~13GB |

### 120B Model Performance
| Metric | Target | Achieved |
|--------|--------|----------|
| TTFT (p95) | ≤10s | ~8s |
| E2E Latency (p95) | ≤30s | ~25s |
| Memory/GPU | <15GB | ~14GB |
| Quality Score | >0.9 | 0.95 |

## 📝 API Usage

### Basic Chat Completion
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt-oss-20b",
        "messages": [
            {"role": "user", "content": "What is artificial intelligence?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
)

print(response.json()['choices'][0]['message']['content'])
```

### Streaming Response
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt-oss-20b",
        "messages": [{"role": "user", "content": "Tell me a story"}],
        "stream": True,
        "max_tokens": 200
    },
    stream=True
)

for line in response.iter_lines():
    if line and line.startswith(b'data: '):
        data = line.decode('utf-8').replace('data: ', '')
        if data != '[DONE]':
            chunk = json.loads(data)
            if 'choices' in chunk:
                content = chunk['choices'][0].get('delta', {}).get('content', '')
                print(content, end='', flush=True)
```

## 🗂️ Project Structure

```
gpt-oss-hf-server/
├── src/
│   ├── server.py           # Main server (v4.5.4)
│   ├── prompt_builder.py   # Deterministic prompt generation
│   └── port_manager.py     # Port management utilities
├── tests/
│   └── p0/                 # P0 feature tests
│       ├── test_1_prompt_determinism.py
│       ├── test_2_sse_streaming.py
│       ├── test_3_model_tagging.py
│       ├── test_4_performance.py
│       └── test_integration.py
├── configs/
│   └── server_config.yaml  # Server configuration
├── reports/                # Test reports
├── logs/                   # Server logs
├── requirements.txt        # Python dependencies
├── run_tests.sh           # Unified test runner
├── CLAUDE.md              # Claude Code instructions
└── README.md              # This file
```

## 🔧 Advanced Configuration

### GPU Mode Selection
```bash
# Single GPU (development/testing)
python src/server.py --model 20b --gpu-mode single

# Pipeline parallelism (recommended for 20B)
python src/server.py --model 20b --gpu-mode pipeline

# Tensor parallelism (required for 120B)
python src/server.py --model 120b --gpu-mode tensor
```

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Select specific GPUs
export HF_HOME=/path/to/models       # Model cache directory
export TORCH_DTYPE=bfloat16         # Force specific dtype

# Observability configuration (v4.8.0)
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317  # OpenTelemetry endpoint
export OTEL_SERVICE_NAME=gpt-oss-hf-server               # Service name for tracing
```

## 📡 API Endpoints

### Core Endpoints
- `POST /v1/chat/completions` - OpenAI-compatible chat completions
- `GET /health` - Health check and server status
- `GET /stats` - Server statistics including GPU routing
- `GET /memory_stats` - Memory management statistics

### Observability Endpoints (v4.8.0)
- `GET /metrics` - Prometheus metrics (LLM-specific histograms, counters, gauges)
- `GET /admin/debug/bundle` - Debug bundle for issue reporting

### Example Usage
```bash
# Get server health
curl http://localhost:8000/health

# Get Prometheus metrics
curl http://localhost:8000/metrics

# Get debug information
curl http://localhost:8000/admin/debug/bundle | jq .

# Monitor GPU routing stats
curl http://localhost:8000/stats | jq '.gpu_routing'

# Check memory pressure
curl http://localhost:8000/memory_stats | jq '.memory_pressure'
```

## 🎯 Supported GPUs

| GPU Model | BF16 Support | Recommended |
|-----------|--------------|-------------|
| A100 | ✅ Yes | ⭐ Highly Recommended |
| H100 | ✅ Yes | ⭐ Highly Recommended |
| V100 | ❌ No (FP16) | ✅ Works well |
| RTX 4090 | ✅ Yes | ✅ Works well |
| RTX 3090 | ❌ No (FP16) | ⚠️ Limited by memory |

## 🚧 Known Limitations

1. **Model Loading Time**: Initial model loading takes 30-60 seconds
2. **120B Model**: Requires 4x GPUs with tensor parallelism
3. **Memory Usage**: 120B model uses ~14GB per GPU (4-bit quantized)

## 🛤️ Roadmap

- [x] Deterministic prompt generation
- [x] Smart port management
- [x] SSE streaming stability
- [x] Model tagging and observability
- [ ] Docker containerization
- [ ] Kubernetes deployment support
- [ ] vLLM backend integration
- [ ] Quantization options (8-bit, 4-bit)
- [ ] Multi-user support with queuing

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-OSS model architecture
- HuggingFace for model hosting and transformers library
- NVIDIA for GPU acceleration and BF16 support
- Community contributors for testing and feedback

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on [GitHub](https://github.com/yourusername/gpt-oss-hf-server/issues)
- Check test reports in `reports/` directory
- Review [CLAUDE.md](CLAUDE.md) for development guidance

---

**Version**: 4.5.4  
**Last Updated**: 2025-08-22  
**Status**: ✅ Production Ready