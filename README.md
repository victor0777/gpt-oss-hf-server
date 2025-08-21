# GPT-OSS HuggingFace Server

High-performance inference server for GPT-OSS models with multi-GPU support and OpenAI API compatibility.

## 🚀 Features

- **Multi-GPU Support**: Pipeline and tensor parallelism for optimal GPU utilization
- **Flexible Configuration**: CLI parameters for GPU mode selection
- **OpenAI API Compatible**: Drop-in replacement for OpenAI Chat Completion API
- **Model Support**: Optimized for both 20b and 120b GPT-OSS models
- **Performance Optimized**: Continuous batching, KV caching, and dynamic batch sizing
- **Production Ready**: Comprehensive testing, monitoring, and error handling

## 📋 Requirements

- Python 3.8+
- CUDA 11.7+ with compatible GPU drivers
- 4x NVIDIA GPUs (A100 or similar recommended)
- Memory Requirements:
  - 20b model: ~12GB GPU memory
  - 120b model: ~60GB GPU memory (distributed)

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/victor0777/gpt-oss-hf-server.git
cd gpt-oss-hf-server

# Install dependencies
pip install -r requirements.txt

# Download models (if not already cached)
# Models will be auto-downloaded from HuggingFace on first run
```

## 🏃 Quick Start

### Basic Usage

```bash
# Start server with 20b model (pipeline parallelism)
./scripts/start_server.sh 20b pipeline 8000

# Start server with 120b model (tensor parallelism)
./scripts/start_server.sh 120b tensor 8000

# Auto-detect optimal GPU mode
./scripts/start_server.sh 20b auto 8000
```

### GPU Mode Options

- `single`: Use single GPU (for testing/development)
- `pipeline`: Distribute model copies across GPUs (recommended for 20b)
- `tensor`: Split model layers across GPUs (required for 120b)
- `auto`: Automatically select based on model size

### Direct Python Usage

```python
python src/server.py 20b --gpu-mode pipeline --port 8000
```

## 📊 Performance Benchmarks

### 20b Model (Pipeline Mode)
| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| QPS | ~1.5 | 2.0 | 75% |
| P95 Latency | ~7,000ms | <10,000ms | ✅ |
| GPU Utilization | 4/4 | 4/4 | ✅ |
| Error Rate | 0% | <1% | ✅ |

### 120b Model (Tensor Mode)
| Metric | Value | Notes |
|--------|-------|-------|
| QPS | 0.14 | Expected for 60GB model |
| P95 Latency | ~38,000ms | Large model latency |
| Memory Usage | ~60GB | Distributed across 4 GPUs |
| Error Rate | 0% | Stable |

## 🔧 Configuration

### Server Configuration
Edit `configs/server_config.yaml` to customize:
- Batch processing parameters
- Performance settings
- SLA targets
- Monitoring options

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # GPU selection
export HF_HOME=/path/to/models       # Model cache directory
```

## 📝 API Usage

### Chat Completion
```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "gpt-oss-20b",
        "messages": [
            {"role": "user", "content": "Hello, how are you?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
)

print(response.json())
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
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        data = json.loads(line.decode('utf-8').replace('data: ', ''))
        print(data['choices'][0]['delta'].get('content', ''), end='')
```

## 🧪 Testing

```bash
# Quick status check
python tests/test_status.py

# QPS performance test
python tests/test_qps.py --duration 60 --concurrent 16

# 120b model test
python tests/test_120b.py

# Monitor GPU usage
python scripts/gpu_monitor.py
```

## 📁 Project Structure

```
gpt-oss-hf-server/
├── src/
│   └── server.py          # Main server implementation (v4.4)
├── scripts/
│   ├── start_server.sh    # Server startup script
│   └── gpu_monitor.py     # GPU monitoring utility
├── tests/
│   ├── test_qps.py       # QPS performance testing
│   ├── test_120b.py      # 120b model specific tests
│   └── test_status.py    # Quick health check
├── configs/
│   └── server_config.yaml # Server configuration
├── examples/
│   └── client_example.py  # API usage examples
├── requirements.txt       # Python dependencies
├── CHANGELOG.md          # Version history
└── README.md             # This file
```

## 🔄 Version History

### v4.4 (Current)
- Added CLI parameter support for GPU modes
- Fixed tensor parallelism bugs
- Improved error handling
- Performance optimizations

### v4.3
- Implemented pipeline parallelism
- Fixed multi-GPU distribution issues
- Reduced P95 latency by 26%

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

## 🚧 Known Limitations

1. **QPS Below Target**: Current QPS (~1.5) is 75% of target (2.0)
   - Consider vLLM integration for better performance
   - TensorRT optimization planned

2. **Memory Requirements**: High memory usage for 120b model
   - Requires 4x A100 GPUs or equivalent
   - Memory optimization in progress

## 🛤️ Roadmap

- [ ] vLLM integration for improved QPS
- [ ] TensorRT optimization
- [ ] Docker containerization
- [ ] Kubernetes deployment manifests
- [ ] Prometheus metrics export
- [ ] Auto-scaling support
- [ ] Model quantization options
- [ ] Multi-model serving

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Development Setup
```bash
# Clone for development
git clone https://github.com/victor0777/gpt-oss-hf-server.git
cd gpt-oss-hf-server

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-OSS model architecture
- HuggingFace for model hosting and transformers library
- NVIDIA for GPU acceleration support

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on [GitHub](https://github.com/victor0777/gpt-oss-hf-server/issues)
- Check [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) for deployment help
- Review [CHANGELOG.md](CHANGELOG.md) for version-specific information

---

**Version**: 4.4.0  
**Last Updated**: 2025-08-21  
**Status**: Production Ready with Performance Optimization Opportunities
