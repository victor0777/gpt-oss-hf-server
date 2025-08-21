# GPT-OSS HuggingFace Server v4.5.3

Production-ready inference server for GPT-OSS models with BF16 support, NumPy 2.x compatibility, and personal use optimization.

## 🚀 Latest Features (v4.5.3)

### Major Improvements
- **BF16/FP16 Auto-Detection**: Automatically selects optimal dtype based on GPU capabilities
- **NumPy 2.x Compatibility**: Works with latest NumPy without downgrade requirements
- **Simplified Architecture**: Removed complex engine system for personal use
- **Real Model Inference**: Actual text generation with both 20B and 120B models
- **CUDA Compatibility Fixed**: Resolved FP8 issues on A100 GPUs

## 🎯 Key Features

### Core Capabilities
- **Multi-GPU Support**: Pipeline and tensor parallelism for optimal GPU utilization
- **Profile System**: Three operational modes (LATENCY_FIRST, QUALITY_FIRST, BALANCED)
- **OpenAI API Compatible**: Drop-in replacement for OpenAI Chat Completion API
- **Model Support**: Both 20B and 120B GPT-OSS models with real inference
- **Performance Optimized**: BF16 on A100/H100, automatic dtype selection
- **Production Ready**: Comprehensive testing, monitoring, and error handling

### Technical Highlights
- **NumPy 2.x Support**: sklearn bypass for compatibility
- **BF16 Support**: Optimal performance on modern GPUs (A100, H100)
- **Streaming API**: Real-time token streaming
- **Comprehensive Metrics**: Request tracking, latency percentiles, QPS monitoring

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
git clone https://github.com/victor0777/gpt-oss-hf-server.git
cd gpt-oss-hf-server

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Models will be auto-downloaded from HuggingFace on first run
```

## 🏃 Quick Start

### Running v4.5.3

```bash
# Fast responses with 20B model (BF16 auto-enabled on A100)
python src/server_v453.py --model 20b --profile latency_first --port 8000

# High quality with 120B model
python src/server_v453.py --model 120b --profile quality_first --port 8000

# Balanced mode
python src/server_v453.py --model 20b --profile balanced --port 8000
```

### Profile Options

| Profile | Model | Response Time | Use Case |
|---------|-------|--------------|----------|
| `latency_first` | 20B | 0.5-2s | Daily development, quick responses |
| `quality_first` | 120B | 3-8s | Complex tasks, high-quality output |
| `balanced` | 20B | 1-4s | Mixed workloads |

## 📊 Performance (v4.5.3)

### 20B Model Performance
| Metric | Value | Notes |
|--------|-------|-------|
| Response Time | ~0.89s | With BF16 on A100 |
| Model Loading | 7.5s | Fast startup |
| Memory Usage | ~13GB | Single GPU |
| Inference | ✅ Real | Actual text generation |

### 120B Model Performance
| Metric | Value | Notes |
|--------|-------|-------|
| Response Time | 3-8s | Tensor parallelism |
| Model Loading | 33s | 15 shards across 4 GPUs |
| Memory/GPU | ~14GB | 4-bit quantized |
| Quality | Excellent | High-quality responses |

## 🔧 Advanced Configuration

### GPU Mode Selection
```bash
# Single GPU (development/testing)
python src/server_v453.py --model 20b --gpu-mode single

# Pipeline parallelism (recommended for 20B)
python src/server_v453.py --model 20b --gpu-mode pipeline

# Tensor parallelism (required for 120B)
python src/server_v453.py --model 120b --gpu-mode tensor
```

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Select specific GPUs
export HF_HOME=/path/to/models       # Model cache directory
export TORCH_DTYPE=bfloat16         # Force specific dtype (auto-detect by default)
```

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

## 🧪 Testing

### Quick Test
```bash
# Run automated test suite
./test_v452.sh --all

# Test specific components
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

### Performance Testing
```bash
# Benchmark requests
for i in {1..10}; do
  time curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "gpt-oss-20b", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 50}'
done
```

## 📁 Project Structure

```
gpt-oss-hf-server/
├── src/
│   ├── server_v453.py     # Latest production server with BF16 support
│   ├── server_v452.py     # Previous version
│   └── server_v451*.py    # Earlier versions
├── scripts/
│   ├── start_vllm.sh      # vLLM startup script
│   └── release_gate*.py   # Release validation scripts
├── tests/
│   └── test_v45.py        # Test suite
├── reports/
│   └── v451/              # Test reports and documentation
├── configs/
│   └── server_config.yaml # Server configuration
├── requirements.txt       # Python dependencies
├── test_v452.sh          # Automated test script
├── ARCHITECTURE.md       # Architecture decisions
├── TEST_RESULTS_V453_FINAL.md # Latest test results
├── CHANGELOG.md          # Version history
└── README.md             # This file
```

## 🔄 Version History

### v4.5.3 (Current - Production Ready)
- Added BF16/FP16 auto-detection for GPU compatibility
- Fixed CUDA issues with A100 GPUs
- Real model inference working
- Maintained NumPy 2.x compatibility

### v4.5.2
- NumPy 2.x compatibility via sklearn bypass
- Removed engine system for simplicity
- Added comprehensive test suite

### v4.5.1
- Added profile system
- Enhanced metrics and monitoring
- Personal mode optimization

See [CHANGELOG.md](CHANGELOG.md) for complete version history.

## 🎯 Supported GPUs

| GPU Model | BF16 Support | Recommended |
|-----------|--------------|-------------|
| A100 | ✅ Yes | ⭐ Highly Recommended |
| H100 | ✅ Yes | ⭐ Highly Recommended |
| V100 | ❌ No (FP16) | ✅ Works well |
| RTX 4090 | ✅ Yes | ✅ Works well |
| RTX 3090 | ❌ No (FP16) | ⚠️ Limited by memory |

## 🚧 Known Limitations

1. **Streaming Stability**: Streaming responses need minor improvements
2. **Prompt Format**: 120B model may need prompt format optimization
3. **Memory Usage**: 120B model requires 4x GPUs with tensor parallelism

## 🛤️ Roadmap

- [x] BF16 support for A100/H100
- [x] NumPy 2.x compatibility
- [x] Remove complex engine system
- [ ] Improved streaming implementation
- [ ] Docker containerization
- [ ] Kubernetes deployment support
- [ ] vLLM backend option
- [ ] Quantization options (8-bit, 4-bit)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### Development Setup
```bash
# Clone for development
git clone https://github.com/victor0777/gpt-oss-hf-server.git
cd gpt-oss-hf-server

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in development mode
pip install -e .
pip install -r requirements.txt
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-OSS model architecture
- HuggingFace for model hosting and transformers library
- NVIDIA for GPU acceleration and BF16 support
- Community contributors for testing and feedback

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on [GitHub](https://github.com/victor0777/gpt-oss-hf-server/issues)
- Review test results in [TEST_RESULTS_V453_FINAL.md](TEST_RESULTS_V453_FINAL.md)
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for design decisions

---

**Version**: 4.5.3  
**Last Updated**: 2025-08-21  
**Status**: ✅ Production Ready - Fully Tested with BF16 Support