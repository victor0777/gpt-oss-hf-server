# Changelog

All notable changes to the GPT-OSS HuggingFace Server project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [4.4.0] - 2025-08-21

### Added
- **CLI Parameter Support**: Added command-line parameters for flexible GPU mode selection
  - `--gpu-mode` flag with options: `single`, `pipeline`, `tensor`, `auto`
  - Support for both 20b and 120b model sizes via CLI
  - Auto-detection of optimal GPU mode based on model size
- **120b Model Support**: Full support for GPT-OSS 120b model with tensor parallelism
- **Continuous Batching**: Implemented continuous batch processing with multiple processors
  - 8 concurrent batch processors for improved throughput
  - Dynamic batch size adjustment based on model size
- **Enhanced Monitoring**: Improved GPU utilization and performance metrics tracking

### Changed
- **Batch Configuration**: Optimized batch sizes for different model sizes
  - 20b model: 64 max batch size, 262KB prefill tokens
  - 120b model: 8 max batch size, 32KB prefill tokens
- **Concurrency**: Increased max concurrent requests to 128
- **Memory Management**: Better memory allocation strategies for multi-GPU setups

### Fixed
- **Multi-GPU Distribution**: Fixed issue where small models only loaded on single GPU
- **Tensor Device Mismatch**: Resolved CUDA device placement errors in batch processing
- **Input Tensor Handling**: Fixed attribute errors in generation pipeline

### Performance
- **20b Model (Pipeline Mode)**:
  - QPS: ~1.5 (improved from 1.10)
  - P95 Latency: ~7,000ms (improved from 9,400ms)
  - GPU Utilization: All 4 GPUs active
  - Error Rate: 0%
- **120b Model (Tensor Mode)**:
  - QPS: 0.14 (expected for large model)
  - P95 Latency: 37,791ms
  - Memory Usage: ~60GB distributed across 4 GPUs
  - Error Rate: 0%

## [4.3.0] - 2025-08-21

### Added
- **Pipeline Parallelism**: Model replication across multiple GPUs for small models
- **Multi-GPU Model Manager**: New class for managing GPU strategies

### Fixed
- **Multi-GPU Issue**: Resolved problem where only GPU0 was being utilized
  - Implemented forced distribution for small models
  - Added round-robin load balancing

### Performance
- Successfully activated all 4 GPUs
- QPS: 1.12
- P95 Latency: 6,914ms

## [4.2.0] - 2025-08-21

### Added
- **Adaptive Batching**: SLA-based tuning for batch windows
- **Multi-GPU Support**: Initial Accelerate integration
- **Health Score System**: Enhanced responsiveness
- **Thread-safe Statistics**: Deque buffers for metrics
- **Prefix Caching**: Improved cache hit rate

### Changed
- Increased batch sizes and concurrency limits
- Optimized prefill and decode windows

### Known Issues
- Multi-GPU only utilizing single GPU (fixed in v4.3)

## [4.1.0] - 2025-08-20

### Added
- **OpenTelemetry Integration**: Comprehensive observability
- **Kernel Manager**: Dynamic kernel selection (FlashInfer, Triton, PyTorch)
- **Advanced Batching**: Separate prefill and decode phases
- **Cache Management**: SHA1-based prefix caching

### Performance
- QPS: ~1.0
- P95 Latency: ~9,000ms

## [4.0.0] - 2025-08-19

### Added
- **FastAPI Framework**: RESTful API with OpenAI compatibility
- **Async Processing**: Asynchronous request handling
- **Basic Batching**: Initial batch processing implementation
- **Health Monitoring**: Basic health check endpoints

## [3.0.0] - 2025-08-18

### Added
- Initial HuggingFace Transformers integration
- Basic model loading and inference
- Simple HTTP server

## Version Management Strategy

### Versioning Scheme
- **Major (X.0.0)**: Breaking API changes or architectural overhauls
- **Minor (x.X.0)**: New features, backwards compatible
- **Patch (x.x.X)**: Bug fixes and performance improvements

### Branch Strategy
- `main`: Stable production-ready code
- `develop`: Active development branch
- `feature/*`: Feature branches
- `hotfix/*`: Emergency fixes

### Archive Strategy
- Keep only current major version in `src/`
- Archive previous versions in `archive/` directory (optional)
- Maintain comprehensive changelog for historical reference