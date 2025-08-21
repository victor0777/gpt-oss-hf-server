#!/usr/bin/env python3
"""
GPT-OSS HuggingFace Server v4.4
Enhanced with CLI parameters and QPS optimizations

Key improvements from v4.3:
- CLI parameter for GPU parallelism mode
- Support for both 20b and 120b models
- Optimized batching for higher QPS
- Continuous batching implementation
- Better memory management for large models
"""

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig
)

# Multi-GPU support
try:
    from accelerate import (
        init_empty_weights,
        load_checkpoint_and_dispatch,
        infer_auto_device_map,
        dispatch_model
    )
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("⚠️ Accelerate not available. Multi-GPU support disabled.")

# Pipeline parallelism for multiple models
try:
    from torch.nn.parallel import DataParallel
    from transformers import pipeline
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

# OpenTelemetry
try:
    from opentelemetry import trace, metrics
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Kernel imports
try:
    import flashinfer
    FLASHINFER_AVAILABLE = True
except ImportError:
    FLASHINFER_AVAILABLE = False

# ================== Configuration ==================

@dataclass
class DynamicConfig:
    """v4.4: Enhanced configuration with model-specific optimizations"""
    
    def __init__(self, model_size: str = "20b", gpu_mode: str = "auto"):
        self.model_size = model_size
        self.gpu_mode = gpu_mode  # "single", "pipeline", "tensor", "auto"
        
        # Model-specific batch configuration
        if model_size == "120b":
            # Larger model needs smaller batches
            self.batch_max_size = int(os.getenv("BATCH_MAX_SIZE", "8"))
            self.prefill_max_batch_tokens = int(os.getenv("PREFILL_MAX_BATCH_TOKENS", "32768"))
            self.decode_max_batch_tokens = int(os.getenv("DECODE_MAX_BATCH_TOKENS", "8192"))
        else:
            # 20b model can handle larger batches
            self.batch_max_size = int(os.getenv("BATCH_MAX_SIZE", "64"))  # Increased for QPS
            self.prefill_max_batch_tokens = int(os.getenv("PREFILL_MAX_BATCH_TOKENS", "262144"))  # 4x increase
            self.decode_max_batch_tokens = int(os.getenv("DECODE_MAX_BATCH_TOKENS", "65536"))  # 2x increase
        
        # Optimized window tuning for QPS
        self.prefill_window_ms_start = float(os.getenv("PREFILL_WINDOW_MS_START", "5"))  # Reduced
        self.decode_window_ms_start = float(os.getenv("DECODE_WINDOW_MS_START", "2"))  # Reduced
        self.prefill_window_ms = self.prefill_window_ms_start
        self.decode_window_ms = self.decode_window_ms_start
        
        # SLA targets
        self.target_queue_len = int(os.getenv("TARGET_QUEUE_LEN", "16"))  # Increased
        self.slo_ttft_ms = float(os.getenv("SLO_TTFT_MS", "5000"))
        self.slo_e2e_ms = float(os.getenv("SLO_E2E_MS", "15000"))
        
        # Cache configuration
        self.cache_max_entries = int(os.getenv("CACHE_MAX_ENTRIES", "10000"))
        self.cache_ttl_seconds = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
        
        # Continuous batching for QPS improvement
        self.enable_continuous_batching = bool(os.getenv("ENABLE_CONTINUOUS_BATCHING", "1"))
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "128"))  # High concurrency
        
        # Health monitoring
        self.health_check_interval_s = float(os.getenv("HEALTH_CHECK_INTERVAL_S", "10"))
        self.health_window_size = int(os.getenv("HEALTH_WINDOW_SIZE", "100"))
        
        # Multi-GPU settings based on mode
        self.device_map = os.getenv("HF_ACCELERATE_DEVICE_MAP", "auto")
        self.num_model_replicas = 1 if gpu_mode == "single" else torch.cuda.device_count()
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
    def get_optimal_gpu_mode(self, model_size_gb: float) -> str:
        """Determine optimal GPU mode based on model size"""
        if self.gpu_mode != "auto":
            return self.gpu_mode
            
        gpu_count = torch.cuda.device_count()
        if gpu_count == 1:
            return "single"
            
        # Get GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Decision logic
        if model_size_gb < gpu_memory_gb * 0.5:  # Model fits easily on one GPU
            return "pipeline"  # Replicate for throughput
        elif model_size_gb < gpu_memory_gb * 0.8:  # Model fits on one GPU with tight memory
            return "single"  # Use single GPU to avoid overhead
        else:  # Model needs multiple GPUs
            return "tensor"  # Split model across GPUs

# ================== Multi-GPU Model Manager ==================

class MultiGPUModelManager:
    """v4.4: Enhanced manager with multiple GPU strategies"""
    
    def __init__(self, model_name: str, config: DynamicConfig):
        self.model_name = model_name
        self.config = config
        self.models = []
        self.tokenizer = None
        self.device_count = torch.cuda.device_count()
        self.current_gpu = 0
        self.lock = Lock()
        self.gpu_mode = config.gpu_mode
        
    async def initialize(self):
        """Initialize models based on GPU mode"""
        print(f"🚀 Initializing Multi-GPU Model Manager")
        print(f"  Model: {self.model_name}")
        print(f"  GPU Mode: {self.gpu_mode}")
        print(f"  Available GPUs: {self.device_count}")
        
        # Load tokenizer once
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Estimate model size
        if "120b" in self.model_name.lower():
            model_size_gb = 240  # Approximate for 120b in bf16
        else:
            model_size_gb = 40  # Approximate for 20b in bf16
            
        # Auto-select GPU mode if needed
        if self.gpu_mode == "auto":
            self.gpu_mode = self.config.get_optimal_gpu_mode(model_size_gb)
            print(f"  Auto-selected GPU mode: {self.gpu_mode}")
        
        # Initialize based on mode
        if self.gpu_mode == "single":
            await self._init_single_gpu()
        elif self.gpu_mode == "pipeline":
            await self._init_pipeline_mode()
        elif self.gpu_mode == "tensor":
            await self._init_tensor_parallel()
        else:
            raise ValueError(f"Unknown GPU mode: {self.gpu_mode}")
            
    async def _init_single_gpu(self):
        """Initialize single GPU mode"""
        print("Loading model on single GPU...")
        
        device = torch.device("cuda:0")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": 0}
        )
        model.eval()
        
        self.models.append({
            "model": model,
            "device": device,
            "gpu_id": 0,
            "active_batches": 0
        })
        
        print(f"✅ Model loaded on GPU 0")
        
    async def _init_pipeline_mode(self):
        """Initialize pipeline parallelism (model replication)"""
        print("Loading model replicas for pipeline parallelism...")
        
        for gpu_id in range(self.device_count):
            print(f"Loading model replica on GPU {gpu_id}...")
            
            device = torch.device(f"cuda:{gpu_id}")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map={"": gpu_id}
            )
            model.eval()
            
            self.models.append({
                "model": model,
                "device": device,
                "gpu_id": gpu_id,
                "active_batches": 0
            })
            
            print(f"✅ Model replica {gpu_id} loaded on GPU {gpu_id}")
            
    async def _init_tensor_parallel(self):
        """Initialize tensor parallelism (model sharding)"""
        print("Loading model with tensor parallelism...")
        
        if not ACCELERATE_AVAILABLE:
            print("⚠️ Accelerate not available, falling back to single GPU")
            await self._init_single_gpu()
            return
        
        # Calculate memory allocation
        max_memory = {}
        for i in range(self.device_count):
            max_memory[i] = "75GB"
        max_memory["cpu"] = "128GB"
        
        # Load with auto device map
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            max_memory=max_memory
        )
        model.eval()
        
        self.models.append({
            "model": model,
            "device": torch.device("cuda"),
            "gpu_id": -1,  # All GPUs
            "active_batches": 0
        })
        
        print(f"✅ Model loaded with tensor parallelism across {self.device_count} GPUs")
        
    def get_next_model(self):
        """Get next available model instance for load balancing"""
        with self.lock:
            if self.gpu_mode == "pipeline":
                # Round-robin selection for pipeline mode
                model_info = self.models[self.current_gpu]
                self.current_gpu = (self.current_gpu + 1) % len(self.models)
                return model_info
            else:
                # Single model for other modes
                return self.models[0]
                
    def get_gpu_stats(self):
        """Get current GPU utilization stats"""
        stats = []
        for i in range(self.device_count):
            try:
                util = torch.cuda.utilization(i)
                mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
                mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                stats.append({
                    "gpu_id": i,
                    "utilization": util,
                    "memory_used_gb": mem_alloc,
                    "memory_total_gb": mem_total,
                    "memory_percent": (mem_alloc / mem_total) * 100
                })
            except:
                stats.append({
                    "gpu_id": i,
                    "utilization": 0,
                    "memory_used_gb": 0,
                    "memory_total_gb": 0,
                    "memory_percent": 0
                })
        return stats

# ================== Continuous Batch Processor ==================

class ContinuousBatchProcessor:
    """v4.4: Optimized batch processor for high QPS"""
    
    def __init__(self, model_manager: MultiGPUModelManager, config: DynamicConfig):
        self.model_manager = model_manager
        self.config = config
        self.request_queue = asyncio.Queue(maxsize=config.max_concurrent_requests)
        self.response_futures = {}
        self.active_batches = {}
        self.stats_lock = Lock()
        
        # Statistics
        self.stats = {
            "total_requests": 0,
            "active_requests": 0,
            "completed_requests": 0,
            "total_tokens_generated": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "gpu_switches": 0
        }
        
        # Performance metrics
        self.ttft_history = deque(maxlen=1000)
        self.e2e_history = deque(maxlen=1000)
        self.throughput_history = deque(maxlen=100)
        
        # Continuous batching state
        self.batch_processors = []
        
    async def start_processors(self):
        """Start multiple batch processors for continuous batching"""
        num_processors = min(self.config.num_model_replicas * 2, 8)  # 2x models for overlap
        
        for i in range(num_processors):
            processor = asyncio.create_task(self._continuous_batch_processor(i))
            self.batch_processors.append(processor)
            
        print(f"✅ Started {num_processors} continuous batch processors")
        
    async def _continuous_batch_processor(self, processor_id: int):
        """Continuous batch processing loop"""
        while True:
            try:
                # Collect batch
                batch = []
                wait_start = time.time()
                
                # Dynamic batch collection
                while len(batch) < self.config.batch_max_size:
                    timeout = max(0.001, (self.config.prefill_window_ms / 1000) - (time.time() - wait_start))
                    
                    try:
                        request = await asyncio.wait_for(
                            self.request_queue.get(),
                            timeout=timeout
                        )
                        batch.append(request)
                        
                        # Process immediately if we have enough
                        if len(batch) >= self.config.batch_max_size // 2:
                            break
                    except asyncio.TimeoutError:
                        if batch:  # Process what we have
                            break
                        continue
                
                if batch:
                    await self._process_batch(batch, processor_id)
                    
            except Exception as e:
                logging.error(f"Processor {processor_id} error: {e}")
                await asyncio.sleep(0.1)
                
    async def _process_batch(self, batch: List[Dict], processor_id: int):
        """Process a batch of requests"""
        model_info = self.model_manager.get_next_model()
        model = model_info["model"]
        device = model_info["device"]
        
        with self.stats_lock:
            self.stats["gpu_switches"] += 1
        
        # Tokenize batch
        prompts = [req["prompt"] for req in batch]
        tokenizer = self.model_manager.tokenizer
        
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        # Move to device
        if model_info["gpu_id"] >= 0:
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
        else:
            input_ids = inputs.input_ids
            attention_mask = inputs.attention_mask
        
        # Generate with optimizations
        with torch.no_grad():
            # Use aggressive generation parameters for speed
            generation_config = GenerationConfig(
                max_new_tokens=batch[0]["max_tokens"],
                temperature=batch[0]["temperature"],
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True  # Enable KV cache
            )
            
            start_gen = time.time()
            outputs = model.generate(
                input_ids,  # Use the moved tensors
                attention_mask=attention_mask,
                generation_config=generation_config
            )
            gen_time = time.time() - start_gen
            
        # Decode and return results
        for i, req in enumerate(batch):
            output_ids = outputs[i]
            generated_text = tokenizer.decode(
                output_ids[input_ids.shape[1]:],
                skip_special_tokens=True
            )
            
            # Calculate metrics
            ttft = (time.time() - req["start_time"]) * 1000
            self.ttft_history.append(ttft)
            
            # Complete request
            future = self.response_futures.pop(req["id"], None)
            if future and not future.done():
                e2e = (time.time() - req["start_time"]) * 1000
                self.e2e_history.append(e2e)
                
                with self.stats_lock:
                    self.stats["completed_requests"] += 1
                    self.stats["active_requests"] -= 1
                    self.stats["total_tokens_generated"] += req["max_tokens"]
                    
                future.set_result({
                    "text": generated_text,
                    "tokens": req["max_tokens"],
                    "ttft_ms": ttft,
                    "e2e_ms": e2e,
                    "batch_size": len(batch),
                    "gpu_id": model_info["gpu_id"]
                })
                
    async def process_request(self, prompt: str, max_tokens: int, temperature: float = 0.7):
        """Process generation request"""
        request_id = str(uuid.uuid4())
        
        with self.stats_lock:
            self.stats["total_requests"] += 1
            self.stats["active_requests"] += 1
            
        # Create future for response
        future = asyncio.Future()
        self.response_futures[request_id] = future
        
        # Queue request
        request = {
            "id": request_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "start_time": time.time()
        }
        
        try:
            await asyncio.wait_for(
                self.request_queue.put(request),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            with self.stats_lock:
                self.stats["active_requests"] -= 1
            raise HTTPException(status_code=503, detail="Server overloaded")
        
        # Wait for completion
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            with self.stats_lock:
                self.stats["active_requests"] -= 1
            raise HTTPException(status_code=504, detail="Request timeout")

# ================== Global instances ==================
model_manager: Optional[MultiGPUModelManager] = None
batch_processor: Optional[ContinuousBatchProcessor] = None
config: Optional[DynamicConfig] = None

# ================== FastAPI app ==================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager"""
    global model_manager, batch_processor, config
    
    print("\n" + "="*50)
    print("🚀 GPT-OSS HuggingFace Server v4.4 Starting...")
    print("="*50)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="GPT-OSS Server v4.4")
    parser.add_argument("model_size", choices=["20b", "120b"], 
                       help="Model size to load (20b or 120b)")
    parser.add_argument("--gpu-mode", choices=["single", "pipeline", "tensor", "auto"],
                       default="auto", help="GPU parallelism mode")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    # For FastAPI lifespan, get args from environment
    model_size = os.getenv("MODEL_SIZE", "20b")
    gpu_mode = os.getenv("GPU_MODE", "auto")
    
    # Initialize configuration
    config = DynamicConfig(model_size=model_size, gpu_mode=gpu_mode)
    
    # Select model based on size
    if model_size == "120b":
        model_name = os.getenv("MODEL_NAME", "openai/gpt-oss-120b")
    else:
        model_name = os.getenv("MODEL_NAME", "openai/gpt-oss-20b")
    
    # Initialize model manager
    model_manager = MultiGPUModelManager(model_name, config)
    await model_manager.initialize()
    
    # Initialize batch processor
    batch_processor = ContinuousBatchProcessor(model_manager, config)
    await batch_processor.start_processors()
    
    # Show configuration
    gpu_stats = model_manager.get_gpu_stats()
    print(f"""
✅ Server initialized successfully!
📊 Configuration:
  - Model: {model_name}
  - Model Size: {model_size}
  - GPU Mode: {model_manager.gpu_mode}
  - GPUs Available: {torch.cuda.device_count()}
  - Batch Size: {config.batch_max_size}
  - Max Concurrent: {config.max_concurrent_requests}
  - Continuous Batching: {config.enable_continuous_batching}
    """)
    
    # Show GPU status
    print("🎮 GPU Status:")
    for stat in gpu_stats:
        print(f"  GPU {stat['gpu_id']}: {stat['memory_used_gb']:.1f}/{stat['memory_total_gb']:.1f}GB ({stat['memory_percent']:.1f}%)")
    
    yield
    
    print("\n🛑 Shutting down server...")

app = FastAPI(title="GPT-OSS v4.4", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== API Endpoints ==================

class ChatCompletionRequest(BaseModel):
    model: str = "gpt-oss"
    messages: List[Dict[str, str]]
    max_tokens: int = Field(default=100, ge=1, le=2048)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    stream: bool = False

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]

@app.get("/health")
async def health():
    """Health check endpoint"""
    if model_manager is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    gpu_stats = model_manager.get_gpu_stats()
    avg_util = sum(s["utilization"] for s in gpu_stats) / len(gpu_stats)
    
    # Calculate health score
    health_score = 100.0
    if avg_util > 90:
        health_score -= 20
    if batch_processor.stats["active_requests"] > config.max_concurrent_requests * 0.8:
        health_score -= 20
        
    return {
        "status": "healthy" if health_score > 70 else "degraded",
        "score": health_score,
        "timestamp": datetime.now().isoformat(),
        "version": "4.4.0",
        "model_size": config.model_size,
        "gpu_mode": model_manager.gpu_mode,
        "gpu_count": len(gpu_stats),
        "avg_gpu_utilization": avg_util
    }

@app.get("/stats")
async def stats():
    """Detailed statistics endpoint"""
    if batch_processor is None:
        return {"error": "Service not initialized"}
        
    # Calculate percentiles
    ttft_p95 = np.percentile(batch_processor.ttft_history, 95) if batch_processor.ttft_history else 0
    e2e_p95 = np.percentile(batch_processor.e2e_history, 95) if batch_processor.e2e_history else 0
    
    # Get GPU stats
    gpu_stats = model_manager.get_gpu_stats()
    
    # Calculate QPS
    if batch_processor.stats["completed_requests"] > 0:
        # Get time since first request
        elapsed_time = max(1, time.time() - app.state.start_time) if hasattr(app.state, 'start_time') else 60
        current_qps = batch_processor.stats["completed_requests"] / elapsed_time
    else:
        current_qps = 0
    
    return {
        "requests": batch_processor.stats,
        "performance": {
            "ttft_p95_ms": ttft_p95,
            "e2e_p95_ms": e2e_p95,
            "current_qps": current_qps,
            "avg_tokens_per_request": batch_processor.stats["total_tokens_generated"] / max(1, batch_processor.stats["completed_requests"])
        },
        "gpu_stats": gpu_stats,
        "config": {
            "model_size": config.model_size,
            "gpu_mode": model_manager.gpu_mode,
            "batch_max_size": config.batch_max_size,
            "max_concurrent_requests": config.max_concurrent_requests
        }
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completion endpoint"""
    if batch_processor is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
        
    # Track start time for QPS calculation
    if not hasattr(app.state, 'start_time'):
        app.state.start_time = time.time()
        
    # Convert messages to prompt
    prompt = ""
    for msg in request.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        prompt += f"{role}: {content}\n"
    prompt += "assistant: "
    
    # Process request
    try:
        result = await batch_processor.process_request(
            prompt=prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        return ChatCompletionResponse(
            id=str(uuid.uuid4()),
            created=int(time.time()),
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result["text"]
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": result["tokens"],
                "total_tokens": len(prompt.split()) + result["tokens"]
            }
        )
    except Exception as e:
        logging.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/models")
async def list_models():
    """List available models"""
    model_id = f"gpt-oss-{config.model_size}" if config else "gpt-oss"
    return {
        "object": "list",
        "data": [{
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "openai"
        }]
    }

# ================== Main entry point ==================

def main():
    """Main entry point with CLI argument parsing"""
    parser = argparse.ArgumentParser(description="GPT-OSS Server v4.4")
    parser.add_argument("model_size", choices=["20b", "120b"], 
                       help="Model size to load (20b or 120b)")
    parser.add_argument("--gpu-mode", choices=["single", "pipeline", "tensor", "auto"],
                       default="auto", help="GPU parallelism mode")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    
    args = parser.parse_args()
    
    # Set environment variables for FastAPI lifespan
    os.environ["MODEL_SIZE"] = args.model_size
    os.environ["GPU_MODE"] = args.gpu_mode
    
    # Run server
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=args.port,
        log_level="info"
    )

if __name__ == "__main__":
    main()