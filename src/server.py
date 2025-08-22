#!/usr/bin/env python3
"""
GPT-OSS HuggingFace Server v4.7.0
- P0 improvements: PromptBuilder, SSE stabilization, model tagging
- BF16 tensor type for A100 compatibility
- NumPy 2.x compatible (sklearn bypass)
- Personal use optimized profiles
"""

import os
import sys
import logging
import warnings
import asyncio
import json
import time
import argparse
import torch
import numpy as np
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass
import uuid
from enum import Enum
from threading import Lock
import hashlib

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info(f"NumPy version: {np.__version__}")

# Monkey patch for sklearn compatibility
import builtins
_original_import = builtins.__import__

def _patched_import(name, *args, **kwargs):
    """Bypass sklearn imports for NumPy 2.x compatibility"""
    if 'sklearn' in name or 'scipy.sparse' in name:
        logger.debug(f"Bypassing import: {name}")
        class DummyModule:
            def __getattr__(self, item):
                return lambda *args, **kwargs: None
        return DummyModule()
    return _original_import(name, *args, **kwargs)

builtins.__import__ = _patched_import

try:
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        if torch.cuda.is_bf16_supported():
            logger.info("✅ BF16 is supported on this GPU")
        else:
            logger.warning("⚠️ BF16 is not supported, using FP16")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Transformers loaded successfully")
except Exception as e:
    logger.error(f"Failed to load transformers: {e}")
    TRANSFORMERS_AVAILABLE = False

builtins.__import__ = _original_import

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# Import our PromptBuilder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prompt_builder import PromptBuilder, PromptConfig, GenerationParams, PromptVersion
from memory_guard import MemoryGuard, MemoryConfig, AdmissionController
from gpu_router import GPURouter, RoutingConfig, RoutingMode

# ================== Configuration ==================

class Profile(Enum):
    """Operational profiles for personal use"""
    LATENCY_FIRST = "latency_first"  # Default: 20B, fast responses
    QUALITY_FIRST = "quality_first"  # 120B, better quality

PROFILE_CONFIGS = {
    Profile.LATENCY_FIRST: {
        "model_size": "20b",
        "gpu_mode": "pipeline",
        "max_batch_size": 8,
        "prefill_window_ms": 6,
        "decode_window_ms": 3,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "description": "Optimized for fast responses (0.5-2s)"
    },
    Profile.QUALITY_FIRST: {
        "model_size": "120b",
        "gpu_mode": "tensor",
        "max_batch_size": 4,
        "prefill_window_ms": 12,
        "decode_window_ms": 8,
        "max_new_tokens": 2048,
        "temperature": 0.8,
        "description": "Optimized for quality (3-8s)"
    }
}

@dataclass
class ServerConfig:
    """Server configuration with P0 improvements"""
    # Model
    model_name: str = "openai/gpt-oss-20b"
    model_size: str = "20b"
    gpu_mode: str = "auto"
    torch_dtype: Optional[torch.dtype] = None
    
    # Profile
    profile: Profile = Profile.LATENCY_FIRST
    
    # Prompt
    prompt_version: PromptVersion = PromptVersion.SYS_V1
    enable_prefix_cache: bool = True
    prefix_cache_ttl: int = 300
    
    # Batching
    max_batch_size: int = 8
    prefill_window_ms: int = 6
    decode_window_ms: int = 3
    prefill_max_batch_tokens: int = 32768
    decode_max_batch_tokens: int = 8192
    
    # Server
    port: int = 8000
    host: str = "0.0.0.0"
    
    def apply_profile(self, profile: Profile):
        """Apply profile configuration"""
        config = PROFILE_CONFIGS[profile]
        self.model_size = config["model_size"]
        self.gpu_mode = config["gpu_mode"]
        self.max_batch_size = config["max_batch_size"]
        self.prefill_window_ms = config["prefill_window_ms"]
        self.decode_window_ms = config["decode_window_ms"]
        
        # Update model name based on size
        if config["model_size"] == "120b":
            self.model_name = "openai/gpt-oss-120b"
        else:
            self.model_name = "openai/gpt-oss-20b"
        
        logger.info(f"Applied profile: {profile.value} - {config['description']}")

# ================== Model Manager ==================

class ModelManager:
    """Model manager with P0 improvements and memory management"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device_map = None
        self.gpu_mode = config.gpu_mode  # Track GPU mode for routing decisions
        self.prompt_builder = None
        self.torch_dtype = self._determine_torch_dtype()
        self.model_metadata = {}
        self.memory_guard = None
        self.admission_controller = None
        self.gpu_router = None  # PR-MG01: GPU Router for large requests
        self._initialize()
    
    def __del__(self):
        """Cleanup resources on deletion"""
        try:
            if self.model is not None:
                del self.model
                self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
    
    def _determine_torch_dtype(self):
        """Determine optimal torch dtype based on GPU"""
        if self.config.torch_dtype:
            return self.config.torch_dtype
        
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            logger.info("Using BF16 for better performance")
            return torch.bfloat16
        else:
            logger.info("Using FP16 (BF16 not supported)")
            return torch.float16
    
    def _initialize(self):
        """Initialize model and tokenizer"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available, using mock mode")
            return
        
        logger.info(f"Loading model: {self.config.model_name}")
        start_time = time.time()
        
        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            if not self.tokenizer.pad_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize PromptBuilder
            prompt_config = PromptConfig(
                prompt_version=self.config.prompt_version,
                enable_cache=self.config.enable_prefix_cache,
                cache_ttl=self.config.prefix_cache_ttl
            )
            self.prompt_builder = PromptBuilder(self.tokenizer, prompt_config)
            
            # Determine device map
            if torch.cuda.device_count() > 1 and self.config.gpu_mode in ["tensor", "auto"]:
                self.device_map = "auto"
                logger.info(f"Using tensor parallelism across {torch.cuda.device_count()} GPUs")
            else:
                self.device_map = {"": 0}
                logger.info("Using single GPU")
            
            # Clear GPU cache before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Load model with explicit dtype and memory optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "75GiB", 1: "75GiB", 2: "75GiB", 3: "75GiB"}  # Adjusted for actual usage
            )
            
            # Store model metadata for tagging
            self.model_metadata = {
                "model_id": self.config.model_name.split("/")[-1],
                "model_size": self.config.model_size,
                "dtype": str(self.torch_dtype).replace("torch.", ""),
                "gpu_mode": self.config.gpu_mode,
                "prompt_version": self.config.prompt_version.value
            }
            
            # Initialize memory management after model loading
            memory_config = MemoryConfig(
                gpu_memory_threshold=0.85,
                mem_safety_reserve_mb=2048,
                session_kv_limit_mb=512,
                max_kv_gb=20.0,
                idle_timeout_seconds=180,  # PR-SESSION02: Reduced from 300 to 180
                large_req_tokens=8000,
                large_req_kv_mb=6000
            )
            
            # Get model config for memory calculations
            if hasattr(self.model, 'config'):
                model_config = {
                    'num_hidden_layers': self.model.config.num_hidden_layers,
                    'num_attention_heads': self.model.config.num_attention_heads,
                    'num_key_value_heads': getattr(self.model.config, 'num_key_value_heads', self.model.config.num_attention_heads),
                    'hidden_size': self.model.config.hidden_size,
                    'torch_dtype': str(self.torch_dtype)
                }
            else:
                # Default values for GPT-OSS models
                model_config = {
                    'num_hidden_layers': 48 if '20b' in self.config.model_size else 80,
                    'num_attention_heads': 64 if '20b' in self.config.model_size else 96,
                    'num_key_value_heads': 64 if '20b' in self.config.model_size else 96,
                    'hidden_size': 6144 if '20b' in self.config.model_size else 12288,
                    'torch_dtype': str(self.torch_dtype)
                }
            
            self.memory_guard = MemoryGuard(memory_config, model_config)
            self.admission_controller = AdmissionController(self.memory_guard)
            logger.info("✅ Memory management initialized")
            
            # Initialize GPU Router for PR-MG01
            routing_config = RoutingConfig(
                large_input_tokens=8000,
                large_kv_mb=6000,
                micro_batches=6
            )
            self.gpu_router = GPURouter(routing_config)
            logger.info("✅ GPU Router initialized for 4-GPU auto-routing")
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded in {load_time:.1f}s")
            logger.info(f"Model metadata: {self.model_metadata}")
            
            # Clear GPU cache after model loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Log GPU memory usage
                for i in range(torch.cuda.device_count()):
                    mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(i) / 1024**3
                    logger.info(f"GPU {i}: Allocated: {mem_allocated:.2f}GB, Reserved: {mem_reserved:.2f}GB")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
            self.tokenizer = None
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
        seed: Optional[int] = None,
        request_id: str = None
    ) -> AsyncIterator[str]:
        """Generate text with P0 improvements"""
        
        if not self.model or not self.tokenizer:
            if stream:
                yield "data: {\"error\": \"Model not loaded\"}\n\n"
            else:
                yield json.dumps({"error": "Model not loaded"})
            return
        
        # Build prompt using PromptBuilder
        params = GenerationParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed
        )
        
        prompt, prompt_metadata = self.prompt_builder.build(
            messages,
            params,
            self.model_metadata["model_id"]
        )
        
        # Log prompt metadata
        logger.info(f"Request {request_id}: Prompt built - {prompt_metadata}")
        
        try:
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt")
            if self.device_map == "auto":
                input_ids = inputs.input_ids
            else:
                input_ids = inputs.input_ids.to("cuda")
            
            # Generation kwargs
            # Handle temperature=0.0 for deterministic generation
            if temperature == 0.0:
                gen_kwargs = {
                    "max_new_tokens": max_tokens,
                    "do_sample": False,  # Greedy decoding for temperature=0
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
            else:
                gen_kwargs = {
                    "max_new_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": True,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
            
            if seed is not None:
                torch.manual_seed(seed)
                if temperature > 0:
                    gen_kwargs["do_sample"] = True
            
            if stream:
                # Streaming generation with proper SSE format
                async for token in self._stream_generate(input_ids, gen_kwargs, request_id):
                    yield token
            else:
                # Non-streaming generation
                with torch.no_grad():
                    outputs = self.model.generate(input_ids, **gen_kwargs)
                
                response = self.tokenizer.decode(
                    outputs[0][len(input_ids[0]):],
                    skip_special_tokens=True
                )
                
                result = {
                    "response": response,
                    "metadata": {
                        **prompt_metadata,
                        **self.model_metadata,
                        "request_id": request_id
                    }
                }
                yield json.dumps(result)
        
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU OOM error for {request_id}: {e}")
            # Clear GPU cache
            torch.cuda.empty_cache()
            error_msg = "GPU out of memory. Request too large or server overloaded."
            if stream:
                yield f"data: {{\"type\": \"error\", \"error\": {json.dumps(error_msg)}}}\n\n"
            else:
                yield json.dumps({"error": error_msg})
        except Exception as e:
            logger.error(f"Generation error for {request_id}: {e}")
            # Clear any GPU cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if stream:
                yield f"data: {{\"type\": \"error\", \"error\": {json.dumps(str(e))}}}\n\n"
            else:
                yield json.dumps({"error": str(e)})
    
    async def _stream_generate(
        self,
        input_ids: torch.Tensor,
        gen_kwargs: dict,
        request_id: str
    ) -> AsyncIterator[str]:
        """Improved streaming generation with SSE format"""
        try:
            # Start streaming
            yield f"data: {{\"type\": \"start\", \"request_id\": \"{request_id}\", \"model\": \"{self.model_metadata['model_id']}\"}}\n\n"
            
            # Generate tokens
            generated_tokens = []
            past_key_values = None
            
            for i in range(gen_kwargs["max_new_tokens"]):
                with torch.no_grad():
                    if past_key_values is None:
                        outputs = self.model(input_ids, use_cache=True)
                    else:
                        outputs = self.model(
                            input_ids[:, -1:],
                            past_key_values=past_key_values,
                            use_cache=True
                        )
                    
                    past_key_values = outputs.past_key_values
                    logits = outputs.logits[:, -1, :]
                    
                    # Apply temperature and sampling
                    if gen_kwargs["temperature"] > 0:
                        logits = logits / gen_kwargs["temperature"]
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, 1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)
                    
                    # Decode token
                    token_text = self.tokenizer.decode(next_token[0], skip_special_tokens=False)
                    generated_tokens.append(next_token)
                    
                    # Send token via SSE
                    yield f"data: {{\"type\": \"token\", \"content\": {json.dumps(token_text)}}}\n\n"
                    
                    # Check for EOS
                    if next_token[0].item() == self.tokenizer.eos_token_id:
                        break
                    
                    # Update input_ids for next iteration
                    input_ids = torch.cat([input_ids, next_token], dim=1)
                    
                    # Allow other tasks to run
                    await asyncio.sleep(0)
            
            # Send completion
            yield f"data: {{\"type\": \"done\", \"request_id\": \"{request_id}\"}}\n\n"
            
        except asyncio.CancelledError:
            # Handle cancellation properly
            logger.info(f"Request {request_id} cancelled")
            yield f"data: {{\"type\": \"cancelled\", \"request_id\": \"{request_id}\"}}\n\n"
            raise
        except Exception as e:
            logger.error(f"Streaming error for {request_id}: {e}")
            yield f"data: {{\"type\": \"error\", \"error\": {json.dumps(str(e))}}}\n\n"

# ================== Stats Tracking ==================

class ServerStats:
    """Enhanced stats tracking with model tagging"""
    
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.cancelled_requests = 0
        self.stream_active = 0
        self.stream_cancelled_total = 0
        self.active_requests = 0
        self.ttft_history = []
        self.e2e_history = []
        self.model_metrics = {}  # Per-model metrics
        self.prompt_builder_metrics = {}
        self.admission_action = None  # PR-MG01: Track routing decisions
        self.routing_stats = {"accept": 0, "route4": 0, "reject": 0}  # PR-MG01
        self.lock = Lock()
    
    def record_request_start(self, model_id: str, model_size: str, prompt_version: str) -> float:
        with self.lock:
            self.total_requests += 1
            self.active_requests += 1
            
            # Initialize model-specific metrics
            key = f"{model_id}_{model_size}_{prompt_version}"
            if key not in self.model_metrics:
                self.model_metrics[key] = {
                    "total": 0,
                    "success": 0,
                    "failed": 0,
                    "cancelled": 0
                }
            self.model_metrics[key]["total"] += 1
            
            return time.time()
    
    def record_request_end(
        self,
        start_time: float,
        success: bool,
        model_id: str,
        model_size: str,
        prompt_version: str,
        ttft: Optional[float] = None,
        cancelled: bool = False
    ):
        with self.lock:
            self.active_requests -= 1
            key = f"{model_id}_{model_size}_{prompt_version}"
            
            if cancelled:
                self.cancelled_requests += 1
                if key in self.model_metrics:
                    self.model_metrics[key]["cancelled"] += 1
            elif success:
                self.successful_requests += 1
                if key in self.model_metrics:
                    self.model_metrics[key]["success"] += 1
            else:
                self.failed_requests += 1
                if key in self.model_metrics:
                    self.model_metrics[key]["failed"] += 1
            
            e2e = (time.time() - start_time) * 1000  # ms
            self.e2e_history.append(e2e)
            if ttft:
                self.ttft_history.append(ttft * 1000)  # ms
            
            # Keep only last 100 samples
            self.ttft_history = self.ttft_history[-100:]
            self.e2e_history = self.e2e_history[-100:]
    
    def record_stream_active(self, delta: int):
        with self.lock:
            self.stream_active += delta
            if delta < 0:  # Stream ended
                self.stream_cancelled_total += 1
    
    def update_prompt_metrics(self, metrics: dict):
        with self.lock:
            self.prompt_builder_metrics = metrics
    
    def get_stats(self) -> dict:
        with self.lock:
            uptime = time.time() - self.start_time
            qps = self.total_requests / uptime if uptime > 0 else 0
            error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0
            
            def percentile(data, p):
                if not data:
                    return 0
                sorted_data = sorted(data)
                idx = int(len(sorted_data) * p / 100)
                return sorted_data[min(idx, len(sorted_data)-1)]
            
            return {
                "requests_total": self.total_requests,
                "requests_success": self.successful_requests,
                "requests_failed": self.failed_requests,
                "requests_cancelled": self.cancelled_requests,
                "active_requests": self.active_requests,
                "stream_active": self.stream_active,
                "stream_cancelled_total": self.stream_cancelled_total,
                "qps": round(qps, 2),
                "error_rate": round(error_rate, 4),
                "p50_ttft_ms": round(percentile(self.ttft_history, 50), 2),
                "p95_ttft_ms": round(percentile(self.ttft_history, 95), 2),
                "p99_ttft_ms": round(percentile(self.ttft_history, 99), 2),
                "p50_e2e_ms": round(percentile(self.e2e_history, 50), 2),
                "p95_e2e_ms": round(percentile(self.e2e_history, 95), 2),
                "p99_e2e_ms": round(percentile(self.e2e_history, 99), 2),
                "uptime_seconds": round(uptime, 1),
                "model_metrics": self.model_metrics,
                "prompt_metrics": self.prompt_builder_metrics
            }

# ================== Request Models ==================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    seed: Optional[int] = None
    profile: Optional[str] = None  # Allow profile override per request

# ================== FastAPI Server ==================

# Global instances
config = ServerConfig()
model_manager = None
stats = ServerStats()
active_streams = {}  # Track active streams for cancellation

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager"""
    global model_manager
    
    logger.info("🚀 Server v4.7.0 starting...")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    
    # Apply profile only if not already applied in main()
    # This prevents overwriting model override from command line
    if not hasattr(config, '_profile_applied'):
        config.apply_profile(config.profile)
    
    # Initialize model manager
    model_manager = ModelManager(config)
    
    # Heartbeat task for SSE
    async def heartbeat():
        while True:
            await asyncio.sleep(5)
            for stream_id in list(active_streams.keys()):
                try:
                    if stream_id in active_streams:
                        # Send heartbeat to keep connection alive
                        pass
                except:
                    pass
    
    asyncio.create_task(heartbeat())
    
    yield
    
    logger.info("🛑 Server v4.7.0 shutting down...")

# Create FastAPI app
app = FastAPI(
    title="GPT-OSS HuggingFace Server P0",
    version="4.7.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================== API Endpoints ==================

@app.get("/health")
async def health():
    """Health check endpoint with model info"""
    global_stats = stats.get_stats()
    
    # Calculate health score (adjusted for realistic thresholds)
    error_rate = global_stats["error_rate"]
    p95_e2e = global_stats["p95_e2e_ms"]
    active_requests = global_stats.get("active_requests", 0)
    
    # More lenient thresholds for personal use
    if error_rate > 0.05 or p95_e2e > 30000 or active_requests > 15:
        health_score = 0.5  # degraded
    elif error_rate > 0.01 or p95_e2e > 20000 or active_requests > 10:
        health_score = 0.8  # warning
    else:
        health_score = 1.0  # healthy
    
    return {
        "status": "healthy" if health_score > 0.5 else "degraded",
        "health_score": health_score,
        "version": "4.7.0",
        "model": model_manager.model_metadata if model_manager else {},
        "dtype": str(model_manager.torch_dtype) if model_manager else "unknown",
        "profile": config.profile.value,
        "stats": {
            "active_requests": global_stats["active_requests"],
            "qps": global_stats["qps"],
            "error_rate": global_stats["error_rate"],
            "p95_ttft_ms": global_stats["p95_ttft_ms"],
            "p95_e2e_ms": global_stats["p95_e2e_ms"]
        }
    }

@app.get("/stats")
async def get_stats():
    """Get detailed statistics with model tagging"""
    # Always update prompt metrics before returning stats
    if model_manager and model_manager.prompt_builder:
        stats.update_prompt_metrics(model_manager.prompt_builder.get_metrics())
    
    global_stats = stats.get_stats()
    
    # Add model metadata
    if model_manager:
        global_stats["model_info"] = model_manager.model_metadata
        
        # Add GPU routing statistics (PR-MG01)
        if model_manager.gpu_router:
            global_stats["gpu_routing"] = model_manager.gpu_router.get_stats()
    
    return global_stats

@app.get("/memory_stats")
async def get_memory_stats():
    """Get memory management statistics (PR-MEM01/02/03)"""
    if not model_manager or not model_manager.memory_guard:
        return {"error": "Memory management not initialized"}
    
    memory_stats = model_manager.memory_guard.get_stats()
    admission_stats = model_manager.admission_controller.get_stats()
    
    # Add current GPU memory status
    if torch.cuda.is_available():
        gpu_stats = []
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            gpu_stats.append({
                "gpu_id": i,
                "free_gb": free / 1024**3,
                "total_gb": total / 1024**3,
                "used_gb": (total - free) / 1024**3,
                "usage_percent": (total - free) / total * 100
            })
        memory_stats["gpu_memory"] = gpu_stats
    
    # Merge admission statistics
    memory_stats.update(admission_stats)
    
    return memory_stats

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest, req: Request):
    """Chat completion endpoint with P0 improvements"""
    request_id = str(uuid.uuid4())
    
    # Check concurrent request limit
    if stats.active_requests >= 10:  # Max 10 concurrent requests
        logger.warning(f"Request {request_id} rejected: too many concurrent requests ({stats.active_requests})")
        raise HTTPException(status_code=503, detail="Server overloaded. Please retry later.")
    
    # Apply profile if specified (removed automatic reinitialization)
    # Profile changes should be done at server startup, not per request
    # if request.profile:
    #     logger.warning(f"Profile change requested but ignored to prevent OOM: {request.profile}")
    
    # Check if model is loaded
    if not model_manager or not model_manager.model:
        logger.error(f"Request {request_id} rejected: Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded. Server is not ready.")
    
    # Get model metadata for tagging
    model_metadata = model_manager.model_metadata if model_manager and hasattr(model_manager, 'model_metadata') else {
        "model_id": "unknown",
        "model_size": "unknown",
        "prompt_version": "unknown"
    }
    
    # PR-MEM01: Admission control with memory estimation
    session_id = req.headers.get("X-Session-ID", request_id)
    
    # Estimate input tokens (rough approximation)
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    input_text = " ".join([m["content"] for m in messages])
    input_tokens = len(input_text.split()) * 2  # Rough estimate: 1 word ≈ 2 tokens
    
    # Check admission with memory guard
    if model_manager.admission_controller:
        should_proceed, admission_result = model_manager.admission_controller.check_admission(
            request_id=request_id,
            input_tokens=input_tokens,
            max_new_tokens=request.max_tokens,
            batch_size=1
        )
        
        logger.info(f"Admission check for {request_id}: {admission_result}")
        
        if not should_proceed:
            stats.record_request_end(start_time, success=False)
            logger.warning(f"Request {request_id} rejected by admission control: {admission_result['reason']}")
            raise HTTPException(
                status_code=503,
                detail=f"Request rejected: {admission_result['reason']}. Please reduce input size or retry later."
            )
        
        # PR-MG01: Check for 4-GPU routing
        gpu_routing_decision = None
        if model_manager.gpu_router:
            # Get current memory pressure
            current_memory_pressure = model_manager.memory_guard._get_gpu_memory_usage()
            
            # Make routing decision
            # Get current GPU mode from model configuration
            current_gpu_mode = "single"  # Default
            if hasattr(model_manager, 'gpu_mode'):
                current_gpu_mode = model_manager.gpu_mode
            elif hasattr(model_manager.model, 'hf_device_map'):
                # Model has device_map, likely in auto mode
                current_gpu_mode = "auto"
            
            gpu_routing_decision = model_manager.gpu_router.should_route_to_multi_gpu(
                input_tokens=input_tokens,  # Already computed above
                max_new_tokens=request.max_tokens,
                current_memory_pressure=current_memory_pressure,
                current_gpu_mode=current_gpu_mode
            )
            
            # Log routing decision in stats
            stats.admission_action = gpu_routing_decision.admission_action
            if gpu_routing_decision.admission_action in stats.routing_stats:
                stats.routing_stats[gpu_routing_decision.admission_action] += 1
            
            if gpu_routing_decision.should_use_multi_gpu:
                logger.info(f"Request {request_id} should use multi-GPU: {gpu_routing_decision.reason}")
                
                # Check current GPU configuration
                if gpu_routing_decision.mode == RoutingMode.AUTO:
                    # Model is already distributed, just log the routing
                    logger.info(f"Model already in multi-GPU mode: {current_gpu_mode}")
                    model_metadata["gpu_mode"] = current_gpu_mode
                    model_metadata["gpu_config"] = gpu_routing_decision.gpu_config
                else:
                    # Model is in single GPU mode but should use multi-GPU
                    logger.warning(gpu_routing_decision.reason)
                    model_metadata["gpu_mode"] = "single"
                    model_metadata["gpu_recommendation"] = "Use --gpu-mode auto or tensor for multi-GPU"
        
        # Apply degraded parameters if needed
        if admission_result.get('degraded'):
            request.max_tokens = admission_result.get('max_new_tokens', request.max_tokens)
            if 'temperature' in admission_result:
                request.temperature = admission_result['temperature']
            if 'top_p' in admission_result:
                request.top_p = admission_result['top_p']
            logger.info(f"Request {request_id} degraded: max_tokens={request.max_tokens}")
        
        # Register session with memory guard
        model_manager.memory_guard.register_session(session_id)
    
    # Record request start
    start_time = stats.record_request_start(
        model_metadata["model_id"],
        model_metadata["model_size"],
        model_metadata["prompt_version"]
    )
    
    ttft = None
    success = False
    cancelled = False
    
    try:
        
        if request.stream:
            # Track active stream
            active_streams[request_id] = True
            stats.record_stream_active(1)
            
            async def generate_stream():
                nonlocal ttft, success, cancelled
                first_token = True
                
                try:
                    async for chunk in model_manager.generate(
                        messages=messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        top_p=request.top_p,
                        stream=True,
                        seed=request.seed,
                        request_id=request_id
                    ):
                        # Check if cancelled
                        if request_id not in active_streams:
                            cancelled = True
                            break
                        
                        if first_token:
                            ttft = time.time() - start_time
                            first_token = False
                        
                        yield chunk
                        
                        # Check disconnection
                        if await req.is_disconnected():
                            cancelled = True
                            break
                    
                    if not cancelled:
                        success = True
                
                except asyncio.CancelledError:
                    cancelled = True
                    logger.info(f"Stream {request_id} cancelled")
                
                finally:
                    # Clean up
                    if request_id in active_streams:
                        del active_streams[request_id]
                        stats.record_stream_active(-1)
                    
                    # Record metrics
                    stats.record_request_end(
                        start_time,
                        success,
                        model_metadata["model_id"],
                        model_metadata["model_size"],
                        model_metadata["prompt_version"],
                        ttft,
                        cancelled
                    )
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream"
            )
        
        else:
            # Non-streaming response
            response_text = ""
            async for chunk in model_manager.generate(
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=False,
                seed=request.seed,
                request_id=request_id
            ):
                response_text = chunk
                ttft = time.time() - start_time
            
            success = True
            
            # Parse response
            try:
                response_data = json.loads(response_text)
                response_content = response_data.get("response", "")
                metadata = response_data.get("metadata", {})
            except:
                response_content = response_text
                metadata = {}
            
            # Format OpenAI-compatible response
            return {
                "id": request_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": metadata.get("tokens_before", 0),
                    "completion_tokens": len(response_content.split()),
                    "total_tokens": metadata.get("tokens_after", 0)
                },
                "metadata": metadata
            }
    
    except Exception as e:
        logger.error(f"Request {request_id} failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        if not request.stream:
            # PR-MEM02: Update session memory tracking
            if model_manager.memory_guard and success:
                try:
                    # Estimate KV cache memory for this request
                    total_tokens = input_tokens + (request.max_tokens if success else 0)
                    kv_cache_bytes = model_manager.memory_guard.estimate_memory(
                        input_tokens=input_tokens,
                        max_new_tokens=request.max_tokens,
                        batch_size=1
                    ).kv_cache_bytes
                    
                    # Update session
                    model_manager.memory_guard.update_session(
                        session_id=session_id,
                        input_tokens=input_tokens,
                        output_tokens=request.max_tokens if success else 0,
                        kv_cache_bytes=kv_cache_bytes
                    )
                except Exception as e:
                    logger.warning(f"Failed to update session memory: {e}")
            
            stats.record_request_end(
                start_time,
                success,
                model_metadata["model_id"],
                model_metadata["model_size"],
                model_metadata["prompt_version"],
                ttft,
                cancelled
            )

@app.post("/cancel/{request_id}")
async def cancel_request(request_id: str):
    """Cancel an active request"""
    if request_id in active_streams:
        del active_streams[request_id]
        return {"status": "cancelled", "request_id": request_id}
    return {"status": "not_found", "request_id": request_id}

@app.get("/metrics")
async def metrics():
    """Prometheus-compatible metrics endpoint - PR-OBS01 enhanced"""
    global_stats = stats.get_stats()
    
    metrics_text = []
    
    # Get model metadata for labels
    model_labels = ""
    if model_manager and model_manager.model_metadata:
        m = model_manager.model_metadata
        model_labels = (f'model_id="{m.get("model_id", "unknown")}",'
                       f'model_size="{m.get("model_size", "unknown")}",'
                       f'dtype="{m.get("dtype", "unknown")}",'
                       f'gpu_mode="{m.get("gpu_mode", "unknown")}",'
                       f'prompt_version="{m.get("prompt_version", "unknown")}"')
    
    # Basic metrics with labels
    metrics_text.append(f'# HELP requests_total Total number of requests')
    metrics_text.append(f'# TYPE requests_total counter')
    metrics_text.append(f'requests_total{{{model_labels}}} {global_stats["requests_total"]}')
    
    metrics_text.append(f'# HELP requests_active Active requests')
    metrics_text.append(f'# TYPE requests_active gauge')
    metrics_text.append(f'requests_active{{{model_labels}}} {global_stats["active_requests"]}')
    
    metrics_text.append(f'# HELP stream_active Active streams')
    metrics_text.append(f'# TYPE stream_active gauge')
    metrics_text.append(f'stream_active{{{model_labels}}} {global_stats["stream_active"]}')
    
    metrics_text.append(f'# HELP stream_cancelled_total Total cancelled streams')
    metrics_text.append(f'# TYPE stream_cancelled_total counter')
    metrics_text.append(f'stream_cancelled_total{{{model_labels}}} {global_stats["stream_cancelled_total"]}')
    
    # Model-specific metrics
    if model_manager:
        model_info = model_manager.model_metadata
        for key, metrics in global_stats.get("model_metrics", {}).items():
            model_id, model_size, prompt_version = key.split("_")
            labels = f'model_id="{model_id}",model_size="{model_size}",prompt_version="{prompt_version}"'
            
            metrics_text.append(f'# HELP model_requests_total Total requests per model')
            metrics_text.append(f'# TYPE model_requests_total counter')
            metrics_text.append(f'model_requests_total{{{labels}}} {metrics["total"]}')
    
    # Latency metrics
    metrics_text.append(f'# HELP ttft_ms Time to first token in milliseconds')
    metrics_text.append(f'# TYPE ttft_ms summary')
    metrics_text.append(f'ttft_ms{{quantile="0.5"}} {global_stats["p50_ttft_ms"]}')
    metrics_text.append(f'ttft_ms{{quantile="0.95"}} {global_stats["p95_ttft_ms"]}')
    metrics_text.append(f'ttft_ms{{quantile="0.99"}} {global_stats["p99_ttft_ms"]}')
    
    metrics_text.append(f'# HELP e2e_ms End-to-end latency in milliseconds')
    metrics_text.append(f'# TYPE e2e_ms summary')
    metrics_text.append(f'e2e_ms{{quantile="0.5"}} {global_stats["p50_e2e_ms"]}')
    metrics_text.append(f'e2e_ms{{quantile="0.95"}} {global_stats["p95_e2e_ms"]}')
    metrics_text.append(f'e2e_ms{{quantile="0.99"}} {global_stats["p99_e2e_ms"]}')
    
    # Prompt cache metrics
    if "prompt_metrics" in global_stats and global_stats["prompt_metrics"]:
        pm = global_stats["prompt_metrics"]
        metrics_text.append(f'# HELP prompt_cache_hits Prompt cache hits')
        metrics_text.append(f'# TYPE prompt_cache_hits counter')
        metrics_text.append(f'prompt_cache_hits {pm.get("cache_hits", 0)}')
        
        metrics_text.append(f'# HELP prompt_cache_hit_rate Prompt cache hit rate')
        metrics_text.append(f'# TYPE prompt_cache_hit_rate gauge')
        metrics_text.append(f'prompt_cache_hit_rate {pm.get("cache_hit_rate", 0)}')
    
    # Memory metrics - PR-SESSION02 & PR-OBS01
    if model_manager and model_manager.memory_guard:
        mem_stats = model_manager.memory_guard.get_stats()
        
        metrics_text.append(f'# HELP sessions_active Active sessions')
        metrics_text.append(f'# TYPE sessions_active gauge')
        metrics_text.append(f'sessions_active{{{model_labels}}} {mem_stats.get("active_sessions", 0)}')
        
        metrics_text.append(f'# HELP sessions_evicted_total Total evicted sessions')
        metrics_text.append(f'# TYPE sessions_evicted_total counter')
        metrics_text.append(f'sessions_evicted_total{{{model_labels}}} {mem_stats.get("sessions_evicted_total", 0)}')
        
        metrics_text.append(f'# HELP kv_in_use_mb KV cache memory in use (MB)')
        metrics_text.append(f'# TYPE kv_in_use_mb gauge')
        metrics_text.append(f'kv_in_use_mb{{{model_labels}}} {mem_stats.get("kv_in_use_mb", 0):.1f}')
        
        # GPU usage (parse percentage string)
        gpu_usage_str = mem_stats.get("gpu_usage", "0%")
        gpu_usage = float(gpu_usage_str.strip('%')) / 100 if '%' in gpu_usage_str else 0
        metrics_text.append(f'# HELP gpu_memory_usage GPU memory usage percentage')
        metrics_text.append(f'# TYPE gpu_memory_usage gauge')
        metrics_text.append(f'gpu_memory_usage{{{model_labels}}} {gpu_usage:.3f}')
    
    return "\n".join(metrics_text)

# ================== Main ==================

def main():
    parser = argparse.ArgumentParser(description="GPT-OSS HF Server v4.7.0")
    parser.add_argument("--model", type=str, choices=["20b", "120b"], default="20b")
    parser.add_argument("--profile", type=str, choices=["latency_first", "quality_first"], 
                       default="latency_first")
    parser.add_argument("--gpu-mode", type=str, choices=["single", "pipeline", "tensor", "auto"],
                       default="auto")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--prompt-version", type=str, default="SYS/v1",
                       help="Prompt version (SYS/v1 or SYS/v2)")
    parser.add_argument("--auto-port", action="store_true",
                       help="Automatically find available port if specified port is in use")
    parser.add_argument("--force-port", action="store_true",
                       help="Force kill existing process on port (use with caution)")
    
    args = parser.parse_args()
    
    # Import port manager
    try:
        from port_manager import interactive_port_resolution, auto_port_resolution
    except ImportError:
        logger.warning("Port manager not available, using default port handling")
        interactive_port_resolution = None
        auto_port_resolution = None
    
    # Handle port resolution
    final_port = args.port
    if interactive_port_resolution and auto_port_resolution:
        if args.auto_port:
            # Automatic port resolution
            try:
                final_port = auto_port_resolution(args.port, auto_kill=args.force_port)
                logger.info(f"Using port: {final_port}")
            except RuntimeError as e:
                logger.error(f"Port resolution failed: {e}")
                sys.exit(1)
        elif args.force_port:
            # Force mode with auto-kill
            try:
                final_port = auto_port_resolution(args.port, auto_kill=True)
                logger.info(f"Using port: {final_port}")
            except RuntimeError as e:
                logger.error(f"Port resolution failed: {e}")
                sys.exit(1)
        else:
            # Interactive mode (default)
            try:
                # Check if running in non-interactive environment
                if not sys.stdin.isatty():
                    # Non-interactive, use auto mode
                    final_port = auto_port_resolution(args.port, auto_kill=False)
                else:
                    # Interactive terminal, ask user
                    final_port = interactive_port_resolution(args.port)
                logger.info(f"Using port: {final_port}")
            except (KeyboardInterrupt, SystemExit):
                logger.info("Port resolution cancelled")
                sys.exit(0)
            except Exception as e:
                logger.error(f"Port resolution failed: {e}")
                sys.exit(1)
    
    # Update config
    config.port = final_port
    config.host = args.host
    config.gpu_mode = args.gpu_mode
    
    # Set profile first
    if args.profile == "latency_first":
        config.profile = Profile.LATENCY_FIRST
    else:
        config.profile = Profile.QUALITY_FIRST
    
    # Apply profile
    config.apply_profile(config.profile)
    config._profile_applied = True  # Mark that profile was applied
    
    # Override model size AFTER profile (so it's not overwritten)
    if args.model == "120b":
        config.model_size = "120b"
        config.model_name = "openai/gpt-oss-120b"
    
    # Set prompt version
    if args.prompt_version == "SYS/v2":
        config.prompt_version = PromptVersion.SYS_V2
    
    # Run server
    uvicorn.run(app, host=config.host, port=config.port)

if __name__ == "__main__":
    main()