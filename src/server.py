#!/usr/bin/env python3
"""
GPT-OSS HuggingFace Server v4.5.4-P0
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
    """Model manager with P0 improvements"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device_map = None
        self.prompt_builder = None
        self.torch_dtype = self._determine_torch_dtype()
        self.model_metadata = {}
        self._initialize()
    
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
            
            # Load model with explicit dtype and memory optimization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "70GiB", 1: "70GiB", 2: "70GiB", 3: "70GiB"}  # Leave some headroom
            )
            
            # Store model metadata for tagging
            self.model_metadata = {
                "model_id": self.config.model_name.split("/")[-1],
                "model_size": self.config.model_size,
                "dtype": str(self.torch_dtype).replace("torch.", ""),
                "gpu_mode": self.config.gpu_mode,
                "prompt_version": self.config.prompt_version.value
            }
            
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
    
    logger.info("🚀 Server v4.5.4-P0 starting...")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    
    # Apply default profile
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
    
    logger.info("🛑 Server v4.5.4-P0 shutting down...")

# Create FastAPI app
app = FastAPI(
    title="GPT-OSS HuggingFace Server P0",
    version="4.5.4-P0",
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
    
    # Calculate health score
    error_rate = global_stats["error_rate"]
    p95_e2e = global_stats["p95_e2e_ms"]
    
    if error_rate > 0.01 or p95_e2e > 20000:
        health_score = 0.5
    elif error_rate > 0.005 or p95_e2e > 10000:
        health_score = 0.8
    else:
        health_score = 1.0
    
    return {
        "status": "healthy" if health_score > 0.5 else "degraded",
        "health_score": health_score,
        "version": "4.5.4-P0",
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
    global_stats = stats.get_stats()
    
    # Add model metadata
    if model_manager:
        global_stats["model_info"] = model_manager.model_metadata
        
        # Update prompt metrics
        if model_manager.prompt_builder:
            stats.update_prompt_metrics(model_manager.prompt_builder.get_metrics())
    
    return global_stats

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest, req: Request):
    """Chat completion endpoint with P0 improvements"""
    request_id = str(uuid.uuid4())
    
    # Check concurrent request limit
    if stats.active_requests >= 10:  # Max 10 concurrent requests
        logger.warning(f"Request {request_id} rejected: too many concurrent requests ({stats.active_requests})")
        raise HTTPException(status_code=503, detail="Server overloaded. Please retry later.")
    
    # Apply profile if specified
    if request.profile:
        try:
            profile = Profile(request.profile)
            config.apply_profile(profile)
            # Reinitialize model if needed
            global model_manager
            model_manager = ModelManager(config)
        except:
            pass
    
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
        # Convert messages
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
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
    """Prometheus-compatible metrics endpoint"""
    global_stats = stats.get_stats()
    
    metrics_text = []
    
    # Basic metrics
    metrics_text.append(f'# HELP requests_total Total number of requests')
    metrics_text.append(f'# TYPE requests_total counter')
    metrics_text.append(f'requests_total {global_stats["requests_total"]}')
    
    metrics_text.append(f'# HELP requests_active Active requests')
    metrics_text.append(f'# TYPE requests_active gauge')
    metrics_text.append(f'requests_active {global_stats["active_requests"]}')
    
    metrics_text.append(f'# HELP stream_active Active streams')
    metrics_text.append(f'# TYPE stream_active gauge')
    metrics_text.append(f'stream_active {global_stats["stream_active"]}')
    
    metrics_text.append(f'# HELP stream_cancelled_total Total cancelled streams')
    metrics_text.append(f'# TYPE stream_cancelled_total counter')
    metrics_text.append(f'stream_cancelled_total {global_stats["stream_cancelled_total"]}')
    
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
    
    return "\n".join(metrics_text)

# ================== Main ==================

def main():
    parser = argparse.ArgumentParser(description="GPT-OSS HF Server v4.5.4-P0")
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
    
    # Set profile
    if args.profile == "latency_first":
        config.profile = Profile.LATENCY_FIRST
    else:
        config.profile = Profile.QUALITY_FIRST
    
    # Override model size if specified
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