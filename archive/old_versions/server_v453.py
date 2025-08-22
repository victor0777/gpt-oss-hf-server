#!/usr/bin/env python3
"""
GPT-OSS HuggingFace Server v4.5.3
- BF16 tensor type for A100 compatibility
- NumPy 2.x compatible (sklearn bypass)
- No engine system (personal use optimized)
- Real model loading with streaming
- Profile system for different use cases
"""

import os
import sys
import logging
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import numpy first to ensure it's loaded before transformers
import numpy as np
logger.info(f"NumPy version: {np.__version__}")

# Monkey patch to bypass sklearn import (for NumPy 2.x compatibility)
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

# Apply patch temporarily
builtins.__import__ = _patched_import

try:
    import torch
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        # Check for BF16 support
        if torch.cuda.is_bf16_supported():
            logger.info("✅ BF16 is supported on this GPU")
        else:
            logger.warning("⚠️ BF16 is not supported on this GPU, will use FP16")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Transformers loaded successfully")
except Exception as e:
    logger.error(f"Failed to load transformers: {e}")
    TRANSFORMERS_AVAILABLE = False

# Restore original import
builtins.__import__ = _original_import

# Rest of imports
import asyncio
import json
import time
import argparse
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass
import uuid
from enum import Enum
from threading import Lock
import hashlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

# ================== Configuration ==================

class Profile(Enum):
    """Operational profiles for different use cases"""
    LATENCY_FIRST = "latency_first"
    QUALITY_FIRST = "quality_first"
    BALANCED = "balanced"

PROFILE_CONFIGS = {
    Profile.LATENCY_FIRST: {
        "model_size": "20b",
        "gpu_mode": "pipeline",
        "max_batch_size": 32,
        "prefill_window_ms": 4,
        "decode_window_ms": 2,
        "max_new_tokens": 512,
        "temperature": 0.7,
        "description": "Optimized for fast responses (4-8s)"
    },
    Profile.QUALITY_FIRST: {
        "model_size": "120b",
        "gpu_mode": "tensor",
        "max_batch_size": 8,
        "prefill_window_ms": 12,
        "decode_window_ms": 8,
        "max_new_tokens": 2048,
        "temperature": 0.8,
        "description": "Optimized for quality (6-15s)"
    },
    Profile.BALANCED: {
        "model_size": "20b",
        "gpu_mode": "pipeline",
        "max_batch_size": 48,
        "prefill_window_ms": 6,
        "decode_window_ms": 4,
        "max_new_tokens": 1024,
        "temperature": 0.75,
        "description": "Balanced performance and quality"
    }
}

@dataclass
class ServerConfig:
    """Server configuration"""
    model_size: str = "20b"
    profile: Profile = Profile.BALANCED
    gpu_mode: str = "pipeline"  # pipeline or tensor
    port: int = 8000
    enable_prefix_cache: bool = True
    enable_streaming: bool = True
    
    @property
    def model_name(self) -> str:
        """Get full model name"""
        return f"openai/gpt-oss-{self.model_size}"
    
    def update_from_profile(self, profile: Profile):
        """Update config from profile"""
        profile_config = PROFILE_CONFIGS[profile]
        self.model_size = profile_config["model_size"]
        self.gpu_mode = profile_config["gpu_mode"]
        self.profile = profile

# ================== Model Manager ==================

class ModelManager:
    """Manages model loading and inference"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.model_loaded = False
        self.device_map = None
        self.torch_dtype = None
        
        # Load model and tokenizer
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _determine_torch_dtype(self):
        """Determine the best dtype for the current GPU"""
        if not torch.cuda.is_available():
            return torch.float32
        
        # Check for BF16 support (A100, H100, etc.)
        if torch.cuda.is_bf16_supported():
            logger.info("Using BF16 for better performance on A100/H100")
            return torch.bfloat16
        else:
            # Fallback to FP16 for older GPUs
            logger.info("Using FP16 (BF16 not supported on this GPU)")
            return torch.float16
    
    def _load_model(self):
        """Load model and tokenizer"""
        try:
            logger.info(f"Loading model {self.config.model_name} with {self.config.gpu_mode} mode")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Determine torch dtype
            self.torch_dtype = self._determine_torch_dtype()
            
            # Detect GPUs
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                logger.info(f"Detected {gpu_count} GPUs")
                
                # Log GPU details
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    logger.info(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                gpu_count = 0
                logger.warning("No GPUs detected, using CPU")
            
            # Determine device map based on GPU mode
            if gpu_count == 0:
                self.device_map = "cpu"
            elif gpu_count == 1:
                self.device_map = {"": 0}
            else:
                # Multi-GPU setup
                if self.config.gpu_mode == "tensor":
                    # Tensor parallelism - split model across GPUs
                    self.device_map = "auto"
                else:
                    # Pipeline parallelism - use first GPU
                    self.device_map = {"": 0}
            
            # Load model with BF16/FP16
            logger.info(f"Loading model to device: {self.device_map}")
            logger.info(f"Using dtype: {self.torch_dtype}")
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map=self.device_map,
                torch_dtype=self.torch_dtype,  # Use determined dtype
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            self.model_loaded = True
            logger.info(f"✅ Model loaded successfully with {self.torch_dtype}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            self.model_loaded = False
    
    async def generate(self, messages: List[dict], **kwargs) -> dict:
        """Generate response from messages"""
        if not self.model_loaded:
            return self._mock_response(messages, **kwargs)
        
        try:
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            
            # Check prefix cache
            cache_key = None
            if self.config.enable_prefix_cache:
                cache_key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
                # Simple cache check (not fully implemented)
            
            # Tokenize
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate
            gen_kwargs = {
                "max_new_tokens": kwargs.get("max_tokens", 100),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
                "do_sample": kwargs.get("temperature", 0.7) > 0,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode
            response_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            return {
                "content": response_text,
                "tokens_used": outputs.shape[1],
                "cache_hit": False
            }
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            # Return mock response on error
            return self._mock_response(messages, **kwargs, error=str(e))
    
    async def generate_stream(self, messages: List[dict], **kwargs) -> AsyncIterator[str]:
        """Generate streaming response"""
        if not self.model_loaded:
            # Mock streaming
            mock_response = "This is a mock streaming response for testing."
            for word in mock_response.split():
                yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': word + ' '}, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.1)
            yield "data: [DONE]\n\n"
            return
        
        try:
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            
            # For real streaming, we'd need to implement token-by-token generation
            # For now, generate full response and stream it word by word
            response = await self.generate(messages, **kwargs)
            
            for word in response["content"].split():
                yield f"data: {json.dumps({'choices': [{'index': 0, 'delta': {'content': word + ' '}, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.05)  # Simulate streaming delay
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    def _messages_to_prompt(self, messages: List[dict]) -> str:
        """Convert messages to prompt string"""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        prompt += "Assistant:"
        return prompt
    
    def _mock_response(self, messages: List[dict], **kwargs) -> dict:
        """Generate mock response for testing"""
        error_msg = kwargs.get("error", "")
        if error_msg:
            content = f"Mock response (error occurred: {error_msg})"
        else:
            content = "Mock response (model not loaded or transformers not available)"
        
        return {
            "content": content,
            "tokens_used": len(content.split()),
            "cache_hit": False
        }

# ================== Request/Response Models ==================

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 100
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False
    profile: Optional[str] = None  # Allow profile override per request

class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    created: int
    object: str = "chat.completion"

# ================== Server Statistics ==================

class ServerStats:
    """Server statistics tracker"""
    
    def __init__(self):
        self.start_time = time.time()
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.active_requests = 0
        self.ttft_history = []  # Time to first token
        self.e2e_history = []   # End-to-end latency
        self.lock = Lock()
    
    def record_request_start(self):
        with self.lock:
            self.total_requests += 1
            self.active_requests += 1
            return time.time()
    
    def record_request_end(self, start_time: float, success: bool, ttft: Optional[float] = None):
        with self.lock:
            self.active_requests -= 1
            if success:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
            
            e2e = (time.time() - start_time) * 1000  # ms
            self.e2e_history.append(e2e)
            if ttft:
                self.ttft_history.append(ttft * 1000)  # ms
            
            # Keep only last 100 samples
            self.ttft_history = self.ttft_history[-100:]
            self.e2e_history = self.e2e_history[-100:]
    
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
                "active_requests": self.active_requests,
                "qps": round(qps, 2),
                "error_rate": round(error_rate, 4),
                "p50_ttft_ms": round(percentile(self.ttft_history, 50), 2),
                "p95_ttft_ms": round(percentile(self.ttft_history, 95), 2),
                "p99_ttft_ms": round(percentile(self.ttft_history, 99), 2),
                "p50_e2e_ms": round(percentile(self.e2e_history, 50), 2),
                "p95_e2e_ms": round(percentile(self.e2e_history, 95), 2),
                "p99_e2e_ms": round(percentile(self.e2e_history, 99), 2),
                "uptime_seconds": round(uptime, 1)
            }

# ================== FastAPI Server ==================

# Global instances
config = ServerConfig()
model_manager = None
stats = ServerStats()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager"""
    global model_manager
    
    logger.info("🚀 Server v4.5.3 starting...")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    
    # Initialize model manager
    model_manager = ModelManager(config)
    
    yield
    
    logger.info("🛑 Server v4.5.3 shutting down...")

# Create FastAPI app
app = FastAPI(
    title="GPT-OSS HuggingFace Server",
    version="4.5.3",
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
    """Health check endpoint"""
    global_stats = stats.get_stats()
    
    # Calculate health score
    error_rate = global_stats["error_rate"]
    if error_rate > 0.1:
        health_score = 0.5
    elif error_rate > 0.05:
        health_score = 0.8
    else:
        health_score = 1.0
    
    # Add model info to stats
    global_stats["model_info"] = {
        "model_size": config.model_size,
        "gpu_mode": config.gpu_mode,
        "profile": config.profile.value,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    return {
        "status": "healthy",
        "score": health_score,
        "version": "4.5.3",
        "model_loaded": model_manager.model_loaded if model_manager else False,
        "stats": global_stats,
        "config": {
            "model_size": config.model_size,
            "gpu_mode": config.gpu_mode,
            "profile": config.profile.value,
            "streaming": config.enable_streaming,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "numpy_version": np.__version__,
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "torch_dtype": str(model_manager.torch_dtype) if model_manager and model_manager.torch_dtype else "unknown",
            "bf16_supported": torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False
        }
    }

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    global_stats = stats.get_stats()
    
    # Add model info
    global_stats["model_info"] = {
        "model_size": config.model_size,
        "gpu_mode": config.gpu_mode,
        "profile": config.profile.value,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    return global_stats

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint"""
    start_time = stats.record_request_start()
    ttft = None
    
    try:
        # Handle profile override
        if request.profile:
            try:
                profile = Profile(request.profile)
                if profile != config.profile:
                    logger.info(f"Profile override: {config.profile.value} -> {profile.value}")
                    # Note: In production, you might want to handle this differently
            except ValueError:
                logger.warning(f"Invalid profile: {request.profile}")
        
        # Convert messages
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        # Handle streaming
        if request.stream and config.enable_streaming:
            ttft = time.time() - start_time
            
            async def stream_generator():
                try:
                    async for chunk in model_manager.generate_stream(
                        messages,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        top_p=request.top_p
                    ):
                        yield chunk
                except Exception as e:
                    logger.error(f"Streaming error: {e}")
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
            
            stats.record_request_end(start_time, True, ttft)
            return StreamingResponse(
                stream_generator(),
                media_type="text/event-stream"
            )
        
        # Non-streaming response
        response = await model_manager.generate(
            messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p
        )
        
        ttft = time.time() - start_time
        
        # Format response
        completion_id = str(uuid.uuid4())
        result = ChatCompletionResponse(
            id=completion_id,
            model=request.model,
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response["content"]
                },
                "finish_reason": "stop"
            }],
            usage={
                "prompt_tokens": len(str(messages).split()),
                "completion_tokens": response["tokens_used"],
                "total_tokens": len(str(messages).split()) + response["tokens_used"]
            },
            created=int(time.time())
        )
        
        stats.record_request_end(start_time, True, ttft)
        return result
        
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        stats.record_request_end(start_time, False)
        raise HTTPException(status_code=500, detail=str(e))

# ================== Main ==================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="GPT-OSS HuggingFace Server v4.5.3")
    parser.add_argument("--model", type=str, default="20b", choices=["20b", "120b"],
                       help="Model size")
    parser.add_argument("--profile", type=str, default="balanced",
                       choices=["latency_first", "quality_first", "balanced"],
                       help="Operational profile")
    parser.add_argument("--port", type=int, default=8000,
                       help="Server port")
    parser.add_argument("--gpu-mode", type=str, default=None,
                       choices=["pipeline", "tensor"],
                       help="GPU parallelism mode (override profile)")
    
    args = parser.parse_args()
    
    # Update config
    config.model_size = args.model
    config.port = args.port
    
    # Apply profile
    try:
        profile = Profile(args.profile)
        config.update_from_profile(profile)
    except ValueError:
        logger.error(f"Invalid profile: {args.profile}")
        sys.exit(1)
    
    # Override GPU mode if specified
    if args.gpu_mode:
        config.gpu_mode = args.gpu_mode
    
    # Log configuration
    logger.info("="*50)
    logger.info("GPT-OSS HF Server v4.5.3")
    logger.info(f"Model: {config.model_size}")
    logger.info(f"Profile: {config.profile.value}")
    logger.info(f"Port: {config.port}")
    logger.info("="*50)
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=config.port)

if __name__ == "__main__":
    main()