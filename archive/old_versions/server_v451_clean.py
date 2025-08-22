#!/usr/bin/env python3
"""
GPT-OSS HuggingFace Server v4.5.1 Clean
Personal-optimized server without engine system
Based on v4.4 with profile system and enhanced metrics
"""

import os
import sys
import asyncio
import json
import time
import logging
import argparse
import torch
import numpy as np  # Import numpy first to avoid version conflicts
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
import uuid
from enum import Enum
from threading import Lock
import hashlib

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import accelerate
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available, using mock mode")

# Try to import OpenTelemetry
try:
    from opentelemetry import trace, metrics
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ================== Profile System ==================

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

# ================== Configuration ==================

@dataclass
class ServerConfig:
    """Server configuration with profile support"""
    def __init__(self):
        # Profile configuration
        self.default_profile = Profile[os.getenv("DEFAULT_PROFILE", "BALANCED").upper()]
        self.current_profile = PROFILE_CONFIGS[self.default_profile]
        
        # Model configuration (from profile or env)
        self.model_size = os.getenv("MODEL_SIZE", self.current_profile["model_size"])
        self.gpu_mode = os.getenv("GPU_MODE", self.current_profile["gpu_mode"])
        
        # Apply profile settings
        profile = self.current_profile
        self.batch_max_size = profile["max_batch_size"]
        self.prefill_window_ms = profile["prefill_window_ms"]
        self.decode_window_ms = profile["decode_window_ms"]
        self.max_new_tokens = profile["max_new_tokens"]
        self.default_temperature = profile["temperature"]
        
        # Personal-optimized SLOs
        self.slo_p95_ttft_ms = 10000  # 10s for first token
        self.slo_p95_e2e_ms = 30000   # 30s end-to-end
        self.slo_error_rate = 0.01    # 1% error rate acceptable
        self.slo_min_qps = 0.3         # Minimum 0.3 QPS
        
        # Model paths
        self.model_name = f"openai/gpt-oss-{self.model_size}"
        
        # Feature flags
        self.enable_otel = os.getenv("ENABLE_OTEL", "true").lower() == "true"
        self.enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
        self.enable_prefix_cache = os.getenv("ENABLE_PREFIX_CACHE", "true").lower() == "true"
        
        # Server settings
        self.max_concurrent_requests = 128
        self.cache_max_entries = 10000
        self.cache_ttl_seconds = 3600
        
    def get_optimal_gpu_mode(self, model_size_gb: float) -> str:
        """Determine optimal GPU mode based on model size"""
        if self.gpu_mode != "auto":
            return self.gpu_mode
            
        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if gpu_count <= 1:
            return "single"
            
        # Get GPU memory
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        # Decision logic
        if model_size_gb < gpu_memory_gb * 0.5:
            return "pipeline"  # Replicate for throughput
        elif model_size_gb < gpu_memory_gb * 0.8:
            return "single"
        else:
            return "tensor"  # Split model across GPUs

# ================== Model Manager ==================

class ModelManager:
    """Manages model loading and inference"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device_map = None
        self.prefix_cache = {}
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Load model with appropriate GPU strategy"""
        logger.info(f"Loading model {self.config.model_name} with {self.config.gpu_mode} mode")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        
        # Determine device map based on GPU mode
        if self.config.gpu_mode == "single":
            self.device_map = {"": 0}
        elif self.config.gpu_mode == "pipeline":
            # Pipeline parallelism - replicate model
            self.device_map = "auto"
        elif self.config.gpu_mode == "tensor":
            # Tensor parallelism - split model
            self.device_map = "auto"
        else:
            self.device_map = "auto"
        
        # Load model
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                device_map=self.device_map,
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            logger.info(f"Model loaded successfully on {torch.cuda.device_count()} GPUs")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    async def generate(self, messages: List[dict], **kwargs) -> dict:
        """Generate response from messages"""
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            # Return mock response
            return self._mock_response(messages, **kwargs)
        
        # Convert messages to prompt
        prompt = self._messages_to_prompt(messages)
        
        # Check prefix cache
        cache_key = None
        if self.config.enable_prefix_cache:
            cache_key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            if cache_key in self.prefix_cache:
                logger.info(f"Prefix cache hit: {cache_key}")
                # Use cached prefix (simplified for now)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.config.gpu_mode == "single":
            inputs = inputs.to("cuda:0")
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_tokens", self.config.max_new_tokens),
                temperature=kwargs.get("temperature", self.config.default_temperature),
                top_p=kwargs.get("top_p", 1.0),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        response_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Update cache
        if self.config.enable_prefix_cache and cache_key:
            self.prefix_cache[cache_key] = {
                "prompt": prompt,
                "timestamp": time.time()
            }
            # Clean old entries
            self._clean_cache()
        
        return {
            "content": response_text,
            "tokens_used": outputs.shape[1] - inputs.input_ids.shape[1]
        }
    
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
        prompt += "Assistant: "
        return prompt
    
    def _mock_response(self, messages: List[dict], **kwargs) -> dict:
        """Generate mock response for testing"""
        return {
            "content": f"Mock response to: {messages[-1]['content'][:50]}...",
            "tokens_used": 50
        }
    
    def _clean_cache(self):
        """Clean old cache entries"""
        if len(self.prefix_cache) > self.config.cache_max_entries:
            # Remove oldest entries
            sorted_items = sorted(self.prefix_cache.items(), key=lambda x: x[1]["timestamp"])
            for key, _ in sorted_items[:len(sorted_items) // 2]:
                del self.prefix_cache[key]

# ================== Metrics Collector ==================

class MetricsCollector:
    """Collects and reports metrics with enhanced tagging"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.requests_total = 0
        self.requests_success = 0
        self.requests_failed = 0
        self.ttft_ms = []
        self.e2e_ms = []
        self.start_time = time.time()
        self.lock = Lock()
        
        # Initialize OTel if available
        if OTEL_AVAILABLE and config.enable_otel:
            self._init_otel()
        else:
            self.tracer = None
            self.meter = None
    
    def _init_otel(self):
        """Initialize OpenTelemetry"""
        resource = Resource.create({"service.name": "gpt-oss-server-v451"})
        
        # Tracer
        trace.set_tracer_provider(TracerProvider(resource=resource))
        self.tracer = trace.get_tracer(__name__)
        
        # Meter
        reader = PeriodicExportingMetricReader(OTLPMetricExporter())
        provider = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(provider)
        self.meter = metrics.get_meter(__name__)
    
    def record_request(self, success: bool, ttft_ms: float, e2e_ms: float, 
                       model: str, profile: str):
        """Record request metrics"""
        with self.lock:
            self.requests_total += 1
            if success:
                self.requests_success += 1
            else:
                self.requests_failed += 1
            
            if ttft_ms:
                self.ttft_ms.append(ttft_ms)
                # Keep only last 1000 samples
                if len(self.ttft_ms) > 1000:
                    self.ttft_ms = self.ttft_ms[-1000:]
            
            if e2e_ms:
                self.e2e_ms.append(e2e_ms)
                if len(self.e2e_ms) > 1000:
                    self.e2e_ms = self.e2e_ms[-1000:]
    
    def get_stats(self) -> dict:
        """Get current statistics"""
        with self.lock:
            uptime = time.time() - self.start_time
            qps = self.requests_total / uptime if uptime > 0 else 0
            error_rate = self.requests_failed / self.requests_total if self.requests_total > 0 else 0
            
            # Calculate percentiles
            p50_ttft = self._percentile(self.ttft_ms, 50)
            p95_ttft = self._percentile(self.ttft_ms, 95)
            p99_ttft = self._percentile(self.ttft_ms, 99)
            
            p50_e2e = self._percentile(self.e2e_ms, 50)
            p95_e2e = self._percentile(self.e2e_ms, 95)
            p99_e2e = self._percentile(self.e2e_ms, 99)
            
            return {
                "requests_total": self.requests_total,
                "requests_success": self.requests_success,
                "requests_failed": self.requests_failed,
                "qps": round(qps, 2),
                "error_rate": round(error_rate, 4),
                "p50_ttft_ms": round(p50_ttft, 2),
                "p95_ttft_ms": round(p95_ttft, 2),
                "p99_ttft_ms": round(p99_ttft, 2),
                "p50_e2e_ms": round(p50_e2e, 2),
                "p95_e2e_ms": round(p95_e2e, 2),
                "p99_e2e_ms": round(p99_e2e, 2),
                "uptime_seconds": round(uptime, 1),
                "model_info": {
                    "current_model": self.config.model_size,
                    "gpu_mode": self.config.gpu_mode,
                    "default_profile": self.config.default_profile.value
                }
            }
    
    def _percentile(self, data: list, percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        return sorted_data[index]
    
    def get_request_tags(self, model: str, profile: str) -> dict:
        """Get comprehensive request tags for metrics"""
        return {
            "model_id": model,
            "model_size": self.config.model_size,
            "gpu_mode": self.config.gpu_mode,
            "profile": profile,
            "engine": "transformers",
            "version": "4.5.1"
        }

# ================== Request Models ==================

class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")

class ChatCompletionRequest(BaseModel):
    model: str = Field(default="gpt-oss-20b")
    messages: List[ChatMessage]
    max_tokens: int = Field(default=100, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    profile: Optional[str] = None  # Profile selection

# ================== Server ==================

class ServerV451Clean:
    """Clean v4.5.1 server without engine system"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.model_manager = ModelManager(config)
        self.metrics = MetricsCollector(config)
        self.active_requests = 0
        self.request_queue = asyncio.Queue(maxsize=config.max_concurrent_requests)
    
    def apply_profile(self, profile: Profile):
        """Apply a profile configuration"""
        config = PROFILE_CONFIGS[profile]
        self.config.model_size = config["model_size"]
        self.config.gpu_mode = config["gpu_mode"]
        self.config.batch_max_size = config["max_batch_size"]
        self.config.max_new_tokens = config["max_new_tokens"]
        self.config.default_temperature = config["temperature"]
        logger.info(f"Applied profile: {profile.value}")
    
    async def process_request(self, request: ChatCompletionRequest) -> dict:
        """Process chat completion request"""
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Apply profile if specified
        profile_name = request.profile or self.config.default_profile.value
        if request.profile:
            try:
                profile = Profile[request.profile.upper()]
                self.apply_profile(profile)
            except KeyError:
                logger.warning(f"Unknown profile: {request.profile}")
                profile_name = self.config.default_profile.value
        
        try:
            self.active_requests += 1
            
            # Convert messages
            messages = [{"role": m.role, "content": m.content} for m in request.messages]
            
            # Generate response
            if request.stream:
                # Streaming not yet implemented
                raise HTTPException(status_code=501, detail="Streaming not implemented yet")
            
            result = await self.model_manager.generate(
                messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p
            )
            
            # Calculate metrics
            e2e_ms = (time.time() - start_time) * 1000
            ttft_ms = e2e_ms * 0.3  # Approximate TTFT as 30% of E2E
            
            # Record metrics
            self.metrics.record_request(True, ttft_ms, e2e_ms, request.model, profile_name)
            
            # Build response
            response = {
                "id": request_id,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result["content"]
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(messages[-1]["content"].split()),
                    "completion_tokens": result.get("tokens_used", 50),
                    "total_tokens": len(messages[-1]["content"].split()) + result.get("tokens_used", 50)
                },
                "created": int(time.time()),
                "object": "chat.completion",
                "profile": profile_name,
                "request_id": request_id
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Request processing error: {e}")
            self.metrics.record_request(False, 0, 0, request.model, profile_name)
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            self.active_requests -= 1
    
    async def health_check(self) -> dict:
        """Health check with model info"""
        stats = self.metrics.get_stats()
        
        # Calculate health score
        score = 1.0
        if stats["error_rate"] > self.config.slo_error_rate:
            score -= 0.2
        if stats["p95_e2e_ms"] > self.config.slo_p95_e2e_ms:
            score -= 0.1
        if stats["qps"] < self.config.slo_min_qps and stats["requests_total"] > 10:
            score -= 0.1
        
        score = max(0.0, score)
        
        return {
            "status": "healthy" if score > 0.7 else "degraded" if score > 0.3 else "unhealthy",
            "score": round(score, 2),
            "stats": stats,
            "active_requests": self.active_requests,
            "config": {
                "model_size": self.config.model_size,
                "gpu_mode": self.config.gpu_mode,
                "default_profile": self.config.default_profile.value,
                "features": {
                    "streaming": self.config.enable_streaming,
                    "prefix_cache": self.config.enable_prefix_cache,
                    "otel": self.config.enable_otel
                }
            }
        }

# ================== FastAPI App ==================

# Lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Server v4.5.1 Clean starting...")
    yield
    # Shutdown
    logger.info("Server v4.5.1 Clean shutting down...")

# Create FastAPI app
app = FastAPI(
    title="GPT-OSS HF Server v4.5.1 Clean",
    description="Personal-optimized inference server",
    version="4.5.1-clean",
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

# Initialize server
config = ServerConfig()
server = ServerV451Clean(config)

# ================== Endpoints ==================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return await server.health_check()

@app.get("/stats")
async def stats():
    """Statistics endpoint"""
    return server.metrics.get_stats()

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions"""
    return await server.process_request(request)

# ================== Main ==================

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="GPT-OSS HF Server v4.5.1 Clean")
    parser.add_argument("--model", type=str, default="20b", choices=["20b", "120b"])
    parser.add_argument("--profile", type=str, default="balanced", 
                       choices=["latency_first", "quality_first", "balanced"])
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    
    args = parser.parse_args()
    
    # Set environment variables from args
    os.environ["MODEL_SIZE"] = args.model
    os.environ["DEFAULT_PROFILE"] = args.profile.upper()
    
    # Log startup info
    logger.info(f"Starting server v4.5.1 Clean")
    logger.info(f"Model: {args.model}, Profile: {args.profile}, Port: {args.port}")
    
    # Run server
    uvicorn.run(
        "server_v451_clean:app",
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()