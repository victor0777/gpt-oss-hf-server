#!/usr/bin/env python3
"""
GPT-OSS HuggingFace Server v4.5.1
Enhanced metrics tagging, personal profiles, and streaming improvements
"""

import os
import sys
import asyncio
import json
import time
import logging
import argparse
from typing import Dict, List, Optional, Any, AsyncIterator
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
import uuid
from enum import Enum

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import engine clients
from engine_client import (
    EngineType, EngineClient, EngineRouter,
    CustomEngineClient, VllmEngineClient, TrtLlmEngineClient,
    GenerateRequest, GenerateResponse, EngineHealth,
    create_engine_client
)

# OpenTelemetry imports (optional)
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Profile types
class Profile(Enum):
    LATENCY_FIRST = "latency_first"  # Daily coding, fast responses
    QUALITY_FIRST = "quality_first"  # Long-form, review, high quality
    BALANCED = "balanced"  # Default balanced mode

# Profile configurations
PROFILE_CONFIGS = {
    Profile.LATENCY_FIRST: {
        "model_size": "20b",
        "gpu_mode": "pipeline",
        "max_batch_size": 32,
        "prefill_window_ms": 4,
        "decode_window_ms": 2,
        "max_new_tokens": 512,
        "temperature": 0.7
    },
    Profile.QUALITY_FIRST: {
        "model_size": "120b",
        "gpu_mode": "tensor",  # Or hybrid TP2+PP2
        "max_batch_size": 8,
        "prefill_window_ms": 12,
        "decode_window_ms": 8,
        "max_new_tokens": 2048,
        "temperature": 0.8
    },
    Profile.BALANCED: {
        "model_size": "20b",
        "gpu_mode": "pipeline",
        "max_batch_size": 48,
        "prefill_window_ms": 6,
        "decode_window_ms": 4,
        "max_new_tokens": 1024,
        "temperature": 0.75
    }
}

# Request/Response models
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "gpt-oss-20b"
    messages: List[ChatMessage]
    max_tokens: int = Field(default=100, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    profile: Optional[str] = None  # New: profile selection

class ServerConfig:
    """Server configuration with profiles"""
    def __init__(self):
        # Engine configuration
        self.engine_type = os.getenv("ENGINE", "custom")
        self.engine_endpoint = os.getenv("ENGINE_ENDPOINT", "http://localhost:8001")
        self.vllm_endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8001")
        self.trtllm_endpoint = os.getenv("TRTLLM_ENDPOINT", "http://localhost:8002")
        
        # Profile configuration
        self.default_profile = Profile[os.getenv("DEFAULT_PROFILE", "BALANCED").upper()]
        self.current_profile = PROFILE_CONFIGS[self.default_profile]
        
        # Model configuration (from profile or env)
        self.model_size = os.getenv("MODEL_SIZE", self.current_profile["model_size"])
        self.gpu_mode = os.getenv("GPU_MODE", self.current_profile["gpu_mode"])
        
        # Feature flags
        self.enable_otel = os.getenv("ENABLE_OTEL", "true").lower() == "true"
        self.enable_prefix_cache = os.getenv("ENABLE_PREFIX_CACHE", "true").lower() == "true"
        self.enable_kv_paging = os.getenv("ENABLE_KV_PAGING", "true").lower() == "true"
        self.enable_cont_batch = os.getenv("ENABLE_CONT_BATCH", "true").lower() == "true"
        self.enable_trtllm = os.getenv("ENABLE_TRTLLM", "false").lower() == "true"
        self.enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
        self.enable_cancel = os.getenv("ENABLE_CANCEL", "true").lower() == "true"
        
        # Auto-disable features for non-custom engines
        if self.engine_type != "custom":
            self.enable_prefix_cache = False
            self.enable_kv_paging = False
            self.enable_cont_batch = False
            logger.info(f"Disabled custom features for {self.engine_type} engine")
        
        # Service configuration
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.workers = int(os.getenv("WORKERS", "1"))
        
        # Personal SLO targets (relaxed for personal use)
        self.slo_p95_ttft_ms = int(os.getenv("SLO_P95_TTFT_MS", "10000"))
        self.slo_p95_e2e_ms = int(os.getenv("SLO_P95_E2E_MS", "30000"))
        self.slo_error_rate = float(os.getenv("SLO_ERROR_RATE", "0.01"))
        self.slo_target_qps = float(os.getenv("SLO_TARGET_QPS", "0.5"))  # Relaxed for personal

class MetricsCollector:
    """Enhanced metrics collector with detailed tagging"""
    def __init__(self, config: ServerConfig):
        self.config = config
        self.requests_total = 0
        self.requests_success = 0
        self.requests_failed = 0
        self.requests_cancelled = 0
        self.ttft_histogram = []
        self.e2e_histogram = []
        self.engine_usage = {}
        self.model_usage = {}
        self.profile_usage = {}
        self.start_time = time.time()
        self.active_streams = set()
        
        if OTEL_AVAILABLE and config.enable_otel:
            self.tracer = trace.get_tracer(__name__)
            self.meter = metrics.get_meter(__name__)
            self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup OTel metrics with enhanced tags"""
        if not OTEL_AVAILABLE:
            return
            
        self.request_counter = self.meter.create_counter(
            "gptoss.requests.total",
            description="Total number of requests"
        )
        
        self.latency_histogram = self.meter.create_histogram(
            "gptoss.latency.ms",
            description="Request latency in milliseconds"
        )
        
        self.active_requests = self.meter.create_up_down_counter(
            "gptoss.requests.active",
            description="Number of active requests"
        )
        
        self.streaming_gauge = self.meter.create_up_down_counter(
            "gptoss.streams.active",
            description="Number of active streaming connections"
        )
    
    def get_request_tags(self, engine: str, model: str, profile: str) -> dict:
        """Get comprehensive tags for metrics"""
        return {
            "engine": engine,
            "model_id": model,
            "model_size": self.config.model_size,
            "gpu_mode": self.config.gpu_mode,
            "tp": "4" if self.config.gpu_mode == "tensor" else "1",
            "pp": "4" if self.config.gpu_mode == "pipeline" else "1",
            "micro_batches": str(self.config.current_profile.get("max_batch_size", 32)),
            "max_model_len": str(self.config.current_profile.get("max_new_tokens", 1024)),
            "profile": profile
        }
    
    def record_request(self, engine: str, model: str, profile: str, 
                      success: bool, ttft_ms: float, e2e_ms: float, cancelled: bool = False):
        """Record request metrics with enhanced tagging"""
        self.requests_total += 1
        if cancelled:
            self.requests_cancelled += 1
        elif success:
            self.requests_success += 1
        else:
            self.requests_failed += 1
        
        if not cancelled:
            self.ttft_histogram.append(ttft_ms)
            self.e2e_histogram.append(e2e_ms)
        
        # Track usage by dimension
        if engine not in self.engine_usage:
            self.engine_usage[engine] = 0
        self.engine_usage[engine] += 1
        
        if model not in self.model_usage:
            self.model_usage[model] = 0
        self.model_usage[model] += 1
        
        if profile not in self.profile_usage:
            self.profile_usage[profile] = 0
        self.profile_usage[profile] += 1
        
        # Keep only last 1000 samples
        if len(self.ttft_histogram) > 1000:
            self.ttft_histogram = self.ttft_histogram[-1000:]
        if len(self.e2e_histogram) > 1000:
            self.e2e_histogram = self.e2e_histogram[-1000:]
        
        # Record OTel metrics with full tags
        if OTEL_AVAILABLE and self.config.enable_otel:
            tags = self.get_request_tags(engine, model, profile)
            tags["success"] = str(success)
            tags["cancelled"] = str(cancelled)
            
            self.request_counter.add(1, tags)
            if not cancelled:
                self.latency_histogram.record(e2e_ms, {**tags, "type": "e2e"})
                self.latency_histogram.record(ttft_ms, {**tags, "type": "ttft"})
    
    def add_stream(self, stream_id: str):
        """Track active stream"""
        self.active_streams.add(stream_id)
        if OTEL_AVAILABLE and self.config.enable_otel:
            self.streaming_gauge.add(1)
    
    def remove_stream(self, stream_id: str):
        """Remove completed stream"""
        if stream_id in self.active_streams:
            self.active_streams.discard(stream_id)
            if OTEL_AVAILABLE and self.config.enable_otel:
                self.streaming_gauge.add(-1)
    
    def get_stats(self) -> dict:
        """Get current stats with enhanced metadata"""
        uptime = time.time() - self.start_time
        qps = self.requests_total / uptime if uptime > 0 else 0
        error_rate = (self.requests_failed / self.requests_total) if self.requests_total > 0 else 0
        cancel_rate = (self.requests_cancelled / self.requests_total) if self.requests_total > 0 else 0
        
        # Calculate percentiles
        def percentile(data, p):
            if not data:
                return 0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data)-1)]
        
        return {
            "requests_total": self.requests_total,
            "requests_success": self.requests_success,
            "requests_failed": self.requests_failed,
            "requests_cancelled": self.requests_cancelled,
            "active_streams": len(self.active_streams),
            "qps": round(qps, 2),
            "error_rate": round(error_rate, 4),
            "cancel_rate": round(cancel_rate, 4),
            "p50_ttft_ms": percentile(self.ttft_histogram, 50),
            "p95_ttft_ms": percentile(self.ttft_histogram, 95),
            "p99_ttft_ms": percentile(self.ttft_histogram, 99),
            "p50_e2e_ms": percentile(self.e2e_histogram, 50),
            "p95_e2e_ms": percentile(self.e2e_histogram, 95),
            "p99_e2e_ms": percentile(self.e2e_histogram, 99),
            "engine_usage": self.engine_usage,
            "model_usage": self.model_usage,
            "profile_usage": self.profile_usage,
            "uptime_seconds": round(uptime, 1),
            "model_info": {
                "current_model": self.config.model_size,
                "gpu_mode": self.config.gpu_mode,
                "default_profile": self.config.default_profile.value
            }
        }

class RequestManager:
    """Manage active requests and cancellation"""
    def __init__(self):
        self.active_requests = {}
        self.cancelled_requests = set()
    
    def add_request(self, request_id: str) -> str:
        """Add new request"""
        self.active_requests[request_id] = {
            "start_time": time.time(),
            "cancelled": False
        }
        return request_id
    
    def cancel_request(self, request_id: str) -> bool:
        """Cancel a request"""
        if request_id in self.active_requests:
            self.active_requests[request_id]["cancelled"] = True
            self.cancelled_requests.add(request_id)
            return True
        return False
    
    def is_cancelled(self, request_id: str) -> bool:
        """Check if request is cancelled"""
        return request_id in self.cancelled_requests
    
    def remove_request(self, request_id: str):
        """Remove completed request"""
        self.active_requests.pop(request_id, None)
        self.cancelled_requests.discard(request_id)

class ServerV451:
    """Main server with enhanced metrics and profiles"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.router = EngineRouter()
        self.metrics = MetricsCollector(config)
        self.request_manager = RequestManager()
        self.active_requests = 0
        self.queue_length = 0
        
        # Initialize engines
        self._init_engines()
        
        # Initialize OTel if enabled
        if OTEL_AVAILABLE and config.enable_otel:
            self._init_otel()
    
    def _init_engines(self):
        """Initialize engine clients"""
        # Skip engine registration in v4.5.1 standalone mode
        # We don't actually use external engines, just internal processing
        pass  # Engines are not used in v4.5.1 standalone mode
        
        # Register vLLM if available
        if self.config.engine_type in ["vllm", "auto"]:
            self.router.register_engine(
                "vllm",
                VllmEngineClient(self.config.vllm_endpoint)
            )
        
        # Register TRT-LLM if enabled
        if self.config.enable_trtllm and self.config.engine_type in ["trtllm", "auto"]:
            self.router.register_engine(
                "trtllm",
                TrtLlmEngineClient(self.config.trtllm_endpoint)
            )
        
        logger.info(f"Initialized engines: {list(self.router.engines.keys())}")
    
    def _init_otel(self):
        """Initialize OpenTelemetry with enhanced resource attributes"""
        resource = Resource.create({
            "service.name": "gpt-oss-server",
            "service.version": "4.5.1",
            "model.size": self.config.model_size,
            "gpu.mode": self.config.gpu_mode,
            "engine.type": self.config.engine_type
        })
        
        # Setup tracing
        trace.set_tracer_provider(TracerProvider(resource=resource))
        tracer_provider = trace.get_tracer_provider()
        
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            span_exporter = OTLPSpanExporter()
            tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        
        # Setup metrics
        if os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"):
            metric_reader = PeriodicExportingMetricReader(
                exporter=OTLPMetricExporter(),
                export_interval_millis=10000
            )
            metrics.set_meter_provider(MeterProvider(
                resource=resource,
                metric_readers=[metric_reader]
            ))
    
    def apply_profile(self, profile: Profile):
        """Apply a performance profile"""
        config = PROFILE_CONFIGS[profile]
        self.config.current_profile = config
        self.config.model_size = config["model_size"]
        self.config.gpu_mode = config["gpu_mode"]
        logger.info(f"Applied profile: {profile.value}")
    
    async def process_request(self, request: ChatCompletionRequest) -> dict:
        """Process chat completion request with enhanced tracking"""
        start_time = time.time()
        ttft = None
        request_id = str(uuid.uuid4())
        self.request_manager.add_request(request_id)
        
        # Apply profile if specified
        profile_name = request.profile or self.config.default_profile.value
        if request.profile:
            try:
                profile = Profile[request.profile.upper()]
                self.apply_profile(profile)
            except KeyError:
                logger.warning(f"Unknown profile: {request.profile}, using default")
                profile_name = self.config.default_profile.value
        
        try:
            self.active_requests += 1
            
            # Check for cancellation
            if self.request_manager.is_cancelled(request_id):
                raise HTTPException(status_code=499, detail="Request cancelled")
            
            # Convert to engine request
            engine_request = GenerateRequest(
                messages=[{"role": m.role, "content": m.content} for m in request.messages],
                model=request.model,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=request.stream,
                stop=request.stop,
                presence_penalty=request.presence_penalty,
                frequency_penalty=request.frequency_penalty
            )
            
            # In v4.5.1 standalone, we use mock responses
            # Real model processing would be implemented here
            engine_name = "custom"
            engine = None  # We'll generate mock response directly
            
            # Add comprehensive tracing
            if OTEL_AVAILABLE and self.config.enable_otel:
                with self.metrics.tracer.start_as_current_span("chat_completion") as span:
                    # Add all required tags
                    tags = self.metrics.get_request_tags(engine_name, request.model, profile_name)
                    for key, value in tags.items():
                        span.set_attribute(key, value)
                    span.set_attribute("request_id", request_id)
                    
                    # Generate mock response for v4.5.1 standalone
                    if request.stream and self.config.enable_streaming:
                        # Streaming not implemented in mock mode
                        raise HTTPException(status_code=501, detail="Streaming not implemented in standalone mode")
                    else:
                        # Generate mock response
                        await asyncio.sleep(0.5)  # Simulate processing time
                        response = self._generate_mock_response(request, request_id)
                    
                    # Record TTFT
                    if ttft is None:
                        ttft = (time.time() - start_time) * 1000
                    
                    span.set_attribute("ttft_ms", ttft)
                    span.set_attribute("e2e_ms", (time.time() - start_time) * 1000)
            else:
                # Generate without tracing
                if request.stream and self.config.enable_streaming:
                    response = await self._handle_streaming(
                        engine, engine_request, request_id, engine_name
                    )
                else:
                    response = await engine.generate(engine_request)
                
                # Record TTFT
                if ttft is None:
                    ttft = (time.time() - start_time) * 1000
            
            # Convert response to dict
            if isinstance(response, GenerateResponse):
                result = asdict(response)
            else:
                result = response
            
            # Record metrics
            e2e = (time.time() - start_time) * 1000
            cancelled = self.request_manager.is_cancelled(request_id)
            self.metrics.record_request(
                engine_name, request.model, profile_name,
                not cancelled, ttft, e2e, cancelled
            )
            
            # Add metadata
            if isinstance(result, dict):
                result["chosen_engine"] = engine_name
                result["profile"] = profile_name
                result["request_id"] = request_id
            
            return result
            
        except Exception as e:
            logger.error(f"Request processing error: {e}")
            e2e = (time.time() - start_time) * 1000
            self.metrics.record_request(
                "unknown", request.model, profile_name,
                False, 0, e2e, False
            )
            raise
        finally:
            self.active_requests -= 1
            self.request_manager.remove_request(request_id)
    
    async def _handle_streaming(self, engine, request, request_id, engine_name):
        """Handle streaming with cancellation support"""
        stream_id = f"stream_{request_id}"
        self.metrics.add_stream(stream_id)
        
        try:
            async for chunk in engine.generate(request):
                # Check cancellation
                if self.request_manager.is_cancelled(request_id):
                    break
                yield chunk
        finally:
            self.metrics.remove_stream(stream_id)
    
    async def health_check(self) -> dict:
        """Comprehensive health check with model info"""
        healths = {}
        
        # In v4.5.1 standalone mode, we don't use external engines
        # Just report internal status
        healths["custom"] = {
            "status": "healthy",
            "latency_ms": 0,
            "score": 1.0,
            "engine_name": "custom",
            "engine_version": "4.5.1",
            "gpu_utilization": 0,
            "memory_used_gb": 0,
            "active_requests": self.active_requests,
            "queued_requests": self.queue_length
        }
        
        # Get current stats
        stats = self.metrics.get_stats()
        
        # Calculate overall health score (relaxed for personal use)
        overall_score = 1.0
        
        # Check SLO violations (relaxed thresholds)
        if stats["error_rate"] > self.config.slo_error_rate:
            overall_score -= 0.2
        if stats["p95_e2e_ms"] > self.config.slo_p95_e2e_ms:
            overall_score -= 0.1
        if stats["p95_ttft_ms"] > self.config.slo_p95_ttft_ms:
            overall_score -= 0.1
        # QPS check removed for personal use
        
        overall_score = max(0.0, overall_score)
        
        return {
            "status": "healthy" if overall_score > 0.7 else "degraded" if overall_score > 0.3 else "unhealthy",
            "score": round(overall_score, 2),
            "engines": healths,
            "stats": stats,
            "active_requests": self.active_requests,
            "queue_length": self.queue_length,
            "config": {
                "engine_type": self.config.engine_type,
                "model_size": self.config.model_size,
                "gpu_mode": self.config.gpu_mode,
                "default_profile": self.config.default_profile.value,
                "current_profile": self.config.current_profile,
                "otel_enabled": self.config.enable_otel,
                "features": {
                    "prefix_cache": self.config.enable_prefix_cache,
                    "kv_paging": self.config.enable_kv_paging,
                    "cont_batch": self.config.enable_cont_batch,
                    "trtllm": self.config.enable_trtllm,
                    "streaming": self.config.enable_streaming,
                    "cancel": self.config.enable_cancel
                }
            }
        }

# Global server instance
server: Optional[ServerV451] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle"""
    global server
    config = ServerConfig()
    server = ServerV451(config)
    logger.info(f"Server v4.5.1 started")
    logger.info(f"Engine: {config.engine_type}, Model: {config.model_size}, Profile: {config.default_profile.value}")
    yield
    logger.info("Server v4.5.1 shutting down")

# Create FastAPI app
app = FastAPI(
    title="GPT-OSS Server v4.5.1",
    version="4.5.1",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/health")
async def health():
    """Health check endpoint with full metadata"""
    if server:
        return await server.health_check()
    return {"status": "starting"}

@app.get("/health/score")
async def health_score():
    """Simple health score endpoint"""
    if server:
        health = await server.health_check()
        return {"score": health["score"]}
    return {"score": 0.0}

@app.get("/stats")
async def stats():
    """Get server statistics with model info"""
    if server:
        return server.metrics.get_stats()
    return {}

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [
            {"id": "gpt-oss-20b", "object": "model", "owned_by": "openai"},
            {"id": "gpt-oss-120b", "object": "model", "owned_by": "openai"}
        ]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Main chat completion endpoint with profile support"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    try:
        if request.stream and server.config.enable_streaming:
            # Streaming response
            async def generate():
                response = await server.process_request(request)
                async for chunk in response:
                    yield f"data: {json.dumps(asdict(chunk))}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate(),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming response
            response = await server.process_request(request)
            return JSONResponse(content=response)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/cancel/{request_id}")
async def cancel_request(request_id: str):
    """Cancel an active request"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    if not server.config.enable_cancel:
        raise HTTPException(status_code=501, detail="Cancellation not enabled")
    
    if server.request_manager.cancel_request(request_id):
        return {"status": "cancelled", "request_id": request_id}
    else:
        raise HTTPException(status_code=404, detail="Request not found")

@app.get("/profiles")
async def list_profiles():
    """List available profiles"""
    return {
        "profiles": [
            {
                "name": Profile.LATENCY_FIRST.value,
                "description": "Optimized for low latency (daily coding)",
                "config": PROFILE_CONFIGS[Profile.LATENCY_FIRST]
            },
            {
                "name": Profile.QUALITY_FIRST.value,
                "description": "Optimized for quality (long-form content)",
                "config": PROFILE_CONFIGS[Profile.QUALITY_FIRST]
            },
            {
                "name": Profile.BALANCED.value,
                "description": "Balanced performance",
                "config": PROFILE_CONFIGS[Profile.BALANCED]
            }
        ],
        "default": server.config.default_profile.value if server else "balanced"
    }

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="GPT-OSS Server v4.5.1")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--engine", choices=["custom", "vllm", "trtllm", "auto"],
                       default="custom", help="Engine type")
    parser.add_argument("--profile", choices=["latency_first", "quality_first", "balanced"],
                       default="balanced", help="Default profile")
    parser.add_argument("--model", choices=["20b", "120b"],
                       default="20b", help="Model size")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    # Set environment variables from args
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)
    os.environ["ENGINE"] = args.engine
    os.environ["DEFAULT_PROFILE"] = args.profile
    os.environ["MODEL_SIZE"] = args.model
    os.environ["WORKERS"] = str(args.workers)
    
    # Run server
    uvicorn.run(
        "server_v451:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )

if __name__ == "__main__":
    main()