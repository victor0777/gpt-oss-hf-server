#!/usr/bin/env python3
"""
GPT-OSS HuggingFace Server v4.5
With Engine Adapter Layer and vLLM/TRT-LLM support
"""

import os
import sys
import asyncio
import json
import time
import logging
import argparse
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
import uuid

from fastapi import FastAPI, HTTPException, Request
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

class ServerConfig:
    """Server configuration"""
    def __init__(self):
        # Engine configuration
        self.engine_type = os.getenv("ENGINE", "custom")
        self.engine_endpoint = os.getenv("ENGINE_ENDPOINT", "http://localhost:8000")
        self.vllm_endpoint = os.getenv("VLLM_ENDPOINT", "http://localhost:8001")
        self.trtllm_endpoint = os.getenv("TRTLLM_ENDPOINT", "http://localhost:8002")
        
        # Feature flags
        self.enable_otel = os.getenv("ENABLE_OTEL", "true").lower() == "true"
        self.enable_prefix_cache = os.getenv("ENABLE_PREFIX_CACHE", "true").lower() == "true"
        self.enable_kv_paging = os.getenv("ENABLE_KV_PAGING", "true").lower() == "true"
        self.enable_cont_batch = os.getenv("ENABLE_CONT_BATCH", "true").lower() == "true"
        self.enable_trtllm = os.getenv("ENABLE_TRTLLM", "false").lower() == "true"
        
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
        
        # SLO targets
        self.slo_p95_ttft_ms = int(os.getenv("SLO_P95_TTFT_MS", "7000"))
        self.slo_p95_e2e_ms = int(os.getenv("SLO_P95_E2E_MS", "20000"))
        self.slo_error_rate = float(os.getenv("SLO_ERROR_RATE", "0.005"))
        self.slo_target_qps = float(os.getenv("SLO_TARGET_QPS", "2.0"))

class MetricsCollector:
    """Collect and track metrics"""
    def __init__(self):
        self.requests_total = 0
        self.requests_success = 0
        self.requests_failed = 0
        self.ttft_histogram = []
        self.e2e_histogram = []
        self.engine_usage = {}
        self.start_time = time.time()
        
        if OTEL_AVAILABLE:
            self.tracer = trace.get_tracer(__name__)
            self.meter = metrics.get_meter(__name__)
            self._setup_metrics()
    
    def _setup_metrics(self):
        """Setup OTel metrics"""
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
    
    def record_request(self, engine: str, success: bool, ttft_ms: float, e2e_ms: float):
        """Record request metrics"""
        self.requests_total += 1
        if success:
            self.requests_success += 1
        else:
            self.requests_failed += 1
        
        self.ttft_histogram.append(ttft_ms)
        self.e2e_histogram.append(e2e_ms)
        
        if engine not in self.engine_usage:
            self.engine_usage[engine] = 0
        self.engine_usage[engine] += 1
        
        # Keep only last 1000 samples for percentile calculation
        if len(self.ttft_histogram) > 1000:
            self.ttft_histogram = self.ttft_histogram[-1000:]
        if len(self.e2e_histogram) > 1000:
            self.e2e_histogram = self.e2e_histogram[-1000:]
        
        # Record OTel metrics
        if OTEL_AVAILABLE:
            attributes = {"engine": engine, "success": str(success)}
            self.request_counter.add(1, attributes)
            self.latency_histogram.record(e2e_ms, {"type": "e2e", "engine": engine})
            self.latency_histogram.record(ttft_ms, {"type": "ttft", "engine": engine})
    
    def get_stats(self) -> dict:
        """Get current stats"""
        uptime = time.time() - self.start_time
        qps = self.requests_total / uptime if uptime > 0 else 0
        error_rate = (self.requests_failed / self.requests_total) if self.requests_total > 0 else 0
        
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
            "qps": round(qps, 2),
            "error_rate": round(error_rate, 4),
            "p95_ttft_ms": percentile(self.ttft_histogram, 95),
            "p95_e2e_ms": percentile(self.e2e_histogram, 95),
            "p50_ttft_ms": percentile(self.ttft_histogram, 50),
            "p50_e2e_ms": percentile(self.e2e_histogram, 50),
            "engine_usage": self.engine_usage,
            "uptime_seconds": round(uptime, 1)
        }

class ServerV45:
    """Main server with engine adapter layer"""
    
    def __init__(self, config: ServerConfig):
        self.config = config
        self.router = EngineRouter()
        self.metrics = MetricsCollector()
        self.active_requests = 0
        self.queue_length = 0
        
        # Initialize engines
        self._init_engines()
        
        # Initialize OTel if enabled
        if OTEL_AVAILABLE and config.enable_otel:
            self._init_otel()
    
    def _init_engines(self):
        """Initialize engine clients"""
        # Always register custom engine
        self.router.register_engine(
            "custom",
            CustomEngineClient(self.config.engine_endpoint)
        )
        
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
        """Initialize OpenTelemetry"""
        resource = Resource.create({
            "service.name": "gpt-oss-server",
            "service.version": "4.5"
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
    
    async def process_request(self, request: ChatCompletionRequest) -> dict:
        """Process chat completion request"""
        start_time = time.time()
        ttft = None
        
        try:
            self.active_requests += 1
            
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
            
            # Select engine
            engine_type = EngineType[self.config.engine_type.upper()]
            engine_name, engine = await self.router.select_engine(engine_request, engine_type)
            
            # Add tracing
            if OTEL_AVAILABLE and self.config.enable_otel:
                with self.metrics.tracer.start_as_current_span("chat_completion") as span:
                    span.set_attribute("engine", engine_name)
                    span.set_attribute("model", request.model)
                    span.set_attribute("max_tokens", request.max_tokens)
                    
                    # Generate response
                    response = await engine.generate(engine_request)
                    
                    # Record TTFT
                    if ttft is None:
                        ttft = (time.time() - start_time) * 1000
                    
                    span.set_attribute("ttft_ms", ttft)
                    span.set_attribute("e2e_ms", (time.time() - start_time) * 1000)
            else:
                # Generate without tracing
                response = await engine.generate(engine_request)
                
                # Record TTFT
                if ttft is None:
                    ttft = (time.time() - start_time) * 1000
            
            # Convert response to dict
            if isinstance(response, GenerateResponse):
                result = asdict(response)
            else:
                # Handle streaming
                result = response
            
            # Record metrics
            e2e = (time.time() - start_time) * 1000
            self.metrics.record_request(engine_name, True, ttft, e2e)
            
            # Add metadata
            if isinstance(result, dict):
                result["chosen_engine"] = engine_name
            
            return result
            
        except Exception as e:
            logger.error(f"Request processing error: {e}")
            e2e = (time.time() - start_time) * 1000
            self.metrics.record_request("unknown", False, 0, e2e)
            raise
        finally:
            self.active_requests -= 1
    
    async def health_check(self) -> dict:
        """Comprehensive health check"""
        healths = {}
        
        # Check each engine
        for name, engine in self.router.engines.items():
            try:
                health = await engine.health()
                healths[name] = asdict(health)
            except Exception as e:
                logger.error(f"Health check failed for {name}: {e}")
                healths[name] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
        
        # Get current stats
        stats = self.metrics.get_stats()
        
        # Calculate overall health score
        overall_score = 1.0
        
        # Check SLO violations
        if stats["error_rate"] > self.config.slo_error_rate:
            overall_score -= 0.3
        if stats["p95_e2e_ms"] > self.config.slo_p95_e2e_ms:
            overall_score -= 0.2
        if stats["p95_ttft_ms"] > self.config.slo_p95_ttft_ms:
            overall_score -= 0.1
        if stats["qps"] < self.config.slo_target_qps * 0.5:
            overall_score -= 0.2
        
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
                "otel_enabled": self.config.enable_otel,
                "features": {
                    "prefix_cache": self.config.enable_prefix_cache,
                    "kv_paging": self.config.enable_kv_paging,
                    "cont_batch": self.config.enable_cont_batch,
                    "trtllm": self.config.enable_trtllm
                }
            }
        }

# Global server instance
server: Optional[ServerV45] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle"""
    global server
    config = ServerConfig()
    server = ServerV45(config)
    logger.info(f"Server v4.5 started with engine: {config.engine_type}")
    yield
    logger.info("Server v4.5 shutting down")

# Create FastAPI app
app = FastAPI(
    title="GPT-OSS Server v4.5",
    version="4.5.0",
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
    """Health check endpoint"""
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
    """Get server statistics"""
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
    """Main chat completion endpoint"""
    if not server:
        raise HTTPException(status_code=503, detail="Server not ready")
    
    try:
        if request.stream:
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
    
    except Exception as e:
        logger.error(f"Chat completion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="GPT-OSS Server v4.5")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--engine", choices=["custom", "vllm", "trtllm", "auto"],
                       default="custom", help="Engine type")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    # Set environment variables from args
    os.environ["HOST"] = args.host
    os.environ["PORT"] = str(args.port)
    os.environ["ENGINE"] = args.engine
    os.environ["WORKERS"] = str(args.workers)
    
    # Run server
    uvicorn.run(
        "server_v45:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )

if __name__ == "__main__":
    main()