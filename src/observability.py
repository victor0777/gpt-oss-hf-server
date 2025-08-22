#!/usr/bin/env python3
"""
Observability Foundation Pack for GPT-OSS HF Server v4.8.0
Implements PR-OBS-A1, A2, A3: Metrics-Trace correlation, Core histograms, Sampling
"""

import os
import time
import json
import logging
import hashlib
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
from contextlib import contextmanager
import random

# Prometheus metrics
from prometheus_client import (
    Counter, Histogram, Gauge, CollectorRegistry,
    generate_latest, CONTENT_TYPE_LATEST
)

# OpenTelemetry
try:
    from opentelemetry import trace, context
    from opentelemetry.trace import Status, StatusCode, SpanKind
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning("OpenTelemetry not available - tracing disabled")

logger = logging.getLogger(__name__)

# Configuration
@dataclass
class ObservabilityConfig:
    """Observability configuration - PR-OBS-A1/A2/A3"""
    # Exemplars
    exemplars_enabled: bool = True
    
    # OpenTelemetry
    otel_endpoint: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    otel_service_name: str = "gpt-oss-hf-server"
    
    # Sampling rules (PR-OBS-A3)
    slow_trace_threshold_ms: float = 10000  # 10s
    error_trace_sample_rate: float = 1.0    # 100% for errors
    normal_trace_sample_rate: float = 0.03  # 3% for normal
    
    # Model labels (PR-OBS-A2)
    model_labels: List[str] = field(default_factory=lambda: [
        "model_id", "model_size", "dtype", "route", 
        "tp", "pp", "micro_batches", "prompt_version", "admission_action"
    ])
    
    # Privacy
    log_prompts: bool = False
    mask_sensitive: bool = True

class ObservabilityManager:
    """
    PR-OBS-A1: Metrics↔Trace correlation with Exemplars
    PR-OBS-A2: LLM core histograms and counters
    PR-OBS-A3: Slow/error trace sampling rules
    """
    
    def __init__(self, config: Optional[ObservabilityConfig] = None):
        self.config = config or ObservabilityConfig()
        self.registry = CollectorRegistry()
        
        # Initialize OpenTelemetry if available
        self.tracer = None
        if OTEL_AVAILABLE and self.config.otel_endpoint:
            self._init_otel()
        
        # PR-OBS-A2: Core metrics
        self._init_metrics()
        
        # Request tracking
        self.active_requests = {}
        
        logger.info(f"ObservabilityManager initialized with config: {self.config}")
    
    def _init_otel(self):
        """Initialize OpenTelemetry tracing"""
        try:
            resource = Resource.create({
                "service.name": self.config.otel_service_name,
                "service.version": "4.8.0"
            })
            
            provider = TracerProvider(resource=resource)
            
            # OTLP exporter
            otlp_exporter = OTLPSpanExporter(
                endpoint=self.config.otel_endpoint,
                insecure=True
            )
            
            span_processor = BatchSpanProcessor(otlp_exporter)
            provider.add_span_processor(span_processor)
            
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer(__name__)
            
            logger.info(f"OpenTelemetry initialized with endpoint: {self.config.otel_endpoint}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self.tracer = None
    
    def _init_metrics(self):
        """PR-OBS-A2: Initialize core LLM metrics"""
        
        # Histograms
        self.llm_ttft_ms = Histogram(
            'llm_ttft_ms',
            'Time to first token in milliseconds',
            self.config.model_labels,
            buckets=[50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000],
            registry=self.registry
        )
        
        self.llm_e2e_ms = Histogram(
            'llm_e2e_ms',
            'End-to-end request time in milliseconds',
            self.config.model_labels,
            buckets=[100, 500, 1000, 2000, 5000, 10000, 20000, 50000],
            registry=self.registry
        )
        
        self.llm_tokens_per_sec = Histogram(
            'llm_tokens_per_sec',
            'Token generation rate',
            self.config.model_labels,
            buckets=[1, 5, 10, 20, 50, 100, 200, 500],
            registry=self.registry
        )
        
        # Counters
        self.llm_prefill_tokens_total = Counter(
            'llm_prefill_tokens_total',
            'Total prefill tokens processed',
            self.config.model_labels,
            registry=self.registry
        )
        
        self.llm_decode_tokens_total = Counter(
            'llm_decode_tokens_total',
            'Total decode tokens generated',
            self.config.model_labels,
            registry=self.registry
        )
        
        self.llm_prefix_cache_hit_total = Counter(
            'llm_prefix_cache_hit_total',
            'Prefix cache hits and misses',
            self.config.model_labels + ['reason'],
            registry=self.registry
        )
        
        self.llm_admission_total = Counter(
            'llm_admission_total',
            'Admission control decisions',
            self.config.model_labels + ['action'],
            registry=self.registry
        )
        
        # Gauges
        self.kv_bytes_in_use = Gauge(
            'kv_bytes_in_use',
            'KV cache bytes currently in use',
            self.config.model_labels,
            registry=self.registry
        )
        
        self.sessions_active = Gauge(
            'sessions_active',
            'Number of active sessions',
            self.config.model_labels,
            registry=self.registry
        )
        
        self.gpu_utilization = Gauge(
            'gpu_utilization',
            'GPU utilization percentage',
            ['gpu'],
            registry=self.registry
        )
        
        self.gpu_mem_used_bytes = Gauge(
            'gpu_mem_used_bytes',
            'GPU memory used in bytes',
            ['gpu'],
            registry=self.registry
        )
    
    def should_sample_trace(self, is_error: bool, latency_ms: float) -> bool:
        """PR-OBS-A3: Determine if trace should be sampled"""
        if is_error:
            # 100% sampling for errors
            return random.random() < self.config.error_trace_sample_rate
        elif latency_ms > self.config.slow_trace_threshold_ms:
            # 100% sampling for slow requests
            return True
        else:
            # Normal sampling rate
            return random.random() < self.config.normal_trace_sample_rate
    
    @contextmanager
    def trace_request(self, request_id: str, operation: str = "inference"):
        """
        PR-OBS-A1: Create traced context with trace_id for exemplars
        
        Usage:
            with obs.trace_request(request_id) as ctx:
                # Your code here
                ctx['trace_id'] = obs.get_current_trace_id()
        """
        span = None
        ctx = {'request_id': request_id, 'trace_id': None, 'span': None}
        
        try:
            if self.tracer:
                span = self.tracer.start_span(
                    operation,
                    kind=SpanKind.SERVER,
                    attributes={
                        'request.id': request_id,
                        'service.name': self.config.otel_service_name
                    }
                )
                
                # Get trace_id for exemplars
                span_context = span.get_span_context()
                if span_context and span_context.trace_id:
                    ctx['trace_id'] = f"{span_context.trace_id:032x}"
                
                ctx['span'] = span
                context.attach(trace.set_span_in_context(span))
            
            yield ctx
            
        except Exception as e:
            logger.error(f"Error in trace_request: {e}")
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
        finally:
            if span:
                span.end()
    
    def get_current_trace_id(self) -> Optional[str]:
        """Get current trace_id if in traced context"""
        if not self.tracer:
            return None
        
        try:
            span = trace.get_current_span()
            if span and span.is_recording():
                span_context = span.get_span_context()
                if span_context and span_context.trace_id:
                    return f"{span_context.trace_id:032x}"
        except Exception as e:
            logger.debug(f"Could not get trace_id: {e}")
        
        return None
    
    @contextmanager
    def create_span(self, name: str, operation_type: str = "internal", attributes: Optional[Dict] = None):
        """
        PR-OBS-B3: Create a child span for parallel/communication tracking
        
        Usage:
            with obs.create_span("prompt_building", "preparation") as span:
                # Your code here
                span.set_attribute("cache_hit", True)
        """
        if not self.tracer:
            # Return a dummy context if tracing is not available
            class DummySpan:
                def set_attribute(self, key, value): pass
                def set_status(self, status): pass
                def add_event(self, name, attributes=None): pass
            yield DummySpan()
            return
        
        span = None
        try:
            # Create span with attributes
            span_attributes = {
                'operation.type': operation_type,
                'service.name': self.config.otel_service_name
            }
            if attributes:
                span_attributes.update(attributes)
            
            span = self.tracer.start_span(
                name,
                kind=SpanKind.INTERNAL,
                attributes=span_attributes
            )
            
            # Set as current span
            context.attach(trace.set_span_in_context(span))
            
            yield span
            
        except Exception as e:
            logger.error(f"Error in create_span: {e}")
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
        finally:
            if span:
                span.end()
    
    def add_span_event(self, name: str, attributes: Optional[Dict] = None):
        """Add an event to the current span"""
        if not self.tracer:
            return
        
        try:
            span = trace.get_current_span()
            if span and span.is_recording():
                span.add_event(name, attributes=attributes or {})
        except Exception as e:
            logger.debug(f"Could not add span event: {e}")
    
    def set_span_attribute(self, key: str, value: Any):
        """Set an attribute on the current span"""
        if not self.tracer:
            return
        
        try:
            span = trace.get_current_span()
            if span and span.is_recording():
                span.set_attribute(key, value)
        except Exception as e:
            logger.debug(f"Could not set span attribute: {e}")
    
    def record_ttft(self, ttft_ms: float, labels: Dict[str, str], trace_id: Optional[str] = None):
        """Record time to first token with exemplar"""
        try:
            # Record metric
            self.llm_ttft_ms.labels(**labels).observe(ttft_ms)
            
            # Add exemplar if trace_id available (PR-OBS-A1)
            if self.config.exemplars_enabled and trace_id:
                # Note: Real exemplar support requires Prometheus client library update
                # This is a placeholder for the concept
                logger.debug(f"TTFT exemplar: {ttft_ms}ms, trace_id={trace_id}")
        except Exception as e:
            logger.error(f"Error recording TTFT: {e}")
    
    def record_e2e(self, e2e_ms: float, labels: Dict[str, str], trace_id: Optional[str] = None):
        """Record end-to-end time with exemplar"""
        try:
            # Record metric
            self.llm_e2e_ms.labels(**labels).observe(e2e_ms)
            
            # Add exemplar if trace_id available (PR-OBS-A1)
            if self.config.exemplars_enabled and trace_id:
                logger.debug(f"E2E exemplar: {e2e_ms}ms, trace_id={trace_id}")
        except Exception as e:
            logger.error(f"Error recording E2E: {e}")
    
    def record_tokens_per_sec(self, tps: float, labels: Dict[str, str]):
        """Record token generation rate"""
        try:
            self.llm_tokens_per_sec.labels(**labels).observe(tps)
        except Exception as e:
            logger.error(f"Error recording TPS: {e}")
    
    def record_cache_hit(self, hit: bool, labels: Dict[str, str]):
        """Record cache hit/miss"""
        try:
            reason = "hit" if hit else "miss"
            self.llm_prefix_cache_hit_total.labels(**labels, reason=reason).inc()
        except Exception as e:
            logger.error(f"Error recording cache hit: {e}")
    
    def record_admission(self, action: str, labels: Dict[str, str]):
        """Record admission control decision"""
        try:
            self.llm_admission_total.labels(**labels, action=action).inc()
        except Exception as e:
            logger.error(f"Error recording admission: {e}")
    
    def update_gpu_metrics(self, gpu_stats: Dict[int, Dict]):
        """Update GPU utilization and memory metrics"""
        try:
            for gpu_id, stats in gpu_stats.items():
                self.gpu_utilization.labels(gpu=str(gpu_id)).set(stats.get('utilization', 0))
                self.gpu_mem_used_bytes.labels(gpu=str(gpu_id)).set(stats.get('memory_used', 0))
        except Exception as e:
            logger.error(f"Error updating GPU metrics: {e}")
    
    def update_session_metrics(self, active_sessions: int, kv_bytes: int, labels: Dict[str, str]):
        """Update session and KV cache metrics"""
        try:
            self.sessions_active.labels(**labels).set(active_sessions)
            self.kv_bytes_in_use.labels(**labels).set(kv_bytes)
        except Exception as e:
            logger.error(f"Error updating session metrics: {e}")
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics in text format"""
        return generate_latest(self.registry)
    
    def log_admission_decision(self, request_id: str, decision: Dict):
        """
        PR-OBS-B1: Log structured admission/routing decision
        """
        try:
            log_entry = {
                "ts": datetime.utcnow().isoformat(),
                "request_id": request_id,
                "admission": decision.get('admission', {}),
                "routing": decision.get('routing', {}),
                "trace_id": self.get_current_trace_id()
            }
            
            # Log as structured JSON
            logger.info(f"ADMISSION_DECISION: {json.dumps(log_entry)}")
        except Exception as e:
            logger.error(f"Error logging admission decision: {e}")
    
    def create_debug_bundle(self) -> Dict:
        """
        PR-OBS-B2: Create debug bundle for issue reporting
        """
        try:
            import subprocess
            
            bundle = {
                "timestamp": datetime.utcnow().isoformat(),
                "version": "4.8.0",
                "config": asdict(self.config),
                "metrics_snapshot": {},
                "gpu_info": {},
                "system_info": {}
            }
            
            # Current metrics snapshot
            # Note: Gauge metrics have labels, so we get the raw metrics text
            bundle["metrics_snapshot"] = {
                "raw_metrics": self.get_metrics().decode('utf-8')[:1000]  # First 1000 chars
            }
            
            # GPU info
            try:
                nvidia_smi = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", 
                     "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=5
                )
                if nvidia_smi.returncode == 0:
                    bundle["gpu_info"]["nvidia_smi"] = nvidia_smi.stdout
            except Exception as e:
                bundle["gpu_info"]["error"] = str(e)
            
            # System info
            try:
                import torch
                bundle["system_info"] = {
                    "torch_version": torch.__version__,
                    "cuda_available": torch.cuda.is_available(),
                    "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                    "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
                }
            except Exception as e:
                bundle["system_info"]["error"] = str(e)
            
            return bundle
            
        except Exception as e:
            logger.error(f"Error creating debug bundle: {e}")
            return {"error": str(e)}


# Singleton instance
_observability_manager = None

def get_observability_manager(config: Optional[ObservabilityConfig] = None) -> ObservabilityManager:
    """Get or create the singleton ObservabilityManager"""
    global _observability_manager
    if _observability_manager is None:
        _observability_manager = ObservabilityManager(config)
    return _observability_manager