"""
Engine Client Interface and Implementations for v4.5
Provides abstraction layer for different inference engines
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, AsyncIterator, Protocol
from dataclasses import dataclass
import asyncio
import aiohttp
import json
import time
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class EngineType(Enum):
    CUSTOM = "custom"
    VLLM = "vllm"
    TRTLLM = "trtllm"
    AUTO = "auto"

@dataclass
class EngineHealth:
    """Engine health status"""
    status: str  # healthy, degraded, unhealthy
    latency_ms: float
    score: float  # 0.0-1.0
    engine_name: str
    engine_version: str
    gpu_utilization: float
    memory_used_gb: float
    active_requests: int
    queued_requests: int

@dataclass
class GenerateRequest:
    """Unified generation request"""
    messages: List[Dict[str, str]]
    model: str
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    
@dataclass
class GenerateResponse:
    """Unified generation response"""
    id: str
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]
    created: int
    object: str = "chat.completion"

class EngineClient(Protocol):
    """Protocol for engine clients"""
    
    async def generate(
        self, 
        request: GenerateRequest,
        **kwargs
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate completion from messages"""
        ...
    
    async def tokenize(self, text: str) -> List[int]:
        """Tokenize text"""
        ...
    
    async def health(self) -> EngineHealth:
        """Get engine health status"""
        ...
    
    def supports(self, feature: str) -> bool:
        """Check if engine supports a feature"""
        ...

class CustomEngineClient:
    """Adapter for our custom v4.4 engine"""
    
    def __init__(self, endpoint: str = "http://localhost:8000"):
        self.endpoint = endpoint
        self.session = None
        self.features = {
            "streaming": True,
            "tokenization": True,
            "multi_gpu": True,
            "pipeline_parallel": True,
            "tensor_parallel": True,
            "continuous_batching": True,
            "kv_cache": True,
            "prefix_cache": True
        }
        
    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
            
    async def generate(
        self,
        request: GenerateRequest,
        **kwargs
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate using custom engine"""
        await self._ensure_session()
        
        payload = {
            "model": request.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream,
            "stop": request.stop,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty
        }
        
        if request.stream:
            return self._stream_generate(payload)
        else:
            return await self._generate(payload)
    
    async def _generate(self, payload: dict) -> GenerateResponse:
        """Non-streaming generation"""
        async with self.session.post(
            f"{self.endpoint}/v1/chat/completions",
            json=payload
        ) as resp:
            data = await resp.json()
            return GenerateResponse(
                id=data["id"],
                model=data["model"],
                choices=data["choices"],
                usage=data["usage"],
                created=data["created"]
            )
    
    async def _stream_generate(self, payload: dict) -> AsyncIterator[GenerateResponse]:
        """Streaming generation"""
        async with self.session.post(
            f"{self.endpoint}/v1/chat/completions",
            json=payload
        ) as resp:
            async for line in resp.content:
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        data = json.loads(data_str)
                        yield GenerateResponse(
                            id=data["id"],
                            model=data["model"],
                            choices=data["choices"],
                            usage=data.get("usage", {}),
                            created=data["created"]
                        )
    
    async def tokenize(self, text: str) -> List[int]:
        """Tokenize using custom engine"""
        # Custom engine doesn't expose tokenization endpoint
        # Return approximate token count
        return list(range(len(text.split()) * 2))  # Rough approximation
    
    async def health(self) -> EngineHealth:
        """Get custom engine health"""
        await self._ensure_session()
        
        start = time.time()
        try:
            async with self.session.get(f"{self.endpoint}/health") as resp:
                latency_ms = (time.time() - start) * 1000
                data = await resp.json()
                
                # Calculate health score
                score = 1.0
                if data.get("queue_length", 0) > 10:
                    score -= 0.2
                if data.get("error_rate", 0) > 0.01:
                    score -= 0.3
                if latency_ms > 500:
                    score -= 0.1
                    
                return EngineHealth(
                    status="healthy" if score > 0.7 else "degraded" if score > 0.3 else "unhealthy",
                    latency_ms=latency_ms,
                    score=max(0.0, score),
                    engine_name="custom",
                    engine_version="4.4",
                    gpu_utilization=data.get("gpu_utilization", 0),
                    memory_used_gb=data.get("memory_used_gb", 0),
                    active_requests=data.get("active_requests", 0),
                    queued_requests=data.get("queue_length", 0)
                )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return EngineHealth(
                status="unhealthy",
                latency_ms=999999,
                score=0.0,
                engine_name="custom",
                engine_version="4.4",
                gpu_utilization=0,
                memory_used_gb=0,
                active_requests=0,
                queued_requests=0
            )
    
    def supports(self, feature: str) -> bool:
        """Check feature support"""
        return self.features.get(feature, False)

class VllmEngineClient:
    """Adapter for vLLM engine"""
    
    def __init__(self, endpoint: str = "http://localhost:8001"):
        self.endpoint = endpoint
        self.session = None
        self.features = {
            "streaming": True,
            "tokenization": True,
            "multi_gpu": True,
            "pipeline_parallel": False,  # vLLM uses different parallelism
            "tensor_parallel": True,
            "continuous_batching": True,  # vLLM native
            "kv_cache": True,  # PagedAttention
            "prefix_cache": True,  # RadixAttention
            "speculative_decoding": True,
            "guided_decoding": True
        }
    
    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def generate(
        self,
        request: GenerateRequest,
        **kwargs
    ) -> GenerateResponse | AsyncIterator[GenerateResponse]:
        """Generate using vLLM engine"""
        await self._ensure_session()
        
        # vLLM OpenAI-compatible format
        payload = {
            "model": request.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": request.stream,
            "stop": request.stop,
            "presence_penalty": request.presence_penalty,
            "frequency_penalty": request.frequency_penalty,
            # vLLM specific optimizations
            "use_beam_search": False,
            "best_of": 1,
            "logprobs": None
        }
        
        if request.stream:
            return self._stream_generate(payload)
        else:
            return await self._generate(payload)
    
    async def _generate(self, payload: dict) -> GenerateResponse:
        """Non-streaming vLLM generation"""
        async with self.session.post(
            f"{self.endpoint}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            data = await resp.json()
            
            # vLLM returns OpenAI-compatible format
            return GenerateResponse(
                id=data["id"],
                model=data["model"],
                choices=data["choices"],
                usage=data["usage"],
                created=data["created"]
            )
    
    async def _stream_generate(self, payload: dict) -> AsyncIterator[GenerateResponse]:
        """Streaming vLLM generation"""
        async with self.session.post(
            f"{self.endpoint}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120)
        ) as resp:
            async for line in resp.content:
                if line:
                    line = line.decode('utf-8').strip()
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        data = json.loads(data_str)
                        yield GenerateResponse(
                            id=data["id"],
                            model=data["model"],
                            choices=data["choices"],
                            usage=data.get("usage", {}),
                            created=data["created"]
                        )
    
    async def tokenize(self, text: str) -> List[int]:
        """Tokenize using vLLM"""
        await self._ensure_session()
        
        # vLLM tokenization endpoint
        async with self.session.post(
            f"{self.endpoint}/tokenize",
            json={"text": text, "model": "gpt-oss-20b"}
        ) as resp:
            data = await resp.json()
            return data["tokens"]
    
    async def health(self) -> EngineHealth:
        """Get vLLM health status"""
        await self._ensure_session()
        
        start = time.time()
        try:
            # vLLM health endpoint
            async with self.session.get(f"{self.endpoint}/health") as resp:
                latency_ms = (time.time() - start) * 1000
                
                # vLLM metrics endpoint for detailed stats
                async with self.session.get(f"{self.endpoint}/metrics") as metrics_resp:
                    metrics = await metrics_resp.text()
                    
                    # Parse Prometheus metrics
                    gpu_util = 0.0
                    memory_gb = 0.0
                    active_reqs = 0
                    queued_reqs = 0
                    
                    for line in metrics.split('\n'):
                        if 'vllm:gpu_cache_usage_perc' in line:
                            gpu_util = float(line.split()[-1])
                        elif 'vllm:num_requests_running' in line:
                            active_reqs = int(line.split()[-1])
                        elif 'vllm:num_requests_waiting' in line:
                            queued_reqs = int(line.split()[-1])
                    
                    # Calculate health score
                    score = 1.0
                    if queued_reqs > 20:
                        score -= 0.3
                    if gpu_util > 0.9:
                        score -= 0.2
                    if latency_ms > 300:
                        score -= 0.1
                    
                    return EngineHealth(
                        status="healthy" if score > 0.7 else "degraded" if score > 0.3 else "unhealthy",
                        latency_ms=latency_ms,
                        score=max(0.0, score),
                        engine_name="vllm",
                        engine_version="0.5.0",
                        gpu_utilization=gpu_util,
                        memory_used_gb=memory_gb,
                        active_requests=active_reqs,
                        queued_requests=queued_reqs
                    )
        except Exception as e:
            logger.error(f"vLLM health check failed: {e}")
            return EngineHealth(
                status="unhealthy",
                latency_ms=999999,
                score=0.0,
                engine_name="vllm",
                engine_version="unknown",
                gpu_utilization=0,
                memory_used_gb=0,
                active_requests=0,
                queued_requests=0
            )
    
    def supports(self, feature: str) -> bool:
        """Check vLLM feature support"""
        return self.features.get(feature, False)

class TrtLlmEngineClient:
    """Stub adapter for TensorRT-LLM engine"""
    
    def __init__(self, endpoint: str = "http://localhost:8002"):
        self.endpoint = endpoint
        self.session = None
        self.features = {
            "streaming": True,
            "tokenization": True,
            "multi_gpu": True,
            "pipeline_parallel": True,
            "tensor_parallel": True,
            "continuous_batching": True,
            "kv_cache": True,
            "prefix_cache": False,
            "int8_quantization": True,
            "fp8_quantization": True,
            "inflight_batching": True
        }
    
    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def generate(
        self,
        request: GenerateRequest,
        **kwargs
    ) -> GenerateResponse:
        """TRT-LLM generation (stub for now)"""
        # Stub implementation - forward to vLLM for testing
        vllm_client = VllmEngineClient(self.endpoint)
        return await vllm_client.generate(request, **kwargs)
    
    async def tokenize(self, text: str) -> List[int]:
        """TRT-LLM tokenization (stub)"""
        # Approximate for now
        return list(range(len(text.split()) * 2))
    
    async def health(self) -> EngineHealth:
        """TRT-LLM health (stub)"""
        return EngineHealth(
            status="healthy",
            latency_ms=50.0,  # TRT-LLM is typically very fast
            score=0.9,
            engine_name="trtllm",
            engine_version="stub",
            gpu_utilization=0.5,
            memory_used_gb=10.0,
            active_requests=0,
            queued_requests=0
        )
    
    def supports(self, feature: str) -> bool:
        """Check TRT-LLM feature support"""
        return self.features.get(feature, False)

class EngineRouter:
    """Intelligent routing between engines"""
    
    def __init__(self):
        self.engines = {}
        self.health_cache = {}
        self.health_cache_ttl = 5.0  # seconds
        
    def register_engine(self, name: str, client: EngineClient):
        """Register an engine"""
        self.engines[name] = client
        
    async def select_engine(
        self,
        request: GenerateRequest,
        engine_type: EngineType = EngineType.AUTO
    ) -> tuple[str, EngineClient]:
        """Select best engine for request"""
        
        if engine_type != EngineType.AUTO:
            # Direct engine selection
            engine_name = engine_type.value
            if engine_name in self.engines:
                return engine_name, self.engines[engine_name]
            else:
                # Fallback to custom
                return "custom", self.engines["custom"]
        
        # Auto routing logic
        input_tokens = sum(len(msg["content"].split()) for msg in request.messages) * 2
        model_size = "120b" if "120b" in request.model else "20b"
        
        # Get health scores
        healths = {}
        for name, engine in self.engines.items():
            cache_key = f"{name}_health"
            if cache_key in self.health_cache:
                cached_time, cached_health = self.health_cache[cache_key]
                if time.time() - cached_time < self.health_cache_ttl:
                    healths[name] = cached_health
                    continue
            
            try:
                health = await engine.health()
                self.health_cache[cache_key] = (time.time(), health)
                healths[name] = health
            except:
                healths[name] = EngineHealth(
                    status="unhealthy", latency_ms=999999, score=0.0,
                    engine_name=name, engine_version="unknown",
                    gpu_utilization=0, memory_used_gb=0,
                    active_requests=0, queued_requests=0
                )
        
        # Routing rules
        candidates = []
        
        # Rule 1: Long context (>8k tokens) or 120b model -> prefer vLLM/TRT-LLM
        if input_tokens > 8000 or model_size == "120b":
            if "vllm" in healths and healths["vllm"].score > 0.7:
                candidates.append(("vllm", 0.9))
            if "trtllm" in healths and healths["trtllm"].score > 0.7:
                candidates.append(("trtllm", 0.95))
        
        # Rule 2: Avoid unhealthy engines (score < 0.7)
        for name, health in healths.items():
            if health.score >= 0.7:
                if name == "custom":
                    candidates.append((name, 0.8))
                elif name not in [c[0] for c in candidates]:
                    candidates.append((name, health.score))
        
        # Select best candidate
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            selected = candidates[0][0]
            logger.info(f"Auto-routing selected: {selected} (score: {candidates[0][1]:.2f})")
            return selected, self.engines[selected]
        
        # Fallback to custom
        return "custom", self.engines.get("custom")

# Factory function
def create_engine_client(engine_type: str, endpoint: str = None) -> EngineClient:
    """Create engine client by type"""
    if engine_type == "custom":
        return CustomEngineClient(endpoint or "http://localhost:8000")
    elif engine_type == "vllm":
        return VllmEngineClient(endpoint or "http://localhost:8001")
    elif engine_type == "trtllm":
        return TrtLlmEngineClient(endpoint or "http://localhost:8002")
    else:
        raise ValueError(f"Unknown engine type: {engine_type}")