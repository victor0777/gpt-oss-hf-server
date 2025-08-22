#!/usr/bin/env python3
"""
GPU Router Module for GPT-OSS HF Server v4.7.0 - Version 2
Implements PR-MG01: Large-Path Auto Routing
Works with existing GPU distribution mechanism
"""

import os
import torch
import logging
from typing import Dict, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import subprocess

logger = logging.getLogger(__name__)

class RoutingMode(Enum):
    """GPU routing modes"""
    SINGLE = "single"      # Single GPU (force single GPU mode)
    AUTO = "auto"          # Auto distribution (HuggingFace device_map="auto")
    TENSOR = "tensor"      # Tensor parallelism across GPUs
    PIPELINE = "pipeline"  # Pipeline parallelism
    
@dataclass
class RoutingConfig:
    """Configuration for GPU routing - PR-MG01"""
    # Routing thresholds
    large_input_tokens: int = 8000        # Trigger: input_tokens > 8k
    large_kv_mb: int = 6000               # Trigger: predicted_kv_mb > 6000
    
    # NCCL settings for multi-GPU
    nccl_async_error_handling: int = 1
    nccl_min_nchannels: int = 4
    nccl_buffsize: str = "8388608"       # 8MB in bytes
    
    # Micro-batching for pipeline parallelism
    micro_batches: int = 6
    
    # GPU utilization targets
    min_gpu_util_percent: float = 60.0    # Each GPU should be ≥60%
    
    # Performance targets
    max_p95_e2e_seconds: float = 20.0     # P95 E2E ≤ 20s
    
@dataclass
class RoutingDecision:
    """Result of routing decision"""
    mode: RoutingMode
    reason: str
    gpu_config: Dict[str, Any]
    estimated_memory_mb: int
    admission_action: str  # "accept", "route4", "reject"
    should_use_multi_gpu: bool  # Whether to use multi-GPU mode
    
class GPURouter:
    """
    PR-MG01: Intelligent GPU routing for large requests
    Works with existing HuggingFace device_map mechanism
    """
    
    def __init__(self, config: Optional[RoutingConfig] = None):
        self.config = config or RoutingConfig()
        self.routing_stats = {
            "total_requests": 0,
            "single_gpu": 0,
            "route4_gpu": 0,
            "route4_triggers": {
                "large_input": 0,
                "large_kv": 0,
                "memory_pressure": 0
            }
        }
        
        # Configure NCCL environment for multi-GPU mode
        self._configure_nccl()
        
        # Check current GPU configuration
        self.gpu_count = torch.cuda.device_count()
        self.multi_gpu_available = self.gpu_count >= 4
        
        logger.info(f"GPURouter initialized: {self.gpu_count} GPUs available")
        logger.info(f"Multi-GPU routing: {'enabled' if self.multi_gpu_available else 'disabled'}")
        
    def _configure_nccl(self):
        """Configure NCCL environment variables for multi-GPU routing"""
        if torch.cuda.device_count() >= 4:
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(self.config.nccl_async_error_handling)
            os.environ["NCCL_MIN_NCHANNELS"] = str(self.config.nccl_min_nchannels)
            os.environ["NCCL_BUFFSIZE"] = self.config.nccl_buffsize
            os.environ["NCCL_P2P_DISABLE"] = "0"  # Enable P2P for better performance
            
            logger.info("NCCL configured for multi-GPU routing")
    
    def should_route_to_multi_gpu(
        self, 
        input_tokens: int,
        max_new_tokens: int,
        current_memory_pressure: float = 0.0,
        current_gpu_mode: str = "auto"
    ) -> RoutingDecision:
        """
        Determine if request should use multi-GPU setup
        
        This works with the existing device_map mechanism:
        - If model is already in "auto" mode, large requests are handled naturally
        - We just track statistics and provide recommendations
        
        Args:
            input_tokens: Number of input tokens
            max_new_tokens: Maximum tokens to generate
            current_memory_pressure: Current GPU memory usage (0.0-1.0)
            current_gpu_mode: Current GPU mode of the model
            
        Returns:
            RoutingDecision with mode and configuration
        """
        self.routing_stats["total_requests"] += 1
        
        # Calculate estimated KV cache size
        total_tokens = input_tokens + max_new_tokens
        estimated_kv_mb = self._estimate_kv_cache_mb(total_tokens)
        
        # Check routing triggers
        trigger_reasons = []
        should_use_multi = False
        
        # Trigger 1: Large input tokens
        if input_tokens > self.config.large_input_tokens:
            trigger_reasons.append(f"large_input ({input_tokens} > {self.config.large_input_tokens})")
            self.routing_stats["route4_triggers"]["large_input"] += 1
            should_use_multi = True
            
        # Trigger 2: Large KV cache prediction
        if estimated_kv_mb > self.config.large_kv_mb:
            trigger_reasons.append(f"large_kv ({estimated_kv_mb}MB > {self.config.large_kv_mb}MB)")
            self.routing_stats["route4_triggers"]["large_kv"] += 1
            should_use_multi = True
            
        # Trigger 3: High memory pressure on single GPU
        if current_memory_pressure > 0.85 and (input_tokens > 4000 or estimated_kv_mb > 3000):
            trigger_reasons.append(f"memory_pressure ({current_memory_pressure:.1%})")
            self.routing_stats["route4_triggers"]["memory_pressure"] += 1
            should_use_multi = True
        
        # Make routing decision based on current setup
        if should_use_multi and self.multi_gpu_available:
            self.routing_stats["route4_gpu"] += 1
            
            # Check if model is already in multi-GPU mode
            if current_gpu_mode in ["auto", "tensor", "pipeline"]:
                # Model is already distributed, just track the routing
                return RoutingDecision(
                    mode=RoutingMode.AUTO,
                    reason=f"Multi-GPU active: {', '.join(trigger_reasons)}",
                    gpu_config={
                        "current_mode": current_gpu_mode,
                        "gpu_count": self.gpu_count,
                        "distributed": True
                    },
                    estimated_memory_mb=estimated_kv_mb,
                    admission_action="route4",
                    should_use_multi_gpu=True
                )
            else:
                # Model is in single GPU mode but should use multi-GPU
                logger.warning(f"Large request detected but model in single GPU mode: {trigger_reasons}")
                return RoutingDecision(
                    mode=RoutingMode.SINGLE,
                    reason=f"Should use multi-GPU but in single mode: {', '.join(trigger_reasons)}",
                    gpu_config={
                        "current_mode": current_gpu_mode,
                        "recommendation": "restart with --gpu-mode auto or tensor"
                    },
                    estimated_memory_mb=estimated_kv_mb,
                    admission_action="route4",
                    should_use_multi_gpu=True
                )
        else:
            self.routing_stats["single_gpu"] += 1
            
            return RoutingDecision(
                mode=RoutingMode.SINGLE if current_gpu_mode == "single" else RoutingMode.AUTO,
                reason="Normal request: within single GPU limits",
                gpu_config={
                    "current_mode": current_gpu_mode,
                    "gpu_count": self.gpu_count
                },
                estimated_memory_mb=estimated_kv_mb,
                admission_action="accept",
                should_use_multi_gpu=False
            )
    
    def _estimate_kv_cache_mb(self, total_tokens: int) -> int:
        """
        Estimate KV cache size in MB
        
        For 120b model:
        - 36 layers, 8 KV heads, head_dim=45
        - BF16 (2 bytes per value)
        - K and V matrices
        """
        num_layers = 36  # 120b model
        num_kv_heads = 8
        head_dim = 45
        dtype_bytes = 2  # BF16
        
        # KV cache size = tokens * layers * (kv_heads * head_dim) * 2(K&V) * dtype_bytes
        kv_bytes = total_tokens * num_layers * (num_kv_heads * head_dim) * 2 * dtype_bytes
        kv_mb = kv_bytes / (1024 * 1024)
        
        return int(kv_mb)
    
    def check_gpu_distribution(self, model) -> Dict:
        """
        Check how the model is currently distributed across GPUs
        """
        distribution = {}
        
        try:
            # Check if model has device_map attribute (from accelerate)
            if hasattr(model, 'hf_device_map'):
                distribution['device_map'] = model.hf_device_map
                distribution['distributed'] = len(set(model.hf_device_map.values())) > 1
            else:
                distribution['device_map'] = 'single'
                distribution['distributed'] = False
            
            # Check GPU memory usage
            for i in range(self.gpu_count):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                distribution[f'gpu_{i}'] = {
                    'allocated_gb': round(allocated, 2),
                    'reserved_gb': round(reserved, 2)
                }
            
            return distribution
            
        except Exception as e:
            logger.error(f"Error checking GPU distribution: {e}")
            return {'error': str(e)}
    
    def get_gpu_utilization(self) -> Dict[int, float]:
        """
        Get current GPU utilization for all GPUs
        """
        utilization = {}
        
        for i in range(self.gpu_count):
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", f"-i={i}"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    util = float(result.stdout.strip())
                    utilization[i] = util
            except Exception as e:
                logger.warning(f"Failed to get GPU {i} utilization: {e}")
                utilization[i] = 0.0
                
        return utilization
    
    def check_gpu_balance(self) -> Tuple[bool, Dict]:
        """
        Check if GPU utilization is balanced across available GPUs
        """
        utilization = self.get_gpu_utilization()
        
        gpu_utils = [utilization.get(i, 0.0) for i in range(self.gpu_count)]
        min_util = min(gpu_utils) if gpu_utils else 0
        max_util = max(gpu_utils) if gpu_utils else 0
        avg_util = sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0
        
        # For multi-GPU, check if load is reasonably balanced
        # Note: Perfect balance is rare, especially with pipeline parallelism
        is_balanced = True
        if self.gpu_count > 1:
            spread = max_util - min_util
            # Consider balanced if spread is less than 40%
            is_balanced = spread < 40.0
        
        stats = {
            "balanced": is_balanced,
            "gpu_count": self.gpu_count,
            "gpu_utilization": {i: gpu_utils[i] for i in range(self.gpu_count)},
            "min_utilization": min_util,
            "max_utilization": max_util,
            "avg_utilization": avg_util,
            "spread": max_util - min_util
        }
        
        return is_balanced, stats
    
    def get_stats(self) -> Dict:
        """Get routing statistics - PR-OBS01 compatible"""
        gpu_balance = self.check_gpu_balance()
        
        return {
            "total_requests": self.routing_stats["total_requests"],
            "single_gpu_requests": self.routing_stats["single_gpu"],
            "route4_gpu_requests": self.routing_stats["route4_gpu"],
            "route4_percentage": (
                self.routing_stats["route4_gpu"] / self.routing_stats["total_requests"] * 100
                if self.routing_stats["total_requests"] > 0 else 0
            ),
            "route4_triggers": self.routing_stats["route4_triggers"],
            "gpu_balance": gpu_balance[1],
            "config": {
                "large_input_tokens": self.config.large_input_tokens,
                "large_kv_mb": self.config.large_kv_mb,
                "micro_batches": self.config.micro_batches,
                "gpu_count": self.gpu_count,
                "multi_gpu_available": self.multi_gpu_available
            }
        }