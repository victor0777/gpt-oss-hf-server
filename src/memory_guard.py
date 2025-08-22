#!/usr/bin/env python3
"""
Memory Guard Module for GPT-OSS HF Server v4.6.0
Implements PR-MEM01, PR-MEM02, PR-MEM03
"""

import torch
import psutil
import time
import logging
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field
from collections import OrderedDict
from threading import Lock
import hashlib
import json

logger = logging.getLogger(__name__)

@dataclass
class MemoryConfig:
    """Memory management configuration - PR-SESSION02 enhanced"""
    gpu_memory_threshold: float = 0.85  # Max GPU utilization
    mem_safety_reserve_mb: int = 2048   # Safety buffer in MB
    session_kv_limit_mb: int = 512      # Per-session KV cache limit
    max_kv_gb: float = 20.0              # Global KV cache limit
    idle_timeout_seconds: int = 180      # PR-SESSION02: Reduced from 300 to 180
    
    # Large request thresholds
    large_req_tokens: int = 8000
    large_req_kv_mb: int = 6000
    
    # Dynamic degradation thresholds
    pressure_low: float = 0.70
    pressure_medium: float = 0.80
    pressure_high: float = 0.85
    pressure_critical: float = 0.90

@dataclass
class SessionMemory:
    """Track memory usage per session"""
    session_id: str
    created_at: float
    last_access: float
    kv_cache_bytes: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    request_count: int = 0

@dataclass
class MemoryEstimate:
    """Memory estimation result"""
    kv_cache_bytes: int
    activation_bytes: int
    total_bytes: int
    gpu_free_bytes: int
    can_proceed: bool
    action: str  # accept, shrink, route4gpu, reject
    reduced_tokens: Optional[int] = None
    reason: Optional[str] = None

class MemoryGuard:
    """
    PR-MEM01: Pre-admission memory estimation and control
    PR-MEM02: Session-based KV cache management with LRU eviction
    PR-MEM03: Dynamic degradation based on memory pressure
    """
    
    def __init__(self, config: MemoryConfig, model_config: dict):
        self.config = config
        self.model_config = model_config
        
        # Model parameters for KV cache calculation
        self.num_layers = model_config.get('num_hidden_layers', 48)
        self.num_kv_heads = model_config.get('num_key_value_heads', 
                                           model_config.get('num_attention_heads', 64))
        self.head_dim = model_config.get('hidden_size', 6144) // model_config.get('num_attention_heads', 64)
        self.dtype_bytes = 2 if 'float16' in str(model_config.get('torch_dtype', 'float16')) else 4
        
        # Session management
        self.sessions: OrderedDict[str, SessionMemory] = OrderedDict()
        self.lock = Lock()
        
        # Global memory tracking
        self.total_kv_bytes = 0
        self.last_cleanup = time.time()
        self.sessions_evicted_total = 0  # PR-SESSION02: Track evictions
        
        logger.info(f"MemoryGuard initialized with config: {config}")
        logger.info(f"Model params: layers={self.num_layers}, kv_heads={self.num_kv_heads}, "
                   f"head_dim={self.head_dim}, dtype_bytes={self.dtype_bytes}")
    
    def estimate_memory(self, 
                       input_tokens: int, 
                       max_new_tokens: int,
                       batch_size: int = 1) -> MemoryEstimate:
        """
        PR-MEM01: Estimate memory requirements for a request
        
        Formula: kv_bytes ≈ tokens * num_layers * (num_kv_heads * head_dim) * 2(K&V) * dtype_bytes
        """
        total_tokens = input_tokens + max_new_tokens
        
        # KV cache memory estimation
        kv_cache_bytes = (
            total_tokens * 
            self.num_layers * 
            (self.num_kv_heads * self.head_dim) * 
            2 *  # K and V
            self.dtype_bytes *
            batch_size
        )
        
        # Activation memory (rough estimate)
        activation_bytes = (
            input_tokens *
            self.model_config.get('hidden_size', 6144) *
            self.dtype_bytes *
            batch_size *
            4  # Multiple intermediate tensors
        )
        
        total_bytes = kv_cache_bytes + activation_bytes
        
        # Get current GPU memory status
        gpu_free_bytes = self._get_gpu_free_memory()
        safety_reserve = self.config.mem_safety_reserve_mb * 1024 * 1024
        
        # Admission decision
        available_bytes = gpu_free_bytes - safety_reserve
        can_proceed = total_bytes < available_bytes * self.config.gpu_memory_threshold
        
        # Determine action
        if can_proceed:
            action = "accept"
            reason = "Sufficient memory available"
        elif input_tokens > self.config.large_req_tokens:
            action = "route4gpu"
            reason = f"Large request ({input_tokens} tokens) - route to 4-GPU"
        elif total_bytes < available_bytes:
            action = "shrink"
            # Calculate reduced tokens
            reduced_ratio = available_bytes / total_bytes * 0.9  # 90% to be safe
            reduced_tokens = int(max_new_tokens * reduced_ratio)
            reason = f"Reduce max_tokens from {max_new_tokens} to {reduced_tokens}"
        else:
            action = "reject"
            reason = "Insufficient memory even with reduction"
        
        return MemoryEstimate(
            kv_cache_bytes=kv_cache_bytes,
            activation_bytes=activation_bytes,
            total_bytes=total_bytes,
            gpu_free_bytes=gpu_free_bytes,
            can_proceed=can_proceed,
            action=action,
            reduced_tokens=reduced_tokens if action == "shrink" else None,
            reason=reason
        )
    
    def register_session(self, session_id: str) -> bool:
        """Register a new session"""
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = SessionMemory(
                    session_id=session_id,
                    created_at=time.time(),
                    last_access=time.time()
                )
                # Move to end (most recent)
                self.sessions.move_to_end(session_id)
                return True
            return False
    
    def update_session(self, 
                      session_id: str, 
                      input_tokens: int,
                      output_tokens: int,
                      kv_cache_bytes: int) -> bool:
        """
        PR-MEM02: Update session memory usage and enforce limits
        """
        with self.lock:
            if session_id not in self.sessions:
                self.register_session(session_id)
            
            session = self.sessions[session_id]
            session.last_access = time.time()
            session.input_tokens += input_tokens
            session.output_tokens += output_tokens
            session.kv_cache_bytes = kv_cache_bytes
            session.request_count += 1
            
            # Move to end (LRU)
            self.sessions.move_to_end(session_id)
            
            # Check session limit
            session_limit_bytes = self.config.session_kv_limit_mb * 1024 * 1024
            if session.kv_cache_bytes > session_limit_bytes:
                logger.warning(f"Session {session_id} exceeds KV limit: "
                             f"{session.kv_cache_bytes / 1024 / 1024:.1f}MB")
                return False
            
            # Update global tracking
            self._update_global_memory()
            
            # Trigger cleanup if needed
            if self._should_cleanup():
                self._cleanup_sessions()
            
            return True
    
    def get_degradation_params(self) -> Dict:
        """
        PR-MEM03: Get dynamic degradation parameters based on memory pressure
        """
        gpu_usage = self._get_gpu_memory_usage()
        
        params = {
            'max_new_tokens': None,
            'temperature': None,
            'top_p': None,
            'batch_size': None,
            'do_sample': None
        }
        
        if gpu_usage < self.config.pressure_low:
            # No degradation needed
            return params
        
        elif gpu_usage < self.config.pressure_medium:
            # Light degradation
            params['max_new_tokens'] = 0.9  # Reduce by 10%
            params['top_p'] = 0.9
            
        elif gpu_usage < self.config.pressure_high:
            # Medium degradation
            params['max_new_tokens'] = 0.7  # Reduce by 30%
            params['temperature'] = 0.5
            params['top_p'] = 0.7
            params['batch_size'] = 0.5  # Halve batch size
            
        elif gpu_usage < self.config.pressure_critical:
            # Heavy degradation
            params['max_new_tokens'] = 0.5  # Reduce by 50%
            params['temperature'] = 0.0  # Greedy
            params['do_sample'] = False
            params['batch_size'] = 0.25  # Quarter batch size
            
        else:
            # Critical - maximum degradation
            params['max_new_tokens'] = 0.3  # Minimum viable
            params['temperature'] = 0.0
            params['do_sample'] = False
            params['batch_size'] = 1  # Single request only
        
        logger.info(f"Memory pressure {gpu_usage:.1%}, degradation params: {params}")
        return params
    
    def should_route_to_4gpu(self, input_tokens: int, estimate: MemoryEstimate) -> bool:
        """Determine if request should be routed to 4-GPU setup"""
        return (
            input_tokens > self.config.large_req_tokens or
            estimate.kv_cache_bytes > self.config.large_req_kv_mb * 1024 * 1024 or
            estimate.action == "route4gpu"
        )
    
    def _get_gpu_free_memory(self) -> int:
        """Get free GPU memory in bytes"""
        if torch.cuda.is_available():
            # Get memory info for GPU 0 (primary)
            free, total = torch.cuda.mem_get_info(0)
            return free
        return 0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage as percentage"""
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return 1.0 - (free / total)
        return 0.0
    
    def _update_global_memory(self):
        """Update global memory tracking"""
        self.total_kv_bytes = sum(
            session.kv_cache_bytes 
            for session in self.sessions.values()
        )
    
    def _should_cleanup(self) -> bool:
        """Check if cleanup is needed - PR-SESSION02 enhanced"""
        now = time.time()
        
        # Time-based cleanup (more frequent)
        if now - self.last_cleanup > 30:  # Every 30 seconds (was 60)
            return True
        
        # Memory-based cleanup (more aggressive)
        max_kv_bytes = self.config.max_kv_gb * 1024 * 1024 * 1024
        if self.total_kv_bytes > max_kv_bytes * 0.7:  # At 70% (was 90%)
            return True
        
        # GPU memory pressure cleanup
        gpu_usage = self._get_gpu_memory_usage()
        if gpu_usage > 0.75:  # Cleanup at 75% GPU usage
            return True
        
        return False
    
    def _cleanup_sessions(self):
        """
        PR-MEM02: Clean up idle and LRU sessions
        """
        now = time.time()
        self.last_cleanup = now
        
        sessions_to_remove = []
        
        # Find idle sessions
        for session_id, session in self.sessions.items():
            if now - session.last_access > self.config.idle_timeout_seconds:
                sessions_to_remove.append(session_id)
                logger.info(f"Removing idle session {session_id}")
        
        # Remove idle sessions
        for session_id in sessions_to_remove:
            del self.sessions[session_id]
            self.sessions_evicted_total += 1  # PR-SESSION02: Track evictions
        
        # If still over limit, remove LRU sessions
        max_kv_bytes = self.config.max_kv_gb * 1024 * 1024 * 1024
        while self.total_kv_bytes > max_kv_bytes and self.sessions:
            # Remove oldest (first in OrderedDict)
            session_id, session = self.sessions.popitem(last=False)
            logger.info(f"Evicting LRU session {session_id}")
            self.sessions_evicted_total += 1  # PR-SESSION02: Track evictions
            self._update_global_memory()
    
    def get_stats(self) -> Dict:
        """Get memory guard statistics - PR-SESSION02 & PR-OBS01 enhanced"""
        with self.lock:
            now = time.time()
            
            # Calculate eviction stats
            sessions_evicted = getattr(self, 'sessions_evicted_total', 0)
            
            return {
                'active_sessions': len(self.sessions),
                'sessions_evicted_total': sessions_evicted,
                'total_kv_mb': self.total_kv_bytes / 1024 / 1024,
                'kv_in_use_mb': self.total_kv_bytes / 1024 / 1024,  # PR-SESSION02
                'gpu_usage': f"{self._get_gpu_memory_usage():.1%}",
                'gpu_free_mb': self._get_gpu_free_memory() / 1024 / 1024,
                'idle_timeout_seconds': self.config.idle_timeout_seconds,
                'cleanup_interval_seconds': 30,
                'session_details': [
                    {
                        'id': s.session_id,
                        'kv_mb': s.kv_cache_bytes / 1024 / 1024,
                        'requests': s.request_count,
                        'idle_seconds': now - s.last_access,
                        'will_expire_in': max(0, self.config.idle_timeout_seconds - (now - s.last_access))
                    }
                    for s in list(self.sessions.values())[:10]  # Top 10
                ]
            }


class AdmissionController:
    """
    Admission control system that uses MemoryGuard for decisions
    """
    
    def __init__(self, memory_guard: MemoryGuard):
        self.memory_guard = memory_guard
        self.rejected_count = 0
        self.routed_4gpu_count = 0
        self.degraded_count = 0
        
    def check_admission(self, 
                        request_id: str,
                        input_tokens: int,
                        max_new_tokens: int,
                        batch_size: int = 1) -> Tuple[bool, Dict]:
        """
        Check if request should be admitted and how
        
        Returns: (should_proceed, action_details)
        """
        # Get memory estimate
        estimate = self.memory_guard.estimate_memory(
            input_tokens, max_new_tokens, batch_size
        )
        
        # Get degradation parameters
        degrade_params = self.memory_guard.get_degradation_params()
        
        # Make admission decision
        result = {
            'request_id': request_id,
            'action': estimate.action,
            'reason': estimate.reason,
            'original_max_tokens': max_new_tokens,
            'memory_estimate_mb': estimate.total_bytes / 1024 / 1024,
            'gpu_free_mb': estimate.gpu_free_bytes / 1024 / 1024
        }
        
        if estimate.action == "accept":
            # Apply degradation if needed
            if degrade_params['max_new_tokens']:
                result['max_new_tokens'] = int(max_new_tokens * degrade_params['max_new_tokens'])
                result['degraded'] = True
                self.degraded_count += 1
            else:
                result['max_new_tokens'] = max_new_tokens
                result['degraded'] = False
            
            # Apply other degradation params
            for key, value in degrade_params.items():
                if value is not None and key != 'max_new_tokens':
                    result[key] = value
            
            return True, result
            
        elif estimate.action == "shrink":
            result['max_new_tokens'] = estimate.reduced_tokens
            result['degraded'] = True
            self.degraded_count += 1
            return True, result
            
        elif estimate.action == "route4gpu":
            result['route_to'] = '4gpu'
            self.routed_4gpu_count += 1
            return True, result
            
        else:  # reject
            self.rejected_count += 1
            return False, result
    
    def get_stats(self) -> Dict:
        """Get admission controller statistics"""
        return {
            'rejected_count': self.rejected_count,
            'routed_4gpu_count': self.routed_4gpu_count,
            'degraded_count': self.degraded_count,
            'memory_stats': self.memory_guard.get_stats()
        }