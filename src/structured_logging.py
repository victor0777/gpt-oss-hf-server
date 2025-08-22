#!/usr/bin/env python3
"""
Structured Logging Module for GPT-OSS HF Server v4.8.0
Implements PR-OBS-B1: Structured JSON logging for observability
"""

import json
import logging
import sys
import time
from typing import Dict, Any, Optional
from datetime import datetime
from enum import Enum
import traceback

class LogLevel(Enum):
    """Log levels for structured logging"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class EventType(Enum):
    """Event types for structured logging"""
    # Request lifecycle
    REQUEST_START = "request.start"
    REQUEST_END = "request.end"
    REQUEST_CANCEL = "request.cancel"
    
    # Admission control
    ADMISSION_CHECK = "admission.check"
    ADMISSION_ACCEPT = "admission.accept"
    ADMISSION_REJECT = "admission.reject"
    ADMISSION_DEGRADE = "admission.degrade"
    
    # Routing decisions
    ROUTING_DECISION = "routing.decision"
    ROUTING_SINGLE_GPU = "routing.single_gpu"
    ROUTING_MULTI_GPU = "routing.multi_gpu"
    
    # Memory management
    MEMORY_PRESSURE = "memory.pressure"
    MEMORY_EVICTION = "memory.eviction"
    SESSION_REGISTER = "session.register"
    SESSION_UPDATE = "session.update"
    SESSION_EVICT = "session.evict"
    
    # Cache events
    CACHE_HIT = "cache.hit"
    CACHE_MISS = "cache.miss"
    CACHE_EVICT = "cache.evict"
    
    # Model operations
    MODEL_LOAD = "model.load"
    MODEL_GENERATE = "model.generate"
    MODEL_ERROR = "model.error"
    
    # Performance events
    TTFT_RECORDED = "performance.ttft"
    TPS_RECORDED = "performance.tps"
    E2E_RECORDED = "performance.e2e"
    
    # System events
    SYSTEM_START = "system.start"
    SYSTEM_SHUTDOWN = "system.shutdown"
    HEALTH_CHECK = "health.check"
    
    # Error events
    ERROR_GPU_OOM = "error.gpu_oom"
    ERROR_TIMEOUT = "error.timeout"
    ERROR_VALIDATION = "error.validation"
    ERROR_INTERNAL = "error.internal"

class StructuredLogger:
    """
    PR-OBS-B1: Structured JSON logger for observability
    
    Features:
    - JSON-formatted log entries
    - Request correlation with trace_id
    - Consistent field naming
    - Error context preservation
    - Performance metrics inclusion
    """
    
    def __init__(self, name: str = "gpt-oss-server", level: str = "INFO"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level))
        
        # Remove default handlers
        self.logger.handlers = []
        
        # Create JSON formatter
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self._create_formatter())
        self.logger.addHandler(handler)
        
        # Context storage for request correlation
        self._context = {}
    
    def _create_formatter(self):
        """Create a JSON formatter for structured logging"""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_obj = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "level": record.levelname.lower(),
                    "logger": record.name,
                    "message": record.getMessage(),
                }
                
                # Add extra fields if present
                if hasattr(record, 'event_type'):
                    log_obj['event_type'] = record.event_type
                if hasattr(record, 'request_id'):
                    log_obj['request_id'] = record.request_id
                if hasattr(record, 'trace_id'):
                    log_obj['trace_id'] = record.trace_id
                if hasattr(record, 'session_id'):
                    log_obj['session_id'] = record.session_id
                if hasattr(record, 'duration_ms'):
                    log_obj['duration_ms'] = record.duration_ms
                if hasattr(record, 'metadata'):
                    log_obj['metadata'] = record.metadata
                if hasattr(record, 'error'):
                    log_obj['error'] = record.error
                
                # Add exception info if present
                if record.exc_info:
                    log_obj['exception'] = {
                        'type': record.exc_info[0].__name__,
                        'message': str(record.exc_info[1]),
                        'traceback': traceback.format_exception(*record.exc_info)
                    }
                
                return json.dumps(log_obj)
        
        return JSONFormatter()
    
    def set_context(self, **kwargs):
        """Set context that will be included in all subsequent logs"""
        self._context.update(kwargs)
    
    def clear_context(self):
        """Clear the logging context"""
        self._context = {}
    
    def _log(self, level: LogLevel, event_type: EventType, message: str, **kwargs):
        """Internal logging method with structured fields"""
        extra = {
            'event_type': event_type.value,
            **self._context,
            **kwargs
        }
        
        # Handle metadata separately to avoid conflicts
        metadata = extra.pop('metadata', {})
        if metadata:
            extra['metadata'] = metadata
        
        # Handle error information
        error = extra.pop('error', None)
        if error:
            if isinstance(error, Exception):
                extra['error'] = {
                    'type': type(error).__name__,
                    'message': str(error)
                }
            else:
                extra['error'] = error
        
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL
        }
        
        self.logger.log(level_map[level], message, extra=extra)
    
    # Convenience methods for different log levels
    def debug(self, event_type: EventType, message: str, **kwargs):
        self._log(LogLevel.DEBUG, event_type, message, **kwargs)
    
    def info(self, event_type: EventType, message: str, **kwargs):
        self._log(LogLevel.INFO, event_type, message, **kwargs)
    
    def warning(self, event_type: EventType, message: str, **kwargs):
        self._log(LogLevel.WARNING, event_type, message, **kwargs)
    
    def error(self, event_type: EventType, message: str, **kwargs):
        self._log(LogLevel.ERROR, event_type, message, **kwargs)
    
    def critical(self, event_type: EventType, message: str, **kwargs):
        self._log(LogLevel.CRITICAL, event_type, message, **kwargs)
    
    # Specialized logging methods for common events
    def log_request_start(self, request_id: str, trace_id: Optional[str] = None, **metadata):
        """Log request start event"""
        self.info(
            EventType.REQUEST_START,
            f"Request {request_id} started",
            request_id=request_id,
            trace_id=trace_id,
            metadata=metadata
        )
    
    def log_request_end(self, request_id: str, duration_ms: float, success: bool, **metadata):
        """Log request end event"""
        self.info(
            EventType.REQUEST_END,
            f"Request {request_id} completed",
            request_id=request_id,
            duration_ms=duration_ms,
            success=success,
            metadata=metadata
        )
    
    def log_admission_decision(self, request_id: str, action: str, reason: str, **metadata):
        """Log admission control decision"""
        event_map = {
            'accept': EventType.ADMISSION_ACCEPT,
            'reject': EventType.ADMISSION_REJECT,
            'degrade': EventType.ADMISSION_DEGRADE
        }
        event_type = event_map.get(action, EventType.ADMISSION_CHECK)
        
        level = LogLevel.INFO if action == 'accept' else LogLevel.WARNING
        self._log(
            level,
            event_type,
            f"Admission {action} for {request_id}: {reason}",
            request_id=request_id,
            action=action,
            reason=reason,
            metadata=metadata
        )
    
    def log_routing_decision(self, request_id: str, route: str, reason: str, **metadata):
        """Log routing decision"""
        event_type = EventType.ROUTING_MULTI_GPU if 'multi' in route.lower() else EventType.ROUTING_SINGLE_GPU
        
        self.info(
            event_type,
            f"Routing {request_id} to {route}: {reason}",
            request_id=request_id,
            route=route,
            reason=reason,
            metadata=metadata
        )
    
    def log_cache_event(self, request_id: str, hit: bool, cache_key: str, **metadata):
        """Log cache hit/miss event"""
        event_type = EventType.CACHE_HIT if hit else EventType.CACHE_MISS
        
        self.debug(
            event_type,
            f"Cache {'hit' if hit else 'miss'} for {request_id}",
            request_id=request_id,
            cache_key=cache_key,
            metadata=metadata
        )
    
    def log_performance_metric(self, metric_type: str, value: float, request_id: str, **metadata):
        """Log performance metrics"""
        event_map = {
            'ttft': EventType.TTFT_RECORDED,
            'tps': EventType.TPS_RECORDED,
            'e2e': EventType.E2E_RECORDED
        }
        event_type = event_map.get(metric_type, EventType.TTFT_RECORDED)
        
        self.info(
            event_type,
            f"Performance metric {metric_type}={value:.2f} for {request_id}",
            request_id=request_id,
            metric_type=metric_type,
            value=value,
            metadata=metadata
        )
    
    def log_error(self, request_id: str, error: Exception, error_type: str = "internal", **metadata):
        """Log error event with context"""
        event_map = {
            'gpu_oom': EventType.ERROR_GPU_OOM,
            'timeout': EventType.ERROR_TIMEOUT,
            'validation': EventType.ERROR_VALIDATION,
            'internal': EventType.ERROR_INTERNAL
        }
        event_type = event_map.get(error_type, EventType.ERROR_INTERNAL)
        
        self.error(
            event_type,
            f"Error in request {request_id}: {str(error)}",
            request_id=request_id,
            error=error,
            metadata=metadata
        )
    
    def log_memory_pressure(self, pressure: float, action: str, **metadata):
        """Log memory pressure event"""
        level = LogLevel.WARNING if pressure > 80 else LogLevel.INFO
        
        self._log(
            level,
            EventType.MEMORY_PRESSURE,
            f"Memory pressure at {pressure:.1f}%, action: {action}",
            pressure=pressure,
            action=action,
            metadata=metadata
        )
    
    def log_session_event(self, event: str, session_id: str, **metadata):
        """Log session lifecycle events"""
        event_map = {
            'register': EventType.SESSION_REGISTER,
            'update': EventType.SESSION_UPDATE,
            'evict': EventType.SESSION_EVICT
        }
        event_type = event_map.get(event, EventType.SESSION_REGISTER)
        
        self.debug(
            event_type,
            f"Session {event}: {session_id}",
            session_id=session_id,
            metadata=metadata
        )

# Singleton instance
_structured_logger = None

def get_structured_logger(name: str = "gpt-oss-server", level: str = "INFO") -> StructuredLogger:
    """Get or create the singleton StructuredLogger"""
    global _structured_logger
    if _structured_logger is None:
        _structured_logger = StructuredLogger(name, level)
    return _structured_logger