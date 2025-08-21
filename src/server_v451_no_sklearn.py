#!/usr/bin/env python3
"""
GPT-OSS HuggingFace Server v4.5.1 - No sklearn version
Bypasses sklearn import issue with NumPy 2.x
"""

import os
import sys
import logging

# Setup logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Monkey patch to skip sklearn import in transformers
import builtins
_original_import = builtins.__import__

def _custom_import(name, *args, **kwargs):
    if 'sklearn' in name or 'scipy.sparse' in name:
        logger.warning(f"Skipping import: {name}")
        # Return a dummy module for sklearn
        class DummyModule:
            def __getattr__(self, item):
                return lambda *args, **kwargs: None
        return DummyModule()
    return _original_import(name, *args, **kwargs)

# Apply the patch temporarily
builtins.__import__ = _custom_import

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
    logger.info("Transformers loaded successfully with sklearn bypass")
except Exception as e:
    logger.error(f"Failed to load transformers: {e}")
    TRANSFORMERS_AVAILABLE = False

# Restore original import
builtins.__import__ = _original_import

# Now import the rest normally
import asyncio
import json
import time
import argparse
import torch
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import asynccontextmanager
from dataclasses import dataclass
import uuid
from enum import Enum
from threading import Lock
import hashlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Rest of the server code from server_v451_clean.py...
# (Copy the rest of the code from server_v451_clean.py starting from Profile enum)

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

# Simple test
if __name__ == "__main__":
    print(f"NumPy version: {np.__version__}")
    print(f"Transformers available: {TRANSFORMERS_AVAILABLE}")
    
    if TRANSFORMERS_AVAILABLE:
        print("Testing model load...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("gpt2")  # Test with small model
            print("✅ Tokenizer loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load tokenizer: {e}")