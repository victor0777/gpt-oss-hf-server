#!/usr/bin/env python3
"""
PromptBuilder for GPT-OSS HF Server v4.5.3
Standardized prompt formatting with versioning and caching
"""

import hashlib
import re
import json
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class PromptVersion(Enum):
    """Versioned system prompts"""
    SYS_V1 = "SYS/v1"
    SYS_V2 = "SYS/v2"
    
    @property
    def system_prompt(self) -> str:
        """Get system prompt for version"""
        prompts = {
            "SYS/v1": "You are a helpful AI assistant.",
            "SYS/v2": "You are GPT-OSS, a helpful and knowledgeable AI assistant created by OpenAI."
        }
        return prompts.get(self.value, prompts["SYS/v1"])


@dataclass
class PromptConfig:
    """Configuration for prompt building - PR-PF01 enhanced"""
    prompt_version: PromptVersion = PromptVersion.SYS_V1
    max_input_tokens: int = 4096
    normalize_input: bool = True
    remove_timestamps: bool = True
    remove_uuids: bool = True
    remove_session_ids: bool = True
    remove_paths: bool = True
    remove_machine_names: bool = True
    remove_random_numbers: bool = True
    remove_extra_whitespace: bool = True
    enable_truncation: bool = True
    enable_cache: bool = True
    cache_ttl: int = 600  # PR-CACHE02: Increased from 300
    cache_max_entries: int = 500  # PR-CACHE02: Configurable cache size


@dataclass
class GenerationParams:
    """Generation parameters for deterministic output"""
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    seed: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for hashing"""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "stop_sequences": sorted(self.stop_sequences) if self.stop_sequences else []
        }


class PromptBuilder:
    """
    Unified prompt builder for consistent formatting across models
    """
    
    def __init__(self, tokenizer, config: Optional[PromptConfig] = None):
        """
        Initialize prompt builder
        
        Args:
            tokenizer: HuggingFace tokenizer instance
            config: Optional configuration
        """
        self.tokenizer = tokenizer
        self.config = config or PromptConfig()
        self._cache = {}
        self._metrics = {
            "cache_hits": 0,
            "cache_misses": 0,
            "truncations": 0,
            "total_builds": 0
        }
        
    def build(
        self,
        messages: List[Dict[str, str]],
        params: GenerationParams,
        model_id: str = "gpt-oss"
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Build standardized prompt from messages
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            params: Generation parameters
            model_id: Model identifier
            
        Returns:
            Tuple of (formatted_prompt, metadata)
        """
        self._metrics["total_builds"] += 1
        
        # Normalize messages
        normalized_messages = self._normalize_messages(messages)
        
        # Check cache
        cache_key = self._compute_cache_key(normalized_messages, params, model_id)
        if self.config.enable_cache and cache_key in self._cache:
            self._metrics["cache_hits"] += 1
            cached_prompt, cached_metadata = self._cache[cache_key]
            cached_metadata["cache_hit"] = True
            return cached_prompt, cached_metadata
        
        self._metrics["cache_misses"] += 1
        
        # Add system prompt if not present
        if not normalized_messages or normalized_messages[0]["role"] != "system":
            system_message = {
                "role": "system",
                "content": self.config.prompt_version.system_prompt
            }
            normalized_messages.insert(0, system_message)
        
        # Build prompt with tokenizer
        prompt = self._build_with_tokenizer(normalized_messages)
        
        # Handle truncation if needed
        tokens_before = len(self.tokenizer.encode(prompt))
        if self.config.enable_truncation and tokens_before > self.config.max_input_tokens:
            prompt, truncation_info = self._truncate_prompt(normalized_messages)
            tokens_after = len(self.tokenizer.encode(prompt))
            self._metrics["truncations"] += 1
        else:
            truncation_info = None
            tokens_after = tokens_before
        
        # Create metadata
        metadata = {
            "prompt_version": self.config.prompt_version.value,
            "model_id": model_id,
            "cache_key": cache_key,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "truncated": truncation_info is not None,
            "truncation_info": truncation_info,
            "params": params.to_dict(),
            "cache_hit": False,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Cache the result
        if self.config.enable_cache:
            self._cache[cache_key] = (prompt, metadata.copy())
        
        return prompt, metadata
    
    def _normalize_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Normalize message content - PR-PF01 enhanced"""
        if not self.config.normalize_input:
            return messages
        
        normalized = []
        for msg in messages:
            content = msg["content"]
            
            # Remove timestamps (enhanced)
            if self.config.remove_timestamps:
                content = re.sub(r'\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[\w\.]*', '<timestamp>', content)
                content = re.sub(r'\d{1,2}[:/]\d{1,2}[:/]\d{2,4}', '<date>', content)
                content = re.sub(r'\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|am|pm)?', '<time>', content)
            
            # Remove UUIDs (enhanced)
            if self.config.remove_uuids:
                content = re.sub(
                    r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}',
                    '<uuid>',
                    content
                )
            
            # Remove session IDs (new)
            if self.config.remove_session_ids:
                content = re.sub(r'session[_-]?id[=:]?\s*[\w-]+', '<session_id>', content, flags=re.IGNORECASE)
                content = re.sub(r'sid[=:]?\s*[\w-]+', '<session_id>', content, flags=re.IGNORECASE)
                content = re.sub(r'X-Session-ID[=:]?\s*[\w-]+', '<session_id>', content, flags=re.IGNORECASE)
            
            # Remove file paths (new)
            if self.config.remove_paths:
                content = re.sub(r'[/\\](?:home|users|var|tmp|etc|usr)[/\\][^\s]+', '<path>', content)
                content = re.sub(r'[A-Z]:\\\\[^\s]+', '<path>', content)  # Windows paths
                content = re.sub(r'\./[\w/]+', '<path>', content)  # Relative paths
            
            # Remove machine names (new)
            if self.config.remove_machine_names:
                content = re.sub(r'localhost:\d+', '<host>', content)
                content = re.sub(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<ip>', content)
                content = re.sub(r'[a-zA-Z0-9-]+\.local', '<hostname>', content)
                content = re.sub(r'http[s]?://[^\s]+', '<url>', content)
            
            # Remove random numbers (new)
            if self.config.remove_random_numbers:
                content = re.sub(r'\b[a-f0-9]{32,}\b', '<hash>', content)  # Long hashes
                content = re.sub(r'\b0x[a-f0-9]{8,}\b', '<hex>', content)  # Hex numbers
                content = re.sub(r'request[_-]?id[=:]?\s*[\w-]+', '<request_id>', content, flags=re.IGNORECASE)
            
            # Remove extra whitespace
            if self.config.remove_extra_whitespace:
                content = re.sub(r'\s+', ' ', content)
                content = content.strip()
            
            normalized.append({
                "role": msg["role"],
                "content": content
            })
        
        return normalized
    
    def _build_with_tokenizer(self, messages: List[Dict[str, str]]) -> str:
        """Build prompt using tokenizer's chat template"""
        try:
            # Use tokenizer's built-in chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return prompt
        except Exception as e:
            logger.warning(f"Failed to use chat template: {e}, falling back to manual formatting")
            return self._manual_format(messages)
    
    def _manual_format(self, messages: List[Dict[str, str]]) -> str:
        """Manual fallback formatting"""
        formatted = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
            else:
                formatted.append(f"{role}: {content}")
        
        formatted.append("Assistant:")
        return "\n\n".join(formatted)
    
    def _truncate_prompt(
        self,
        messages: List[Dict[str, str]]
    ) -> Tuple[str, Dict[str, str]]:
        """
        Truncate prompt to fit max tokens
        
        Strategy:
        1. Remove headers/metadata
        2. Compress past turns
        3. Insert summary for truncated content
        """
        truncation_info = {
            "strategy": "progressive",
            "steps": []
        }
        
        # Step 1: Try removing headers
        cleaned_messages = []
        for msg in messages:
            content = msg["content"]
            # Remove markdown headers
            content = re.sub(r'^#+\s+.*$', '', content, flags=re.MULTILINE)
            cleaned_messages.append({
                "role": msg["role"],
                "content": content.strip()
            })
        
        prompt = self._build_with_tokenizer(cleaned_messages)
        if len(self.tokenizer.encode(prompt)) <= self.config.max_input_tokens:
            truncation_info["steps"].append("removed_headers")
            return prompt, truncation_info
        
        # Step 2: Compress past turns (keep system, first user, and last user)
        if len(messages) > 3:
            compressed = [
                messages[0],  # System
                messages[1],  # First user
                {
                    "role": "assistant",
                    "content": "[Previous conversation summarized for context]"
                },
                messages[-1]  # Last user
            ]
            prompt = self._build_with_tokenizer(compressed)
            if len(self.tokenizer.encode(prompt)) <= self.config.max_input_tokens:
                truncation_info["steps"].append("compressed_history")
                return prompt, truncation_info
        
        # Step 3: Hard truncate with summary
        truncated_messages = [
            messages[0],  # System
            {
                "role": "user",
                "content": f"[Input truncated to fit context. Original: {len(messages)} messages] " + 
                          messages[-1]["content"][:500]
            }
        ]
        prompt = self._build_with_tokenizer(truncated_messages)
        truncation_info["steps"].append("hard_truncate")
        return prompt, truncation_info
    
    def _compute_cache_key(
        self,
        messages: List[Dict[str, str]],
        params: GenerationParams,
        model_id: str
    ) -> str:
        """Compute deterministic cache key"""
        # Create normalized representation
        cache_data = {
            "messages": messages,
            "params": params.to_dict(),
            "model_id": model_id,
            "prompt_version": self.config.prompt_version.value
        }
        
        # Convert to JSON for consistent hashing
        cache_json = json.dumps(cache_data, sort_keys=True, separators=(',', ':'))
        
        # Compute SHA256 hash
        return hashlib.sha256(cache_json.encode()).hexdigest()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get builder metrics"""
        hit_rate = 0
        if self._metrics["total_builds"] > 0:
            hit_rate = self._metrics["cache_hits"] / self._metrics["total_builds"]
        
        return {
            **self._metrics,
            "cache_hit_rate": hit_rate,
            "cache_size": len(self._cache)
        }
    
    def clear_cache(self):
        """Clear the cache"""
        self._cache.clear()
        logger.info(f"Cleared prompt cache")


# Example usage and testing
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Mock tokenizer for testing
    class MockTokenizer:
        def encode(self, text):
            return text.split()
        
        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            result = []
            for msg in messages:
                result.append(f"{msg['role']}: {msg['content']}")
            if add_generation_prompt:
                result.append("assistant:")
            return "\n".join(result)
    
    tokenizer = MockTokenizer()
    builder = PromptBuilder(tokenizer)
    
    # Test messages
    messages = [
        {"role": "user", "content": "Hello, how are you?"}
    ]
    
    params = GenerationParams(temperature=0.7, max_tokens=100, seed=42)
    
    # Build prompt
    prompt, metadata = builder.build(messages, params, "gpt-oss-20b")
    
    print("Prompt:", prompt)
    print("Metadata:", json.dumps(metadata, indent=2))
    
    # Test cache hit
    prompt2, metadata2 = builder.build(messages, params, "gpt-oss-20b")
    print("Cache hit:", metadata2["cache_hit"])
    
    # Get metrics
    print("Metrics:", json.dumps(builder.get_metrics(), indent=2))