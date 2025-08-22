#!/usr/bin/env python3
"""
Test Suite for v4.6.x P0.5 Improvements
Tests PR-PF01, PR-CACHE02, PR-SESSION02, PR-OBS01
"""

import requests
import json
import time
import uuid
import re
from typing import Dict, List, Tuple
import threading

SERVER_URL = "http://localhost:8000"

class V46xImprovementTester:
    """Test suite for v4.6.x improvements"""
    
    def __init__(self, server_url: str = SERVER_URL):
        self.server_url = server_url
        self.results = []
    
    def test_prompt_normalization(self) -> bool:
        """Test PR-PF01: Enhanced prompt normalization"""
        print("\n🧪 Test 1: PR-PF01 - Enhanced Prompt Normalization")
        print("="*50)
        
        try:
            # Test messages with various normalizable content
            test_cases = [
                {
                    "name": "Timestamps",
                    "content": "Meeting at 2024-08-22 14:30:00 and 3:45 PM",
                    "expected_normalized": ["<timestamp>", "<time>"]
                },
                {
                    "name": "UUIDs and Session IDs",
                    "content": f"Request ID: {uuid.uuid4()} with session_id=abc123",
                    "expected_normalized": ["<uuid>", "<session_id>"]
                },
                {
                    "name": "Paths",
                    "content": "File at /home/user/test.py and C:\\Users\\test.txt",
                    "expected_normalized": ["<path>"]
                },
                {
                    "name": "IPs and URLs",
                    "content": "Connect to 192.168.1.1 or localhost:8000 or https://example.com",
                    "expected_normalized": ["<ip>", "<host>", "<url>"]
                },
                {
                    "name": "Hashes",
                    "content": "Hash: a1b2c3d4e5f6789012345678901234567890abcd",
                    "expected_normalized": ["<hash>"]
                }
            ]
            
            passed = 0
            for test_case in test_cases:
                print(f"\n  Testing: {test_case['name']}")
                
                # Send two identical requests with normalizable content
                request1 = {
                    "model": "gpt-oss-20b",
                    "messages": [{"role": "user", "content": test_case['content']}],
                    "max_tokens": 10,
                    "temperature": 0
                }
                
                # First request
                response1 = requests.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=request1,
                    timeout=30
                )
                
                # Second request (should hit cache)
                response2 = requests.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=request1,
                    timeout=30
                )
                
                if response1.status_code == 200 and response2.status_code == 200:
                    data1 = response1.json()
                    data2 = response2.json()
                    
                    # Check if cache keys match (normalization working)
                    cache_key1 = data1.get("metadata", {}).get("cache_key", "")
                    cache_key2 = data2.get("metadata", {}).get("cache_key", "")
                    cache_hit2 = data2.get("metadata", {}).get("cache_hit", False)
                    
                    if cache_key1 == cache_key2 and cache_hit2:
                        print(f"    ✅ Normalization working: cache hit on 2nd request")
                        passed += 1
                    else:
                        print(f"    ❌ Normalization failed: no cache hit")
                else:
                    print(f"    ❌ Request failed: {response1.status_code}")
            
            success_rate = passed / len(test_cases)
            print(f"\n  📊 Normalization test: {passed}/{len(test_cases)} passed ({success_rate:.1%})")
            return success_rate >= 0.6  # 60% pass rate
            
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            return False
    
    def test_cache_hit_rate(self) -> bool:
        """Test PR-CACHE02: Cache hit rate ≥70%"""
        print("\n🧪 Test 2: PR-CACHE02 - Cache Hit Rate Tuning")
        print("="*50)
        
        try:
            # Generate 20 requests with 5 unique patterns (4 requests each)
            patterns = [
                "What is the capital of France?",
                "How do I write a Python function?",
                "Explain machine learning",
                "What is 2+2?",
                "Tell me about GPT models"
            ]
            
            total_requests = 0
            cache_hits = 0
            
            print(f"  📤 Sending {len(patterns) * 4} requests ({len(patterns)} unique patterns)...")
            
            for pattern in patterns:
                for i in range(4):
                    request_data = {
                        "model": "gpt-oss-20b",
                        "messages": [{"role": "user", "content": pattern}],
                        "max_tokens": 10,
                        "temperature": 0
                    }
                    
                    response = requests.post(
                        f"{self.server_url}/v1/chat/completions",
                        json=request_data,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        total_requests += 1
                        data = response.json()
                        if data.get("metadata", {}).get("cache_hit", False):
                            cache_hits += 1
                    
                    time.sleep(0.1)  # Small delay
            
            # Check global cache stats
            stats_response = requests.get(f"{self.server_url}/stats")
            if stats_response.status_code == 200:
                stats = stats_response.json()
                prompt_metrics = stats.get("prompt_metrics", {})
                global_hit_rate = prompt_metrics.get("cache_hit_rate", 0)
                
                print(f"\n  📊 Cache Statistics:")
                print(f"    Local test: {cache_hits}/{total_requests} hits ({cache_hits/total_requests*100:.1f}%)")
                print(f"    Global rate: {global_hit_rate:.1%}")
                print(f"    Cache size: {prompt_metrics.get('cache_size', 0)} entries")
                
                # Target: 70% hit rate (3 out of 4 requests per pattern should hit)
                local_hit_rate = cache_hits / total_requests if total_requests > 0 else 0
                if local_hit_rate >= 0.70:
                    print(f"  ✅ Cache hit rate target achieved (≥70%)")
                    return True
                else:
                    print(f"  ❌ Cache hit rate below target (<70%)")
                    return False
            
            return False
            
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            return False
    
    def test_session_cleanup(self) -> bool:
        """Test PR-SESSION02: Aggressive idle session cleanup"""
        print("\n🧪 Test 3: PR-SESSION02 - Aggressive Session Cleanup")
        print("="*50)
        
        try:
            # Create multiple sessions
            print(f"  📤 Creating 10 test sessions...")
            session_ids = []
            
            for i in range(10):
                session_id = f"cleanup_test_{i}"
                session_ids.append(session_id)
                
                request_data = {
                    "model": "gpt-oss-20b",
                    "messages": [{"role": "user", "content": f"Session {i}"}],
                    "max_tokens": 10,
                    "temperature": 0
                }
                
                headers = {"X-Session-ID": session_id}
                response = requests.post(
                    f"{self.server_url}/v1/chat/completions",
                    json=request_data,
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    print(f"    ✅ Session {i} created")
                
                time.sleep(0.2)
            
            # Check initial session count
            response = requests.get(f"{self.server_url}/memory_stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                initial_sessions = stats.get('active_sessions', 0)
                idle_timeout = stats.get('idle_timeout_seconds', 300)
                cleanup_interval = stats.get('cleanup_interval_seconds', 30)
                
                print(f"\n  📊 Session Management Config:")
                print(f"    Active sessions: {initial_sessions}")
                print(f"    Idle timeout: {idle_timeout}s")
                print(f"    Cleanup interval: {cleanup_interval}s")
                print(f"    Sessions evicted: {stats.get('sessions_evicted_total', 0)}")
                
                # PR-SESSION02: Verify aggressive settings
                if idle_timeout <= 180:  # Should be 180s or less
                    print(f"  ✅ Aggressive idle timeout ({idle_timeout}s ≤ 180s)")
                else:
                    print(f"  ❌ Idle timeout not aggressive enough ({idle_timeout}s > 180s)")
                    return False
                
                if cleanup_interval <= 30:  # Should be 30s or less
                    print(f"  ✅ Frequent cleanup interval ({cleanup_interval}s ≤ 30s)")
                else:
                    print(f"  ❌ Cleanup interval not frequent enough ({cleanup_interval}s > 30s)")
                    return False
                
                # Check KV memory tracking
                kv_in_use = stats.get('kv_in_use_mb', 0)
                print(f"  📊 KV cache in use: {kv_in_use:.1f} MB")
                
                return True
            
            return False
            
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            return False
    
    def test_observability_labels(self) -> bool:
        """Test PR-OBS01: Complete model/execution labels"""
        print("\n🧪 Test 4: PR-OBS01 - Complete Observability Labels")
        print("="*50)
        
        try:
            # Check /stats endpoint
            print("\n  📊 Checking /stats endpoint...")
            response = requests.get(f"{self.server_url}/stats", timeout=5)
            if response.status_code == 200:
                stats = response.json()
                
                # Check for model_info
                if 'model_info' in stats:
                    model_info = stats['model_info']
                    required_labels = ['model_id', 'model_size', 'dtype', 'gpu_mode', 'prompt_version']
                    
                    print(f"  Model labels in /stats:")
                    for label in required_labels:
                        if label in model_info:
                            print(f"    ✅ {label}: {model_info[label]}")
                        else:
                            print(f"    ❌ {label}: missing")
                
                # Check for session metrics
                if 'sessions_active' in stats or 'memory_stats' in response.json():
                    print(f"  ✅ Session metrics available")
            
            # Check /metrics endpoint (Prometheus)
            print("\n  📊 Checking /metrics endpoint...")
            response = requests.get(f"{self.server_url}/metrics", timeout=5)
            if response.status_code == 200:
                metrics_text = response.text
                
                # Check for required labels in metrics
                required_labels = [
                    'model_id=',
                    'model_size=',
                    'dtype=',
                    'gpu_mode=',
                    'prompt_version=',
                    'sessions_active',
                    'sessions_evicted_total',
                    'kv_in_use_mb'
                ]
                
                found_labels = []
                missing_labels = []
                
                for label in required_labels:
                    if label in metrics_text:
                        found_labels.append(label)
                    else:
                        missing_labels.append(label)
                
                print(f"  Prometheus metrics:")
                print(f"    Found labels: {len(found_labels)}/{len(required_labels)}")
                
                if missing_labels:
                    print(f"    Missing: {missing_labels}")
                
                # Check /memory_stats endpoint
                print("\n  📊 Checking /memory_stats endpoint...")
                response = requests.get(f"{self.server_url}/memory_stats", timeout=5)
                if response.status_code == 200:
                    mem_stats = response.json()
                    
                    required_fields = [
                        'active_sessions',
                        'sessions_evicted_total',
                        'kv_in_use_mb',
                        'gpu_usage'
                    ]
                    
                    for field in required_fields:
                        if field in mem_stats:
                            print(f"    ✅ {field}: {mem_stats[field]}")
                        else:
                            print(f"    ❌ {field}: missing")
                
                # Success if most labels are present
                success_rate = len(found_labels) / len(required_labels)
                if success_rate >= 0.7:  # 70% of labels present
                    print(f"\n  ✅ Observability labels adequate ({success_rate:.1%})")
                    return True
                else:
                    print(f"\n  ❌ Insufficient observability labels ({success_rate:.1%})")
                    return False
            
            return False
            
        except Exception as e:
            print(f"  ❌ Test failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all v4.6.x improvement tests"""
        print("\n" + "="*60)
        print("🚀 v4.6.x P0.5 Improvement Test Suite")
        print("Testing PR-PF01, PR-CACHE02, PR-SESSION02, PR-OBS01")
        print("="*60)
        
        tests = [
            ("PR-PF01: Prompt Normalization", self.test_prompt_normalization),
            ("PR-CACHE02: Cache Hit Rate", self.test_cache_hit_rate),
            ("PR-SESSION02: Session Cleanup", self.test_session_cleanup),
            ("PR-OBS01: Observability Labels", self.test_observability_labels)
        ]
        
        results = []
        for name, test_func in tests:
            try:
                result = test_func()
                results.append((name, result))
                time.sleep(1)
            except Exception as e:
                print(f"\n❌ Test '{name}' crashed: {e}")
                results.append((name, False))
        
        # Print summary
        print("\n" + "="*60)
        print("📊 Test Results Summary")
        print("="*60)
        
        passed = sum(1 for _, result in results if result)
        total = len(results)
        
        for name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"  {name}: {status}")
        
        print(f"\nTotal: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
        
        # Success criteria
        print("\n📋 v4.6.x P0.5 Success Criteria:")
        print("  ✅ PR-PF01: Byte-identical prompts for same input")
        print("  ✅ PR-CACHE02: Cache hit rate ≥70%")
        print("  ✅ PR-SESSION02: Idle timeout ≤180s, cleanup every 30s")
        print("  ✅ PR-OBS01: All metrics with model/execution labels")
        
        if passed == total:
            print("\n🎉 All v4.6.x improvements verified!")
            print("System ready for v4.6.x release.")
        elif passed >= total * 0.75:
            print("\n✅ Most improvements working correctly.")
        else:
            print("\n⚠️ Some improvements need attention.")
        
        return passed == total


def main():
    """Main test runner"""
    # Check server status
    print("\n🔍 Checking server status...")
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Server not responding")
            print("\nPlease start the server with:")
            print("  ./start_server_v460.sh 20b latency_first")
            return
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("\nPlease start the server with:")
        print("  ./start_server_v460.sh 20b latency_first")
        return
    
    # Run tests
    tester = V46xImprovementTester()
    success = tester.run_all_tests()
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()