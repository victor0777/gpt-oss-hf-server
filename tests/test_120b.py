#!/usr/bin/env python3
"""
Test 120b Model Performance
"""

import requests
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import json

def get_gpu_stats():
    """Get current GPU utilization"""
    cmd = "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits"
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    
    stats = {}
    for line in result.stdout.strip().split('\n'):
        parts = line.split(', ')
        gpu_id = int(parts[0])
        util = float(parts[1])
        mem = float(parts[2])
        stats[gpu_id] = {"util": util, "mem_mb": mem}
    return stats

def send_request(idx):
    """Send a single request to 120b model"""
    try:
        start = time.time()
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "gpt-oss-120b",
                "messages": [{"role": "user", "content": f"Test {idx}: Explain quantum computing in simple terms"}],
                "max_tokens": 100,
                "temperature": 0.7
            },
            timeout=60  # Longer timeout for larger model
        )
        
        if response.status_code == 200:
            latency = (time.time() - start) * 1000
            return {"success": True, "latency_ms": latency, "idx": idx}
        else:
            return {"success": False, "error": response.text[:100], "idx": idx}
    except Exception as e:
        return {"success": False, "error": str(e)[:100], "idx": idx}

def main():
    print("="*60)
    print("🚀 Testing 120b Model Performance")
    print("="*60)
    
    # Check health
    print("\n🏥 Health Check:")
    try:
        health = requests.get("http://localhost:8000/health", timeout=5).json()
        print(f"  Status: {health['status']}")
        print(f"  Model Size: {health.get('model_size', 'unknown')}")
        print(f"  GPU Mode: {health.get('gpu_mode', 'unknown')}")
        print(f"  GPU Count: {health['gpu_count']}")
    except Exception as e:
        print(f"  ❌ Health check failed: {e}")
        return False
    
    # Initial GPU state
    print("\n📊 Initial GPU State:")
    initial_stats = get_gpu_stats()
    for gpu_id, stats in initial_stats.items():
        print(f"  GPU {gpu_id}: {stats['util']:.1f}% util, {stats['mem_mb']:.0f} MB")
    
    # Test configuration (reduced for 120b model)
    concurrent_requests = 4  # Lower concurrency for larger model
    total_requests = 20  # Fewer requests for quick test
    
    print(f"\n🚀 Sending {total_requests} requests with {concurrent_requests} concurrent...")
    print("  (Note: 120b model will be slower than 20b)")
    
    # Send requests
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(send_request, i) for i in range(total_requests)]
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            
            if result["success"]:
                print(f"  ✅ Request {result['idx']}: {result['latency_ms']:.0f}ms")
            else:
                print(f"  ❌ Request {result['idx']}: {result['error']}")
            
            # Sample GPU usage
            if i % 5 == 0:
                gpu_stats = get_gpu_stats()
                max_util = max(s["util"] for s in gpu_stats.values())
                if max_util > 0:
                    print(f"     GPU Utilization: {max_util:.0f}%")
    
    total_time = time.time() - start_time
    
    # Final GPU state
    print("\n📊 Final GPU State:")
    final_stats = get_gpu_stats()
    for gpu_id, stats in final_stats.items():
        print(f"  GPU {gpu_id}: {stats['util']:.1f}% util, {stats['mem_mb']:.0f} MB")
    
    # Analyze results
    print(f"\n📈 Results Summary:")
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"  Total Requests: {len(results)}")
    print(f"  Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"  Failed: {len(failed)}")
    print(f"  Total Time: {total_time:.2f}s")
    
    if successful:
        latencies = [r["latency_ms"] for r in successful]
        qps = len(successful) / total_time
        
        print(f"\n⚡ Performance Metrics:")
        print(f"  QPS: {qps:.2f}")
        print(f"  Avg Latency: {np.mean(latencies):.0f}ms")
        print(f"  P50 Latency: {np.percentile(latencies, 50):.0f}ms")
        print(f"  P95 Latency: {np.percentile(latencies, 95):.0f}ms")
        print(f"  P99 Latency: {np.percentile(latencies, 99):.0f}ms")
        
        # Model size comparison
        print(f"\n📊 Model Comparison:")
        print(f"  120b Model:")
        print(f"    - Memory Usage: ~60GB across 4 GPUs")
        print(f"    - QPS: {qps:.2f}")
        print(f"    - P95 Latency: {np.percentile(latencies, 95):.0f}ms")
        print(f"  20b Model (for reference):")
        print(f"    - Memory Usage: ~13GB per GPU (pipeline mode)")
        print(f"    - QPS Target: ≥2.0")
        print(f"    - P95 Latency Target: ≤7000ms")
    
    # Server statistics
    try:
        stats = requests.get("http://localhost:8000/stats", timeout=5).json()
        print(f"\n📊 Server Statistics:")
        print(f"  Completed Requests: {stats['requests']['completed_requests']}")
        print(f"  Current QPS: {stats['performance'].get('current_qps', 0):.2f}")
    except:
        pass
    
    print("\n" + "="*60)
    
    # Success criteria for 120b model (relaxed due to model size)
    if successful:
        success = len(successful) / len(results) >= 0.8  # 80% success rate
        print(f"\n{'✅' if success else '❌'} Test {'Passed' if success else 'Failed'}")
        print(f"  Note: 120b model naturally has lower QPS due to size")
        return success
    else:
        print("\n❌ Test Failed - No successful requests")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)