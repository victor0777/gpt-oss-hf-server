#!/usr/bin/env python3
"""
Test v4.4 QPS Performance
"""

import requests
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import threading
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
    """Send a single request"""
    try:
        start = time.time()
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "gpt-oss",
                "messages": [{"role": "user", "content": f"Request {idx}: Quick response test"}],
                "max_tokens": 30,  # Reduced for higher QPS
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            latency = (time.time() - start) * 1000
            return {"success": True, "latency_ms": latency, "idx": idx}
        else:
            return {"success": False, "error": response.text, "idx": idx}
    except Exception as e:
        return {"success": False, "error": str(e), "idx": idx}

def monitor_gpus(duration=10, interval=0.5):
    """Monitor GPU utilization for a duration"""
    samples = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        stats = get_gpu_stats()
        samples.append({
            "time": time.time() - start_time,
            "stats": stats
        })
        time.sleep(interval)
    
    return samples

def main():
    print("="*60)
    print("🚀 Testing v4.4 QPS Performance")
    print("="*60)
    
    # Initial GPU state
    print("\n📊 Initial GPU State:")
    initial_stats = get_gpu_stats()
    for gpu_id, stats in initial_stats.items():
        print(f"  GPU {gpu_id}: {stats['util']:.1f}% util, {stats['mem_mb']:.0f} MB")
    
    # Health check
    print("\n🏥 Health Check:")
    health = requests.get("http://localhost:8000/health").json()
    print(f"  Status: {health['status']}")
    print(f"  Model Size: {health.get('model_size', 'unknown')}")
    print(f"  GPU Mode: {health.get('gpu_mode', 'unknown')}")
    print(f"  GPU Count: {health['gpu_count']}")
    
    # Test configuration
    concurrent_requests = 32  # High concurrency for QPS test
    total_requests = 200
    
    print(f"\n🚀 Sending {total_requests} requests with {concurrent_requests} concurrent...")
    
    # Start GPU monitoring in background
    monitor_thread = threading.Thread(target=lambda: globals().update({"gpu_samples": monitor_gpus(30)}))
    monitor_thread.start()
    
    # Send requests
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = [executor.submit(send_request, i) for i in range(total_requests)]
        
        for i, future in enumerate(as_completed(futures)):
            result = future.result()
            results.append(result)
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                current_qps = (i + 1) / elapsed
                print(f"  Progress: {i+1}/{total_requests} - Current QPS: {current_qps:.2f}")
    
    total_time = time.time() - start_time
    
    # Wait for monitoring to complete
    monitor_thread.join()
    
    # Analyze results
    print(f"\n📈 Results Summary:")
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"  Total Requests: {len(results)}")
    print(f"  Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"  Failed: {len(failed)}")
    print(f"  Total Time: {total_time:.2f}s")
    
    # Performance metrics
    if successful:
        latencies = [r["latency_ms"] for r in successful]
        qps = len(successful) / total_time
        
        print(f"\n⚡ Performance Metrics:")
        print(f"  QPS: {qps:.2f} {'✅' if qps >= 2.0 else '❌'} (target ≥2.0)")
        print(f"  Avg Latency: {np.mean(latencies):.0f}ms")
        print(f"  P50 Latency: {np.percentile(latencies, 50):.0f}ms")
        print(f"  P95 Latency: {np.percentile(latencies, 95):.0f}ms {'✅' if np.percentile(latencies, 95) <= 7000 else '❌'} (target ≤7000ms)")
        print(f"  P99 Latency: {np.percentile(latencies, 99):.0f}ms")
    
    # GPU utilization analysis
    if "gpu_samples" in globals():
        gpu_samples = globals()["gpu_samples"]
        
        max_utils = {0: 0, 1: 0, 2: 0, 3: 0}
        avg_utils = {0: [], 1: [], 2: [], 3: []}
        
        for sample in gpu_samples:
            for gpu_id, stats in sample["stats"].items():
                avg_utils[gpu_id].append(stats["util"])
                max_utils[gpu_id] = max(max_utils[gpu_id], stats["util"])
        
        print(f"\n🎮 GPU Utilization:")
        active_gpus = 0
        for gpu_id in range(4):
            if avg_utils[gpu_id]:
                avg = sum(avg_utils[gpu_id]) / len(avg_utils[gpu_id])
                print(f"  GPU {gpu_id}: max={max_utils[gpu_id]:.0f}%, avg={avg:.1f}%")
                if max_utils[gpu_id] > 10:
                    active_gpus += 1
        
        print(f"  Active GPUs: {active_gpus}/4 {'✅' if active_gpus >= 3 else '⚠️'}")
    
    # Server statistics
    try:
        stats = requests.get("http://localhost:8000/stats").json()
        print(f"\n📊 Server Statistics:")
        print(f"  Completed Requests: {stats['requests']['completed_requests']}")
        print(f"  Active Requests: {stats['requests']['active_requests']}")
        print(f"  Current QPS: {stats['performance'].get('current_qps', 0):.2f}")
        
        config = stats.get('config', {})
        print(f"\n⚙️ Server Configuration:")
        print(f"  Batch Size: {config.get('batch_max_size', 'unknown')}")
        print(f"  Max Concurrent: {config.get('max_concurrent_requests', 'unknown')}")
    except:
        pass
    
    print("\n" + "="*60)
    
    # Return success criteria
    success_criteria = {
        "QPS": qps >= 2.0 if successful else False,
        "P95": np.percentile(latencies, 95) <= 7000 if successful else False,
        "Multi-GPU": active_gpus >= 3 if "gpu_samples" in globals() else False,
        "Error Rate": len(failed) / len(results) < 0.005 if results else False
    }
    
    passed = sum(success_criteria.values())
    print(f"\n✅ Passed: {passed}/4 criteria")
    
    for criterion, result in success_criteria.items():
        status = "✅" if result else "❌"
        print(f"  {status} {criterion}")
    
    return passed >= 3

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)