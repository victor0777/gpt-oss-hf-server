#!/usr/bin/env python3
"""
Quick 5-minute status check for v4.3
"""

import requests
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import json

def get_gpu_stats():
    """Get GPU utilization"""
    cmd = "nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits"
    result = subprocess.run(cmd.split(), capture_output=True, text=True)
    
    utils = []
    for line in result.stdout.strip().split('\n'):
        parts = line.split(', ')
        utils.append(float(parts[1]))
    return utils

def send_request(idx):
    """Send test request"""
    try:
        start = time.time()
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json={
                "model": "gpt-oss-20b",
                "messages": [{"role": "user", "content": f"Test {idx}: Hello world"}],
                "max_tokens": 50,
                "temperature": 0.7
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return {
                "success": True,
                "latency_ms": (time.time() - start) * 1000,
                "idx": idx
            }
        else:
            return {"success": False, "idx": idx}
    except Exception as e:
        return {"success": False, "idx": idx, "error": str(e)}

def main():
    print("="*60)
    print("🚀 v4.3 5-Minute Status Check")
    print("="*60)
    
    # Test configuration
    test_duration = 300  # 5 minutes
    concurrent_requests = 8
    
    print(f"\n📊 Test Configuration:")
    print(f"  Duration: {test_duration}s")
    print(f"  Concurrent Requests: {concurrent_requests}")
    print(f"  Target QPS: ≥2.0")
    print(f"  Target P95 Latency: ≤7000ms")
    
    # Check initial health
    try:
        health = requests.get("http://localhost:8000/health", timeout=5).json()
        print(f"\n🏥 Initial Health: {health['status']} (score: {health['score']:.0f})")
    except:
        print("❌ Server not responding!")
        return False
    
    # Run test
    print(f"\n🔄 Running {test_duration}s test...")
    
    start_time = time.time()
    results = []
    gpu_samples = []
    request_idx = 0
    
    with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
        futures = []
        
        while time.time() - start_time < test_duration:
            # Submit new requests
            while len(futures) < concurrent_requests:
                futures.append(executor.submit(send_request, request_idx))
                request_idx += 1
            
            # Collect completed requests
            done_futures = []
            for future in futures:
                if future.done():
                    try:
                        result = future.result()
                        results.append(result)
                        done_futures.append(future)
                        
                        # Print progress every 10 requests
                        if len(results) % 10 == 0:
                            success_rate = sum(1 for r in results if r["success"]) / len(results) * 100
                            print(f"  Progress: {len(results)} requests, {success_rate:.1f}% success")
                    except:
                        done_futures.append(future)
            
            # Remove completed futures
            for f in done_futures:
                futures.remove(f)
            
            # Sample GPU usage periodically
            if len(gpu_samples) == 0 or time.time() - gpu_samples[-1]["time"] > 2:
                gpu_utils = get_gpu_stats()
                gpu_samples.append({
                    "time": time.time() - start_time,
                    "utils": gpu_utils
                })
            
            time.sleep(0.1)
    
    # Analysis
    total_time = time.time() - start_time
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    print(f"\n📊 Results Summary:")
    print(f"  Total Requests: {len(results)}")
    print(f"  Successful: {len(successful)} ({len(successful)/len(results)*100:.1f}%)")
    print(f"  Failed: {len(failed)}")
    print(f"  Test Duration: {total_time:.1f}s")
    
    if successful:
        latencies = [r["latency_ms"] for r in successful]
        qps = len(successful) / total_time
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        print(f"\n⚡ Performance Metrics:")
        print(f"  QPS: {qps:.2f} {'✅' if qps >= 2.0 else '❌'} (target ≥2.0)")
        print(f"  P50 Latency: {p50:.0f}ms")
        print(f"  P95 Latency: {p95:.0f}ms {'✅' if p95 <= 7000 else '❌'} (target ≤7000ms)")
        print(f"  P99 Latency: {p99:.0f}ms")
    
    # GPU utilization analysis
    if gpu_samples:
        max_utils = [0, 0, 0, 0]
        avg_utils = [[], [], [], []]
        
        for sample in gpu_samples:
            for i, util in enumerate(sample["utils"]):
                max_utils[i] = max(max_utils[i], util)
                avg_utils[i].append(util)
        
        print(f"\n🎮 GPU Utilization:")
        active_gpus = 0
        for i in range(4):
            avg = sum(avg_utils[i]) / len(avg_utils[i]) if avg_utils[i] else 0
            print(f"  GPU {i}: max={max_utils[i]:.0f}%, avg={avg:.1f}%")
            if max_utils[i] > 10:
                active_gpus += 1
        
        print(f"  Active GPUs: {active_gpus}/4 {'✅' if active_gpus >= 3 else '❌'}")
    
    # Get server stats
    try:
        stats = requests.get("http://localhost:8000/stats", timeout=5).json()
        print(f"\n📈 Server Statistics:")
        print(f"  Total Processed: {stats['requests']['completed_requests']}")
        print(f"  Active Requests: {stats['requests']['active_requests']}")
        print(f"  GPU Switches: {stats['requests']['gpu_switches']}")
    except:
        pass
    
    # Final verdict
    print(f"\n{'='*60}")
    
    success_criteria = {
        "QPS": qps >= 2.0 if successful else False,
        "P95 Latency": p95 <= 7000 if successful else False,
        "Multi-GPU": active_gpus >= 3 if gpu_samples else False,
        "Error Rate": len(failed) / len(results) < 0.005 if results else False
    }
    
    passed = sum(success_criteria.values())
    total = len(success_criteria)
    
    print(f"✅ Passed: {passed}/{total} criteria")
    
    if passed == total:
        print("🎉 ALL CRITERIA MET - Ready for full test!")
        return True
    elif passed >= 3:
        print("⚠️ PARTIAL SUCCESS - Some optimization needed")
        return True
    else:
        print("❌ INSUFFICIENT PERFORMANCE - Major issues remain")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)