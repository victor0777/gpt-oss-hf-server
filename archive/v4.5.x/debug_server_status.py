#!/usr/bin/env python3
"""
Debug script to check server and model loading status
"""

import torch
import psutil
import requests
import json

def check_gpu_status():
    """Check GPU availability and memory"""
    print("\n🖥️ GPU Status")
    print("="*50)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    print(f"📌 CUDA version: {torch.version.cuda}")
    print(f"📌 GPU count: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)
        
        print(f"\nGPU {i}: {props.name}")
        print(f"  Total memory: {total / 1024**3:.1f} GB")
        print(f"  Free memory: {free / 1024**3:.1f} GB")
        print(f"  Used memory: {(total - free) / 1024**3:.1f} GB")
        print(f"  Usage: {(total - free) / total * 100:.1f}%")
        
        if free / 1024**3 < 30:
            print(f"  ⚠️ Low free memory! Need at least 30GB for 20B model")
    
    return True

def check_system_memory():
    """Check system RAM"""
    print("\n💾 System Memory")
    print("="*50)
    
    mem = psutil.virtual_memory()
    print(f"Total RAM: {mem.total / 1024**3:.1f} GB")
    print(f"Available: {mem.available / 1024**3:.1f} GB")
    print(f"Used: {mem.used / 1024**3:.1f} GB ({mem.percent}%)")
    
    if mem.available / 1024**3 < 16:
        print("⚠️ Low system memory! Recommend at least 16GB free")
    else:
        print("✅ Sufficient system memory")

def check_server_health():
    """Check server health status"""
    print("\n🔍 Server Health Check")
    print("="*50)
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            
            status = health.get('status', 'unknown')
            print(f"Server status: {status}")
            
            if status == "healthy":
                print("✅ Server is healthy")
            elif status == "degraded":
                print("⚠️ Server is degraded - check logs")
            else:
                print(f"❌ Server status: {status}")
            
            # Check model info
            if 'model' in health and health['model']:
                model = health['model']
                print(f"\n📌 Model Info:")
                print(f"  Model ID: {model.get('model_id', 'N/A')}")
                print(f"  Model size: {model.get('model_size', 'N/A')}")
                print(f"  dtype: {model.get('dtype', 'N/A')}")
                print(f"  GPU mode: {model.get('gpu_mode', 'N/A')}")
            else:
                print("\n❌ No model information - model may not be loaded")
            
            return status == "healthy"
        else:
            print(f"❌ Server returned status code: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot connect to server: {e}")
        print("\nPlease start the server with:")
        print("  python src/server.py --model 20b --profile latency_first")
        return False

def check_memory_stats():
    """Check memory management stats"""
    print("\n📊 Memory Management Stats")
    print("="*50)
    
    try:
        response = requests.get("http://localhost:8000/memory_stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            
            print(f"Active sessions: {stats.get('active_sessions', 0)}")
            print(f"Total KV cache: {stats.get('total_kv_mb', 0):.1f} MB")
            print(f"Rejected requests: {stats.get('rejected_count', 0)}")
            print(f"Degraded requests: {stats.get('degraded_count', 0)}")
            
            if 'gpu_memory' in stats and stats['gpu_memory']:
                print("\nGPU Memory Status:")
                for gpu in stats['gpu_memory']:
                    print(f"  GPU {gpu['gpu_id']}: {gpu['used_gb']:.1f}/{gpu['total_gb']:.1f} GB ({gpu['usage_percent']:.1f}%)")
            
            return True
        else:
            print(f"❌ Memory stats returned: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Cannot get memory stats: {e}")
        return False

def test_simple_request():
    """Test a simple request"""
    print("\n🧪 Testing Simple Request")
    print("="*50)
    
    try:
        request_data = {
            "model": "gpt-oss-20b",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10,
            "temperature": 0
        }
        
        print("Sending test request...")
        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json=request_data,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ Request successful")
            data = response.json()
            if 'choices' in data and data['choices']:
                content = data['choices'][0]['message']['content']
                print(f"Response: {content[:100]}...")
            return True
        else:
            print(f"❌ Request failed: {response.status_code}")
            try:
                error = response.json()
                print(f"Error: {error.get('detail', 'Unknown error')}")
            except:
                print(f"Response: {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"❌ Request error: {e}")
        return False

def main():
    """Run all diagnostics"""
    print("\n" + "="*60)
    print("🔧 GPT-OSS Server Diagnostics")
    print("="*60)
    
    results = []
    
    # Check GPU
    results.append(("GPU Status", check_gpu_status()))
    
    # Check system memory
    check_system_memory()
    
    # Check server health
    results.append(("Server Health", check_server_health()))
    
    # Check memory stats
    results.append(("Memory Stats", check_memory_stats()))
    
    # Test request
    results.append(("Test Request", test_simple_request()))
    
    # Summary
    print("\n" + "="*60)
    print("📊 Diagnostic Summary")
    print("="*60)
    
    all_pass = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        if not result:
            all_pass = False
    
    if all_pass:
        print("\n✅ All diagnostics passed! Server is ready.")
    else:
        print("\n⚠️ Some diagnostics failed. Please check:")
        print("1. Ensure sufficient GPU memory (>30GB free)")
        print("2. Check server logs for model loading errors")
        print("3. Restart server if model not loaded")
        print("\nRestart command:")
        print("  python src/server.py --model 20b --profile latency_first")

if __name__ == "__main__":
    main()