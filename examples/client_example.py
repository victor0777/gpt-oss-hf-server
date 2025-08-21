#!/usr/bin/env python3
"""
Example client for GPT-OSS HuggingFace Server
"""

import requests
import json
import time

def simple_chat_completion():
    """Simple chat completion example"""
    
    url = "http://localhost:8000/v1/chat/completions"
    
    payload = {
        "model": "gpt-oss",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        "max_tokens": 100,
        "temperature": 0.7
    }
    
    print("Sending request...")
    start_time = time.time()
    
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nResponse received in {time.time() - start_time:.2f} seconds")
        print(f"Assistant: {result['choices'][0]['message']['content']}")
        print(f"\nTokens used: {result['usage']['total_tokens']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

def streaming_example():
    """Streaming response example (if implemented)"""
    # Note: Streaming not yet implemented in current version
    pass

def batch_requests_example():
    """Example of sending multiple concurrent requests"""
    
    import concurrent.futures
    
    def send_request(idx):
        url = "http://localhost:8000/v1/chat/completions"
        payload = {
            "model": "gpt-oss",
            "messages": [
                {"role": "user", "content": f"Tell me a fact about the number {idx}"}
            ],
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        return None
    
    print("Sending 5 concurrent requests...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(send_request, i) for i in range(1, 6)]
        
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            if result:
                print(f"\nResponse {i}: {result[:100]}...")

def check_server_health():
    """Check server health and status"""
    
    # Health check
    health_url = "http://localhost:8000/health"
    response = requests.get(health_url)
    
    if response.status_code == 200:
        health = response.json()
        print("Server Health:")
        print(f"  Status: {health['status']}")
        print(f"  Score: {health['score']}")
        print(f"  Model Size: {health.get('model_size', 'unknown')}")
        print(f"  GPU Mode: {health.get('gpu_mode', 'unknown')}")
        print(f"  GPU Count: {health['gpu_count']}")
    
    # Stats
    stats_url = "http://localhost:8000/stats"
    response = requests.get(stats_url)
    
    if response.status_code == 200:
        stats = response.json()
        print("\nServer Statistics:")
        print(f"  Total Requests: {stats['requests']['total_requests']}")
        print(f"  Completed: {stats['requests']['completed_requests']}")
        print(f"  Active: {stats['requests']['active_requests']}")
        if 'performance' in stats:
            print(f"  Current QPS: {stats['performance'].get('current_qps', 0):.2f}")

if __name__ == "__main__":
    print("GPT-OSS Server Client Examples")
    print("="*40)
    
    # Check server health first
    print("\n1. Checking server health...")
    check_server_health()
    
    # Simple chat completion
    print("\n2. Simple chat completion...")
    simple_chat_completion()
    
    # Batch requests
    print("\n3. Batch requests example...")
    batch_requests_example()
    
    print("\nDone!")