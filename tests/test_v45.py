#!/usr/bin/env python3
"""
Test script for v4.5 with engine adapter layer
Tests custom, vLLM, and auto routing
"""

import asyncio
import aiohttp
import time
import json
import sys
import argparse
from typing import Dict, List, Tuple
import statistics
from datetime import datetime

class V45Tester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = {
            "custom": {"latencies": [], "errors": 0, "success": 0},
            "vllm": {"latencies": [], "errors": 0, "success": 0},
            "auto": {"latencies": [], "errors": 0, "success": 0}
        }
        self.engine_distribution = {}
        
    async def test_health(self) -> dict:
        """Test health endpoint"""
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{self.base_url}/health") as resp:
                return await resp.json()
    
    async def test_completion(
        self,
        engine: str = "custom",
        messages: List[dict] = None,
        max_tokens: int = 50
    ) -> Tuple[bool, float, str]:
        """Test single completion"""
        if messages is None:
            messages = [{"role": "user", "content": "Hello, how are you?"}]
        
        payload = {
            "model": "gpt-oss-20b",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        # Set engine via header or env
        headers = {}
        if engine != "custom":
            headers["X-Engine-Type"] = engine
        
        start = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    latency = (time.time() - start) * 1000
                    
                    if resp.status == 200:
                        data = await resp.json()
                        chosen_engine = data.get("chosen_engine", "unknown")
                        return True, latency, chosen_engine
                    else:
                        return False, latency, "error"
        except Exception as e:
            latency = (time.time() - start) * 1000
            print(f"Error testing {engine}: {e}")
            return False, latency, "error"
    
    async def test_engine_routing(self, num_requests: int = 10):
        """Test routing between engines"""
        print(f"\n{'='*60}")
        print("Testing Engine Routing")
        print(f"{'='*60}")
        
        # Test different input sizes to trigger routing rules
        test_cases = [
            # (engine_mode, message_content, expected_route)
            ("auto", "Short prompt", "custom"),  # Short -> custom
            ("auto", " ".join(["word"] * 5000), "vllm"),  # Long -> vLLM
            ("custom", "Force custom", "custom"),  # Explicit custom
            ("vllm", "Force vLLM", "vllm"),  # Explicit vLLM
        ]
        
        for engine_mode, content, expected in test_cases:
            messages = [{"role": "user", "content": content}]
            success, latency, chosen = await self.test_completion(
                engine=engine_mode,
                messages=messages,
                max_tokens=10
            )
            
            status = "✓" if success else "✗"
            route_match = "✓" if chosen == expected else f"✗ (got {chosen})"
            
            print(f"{status} Engine={engine_mode:6} Input={len(content):5} chars "
                  f"-> Routed to {chosen:8} Expected={expected:8} {route_match} "
                  f"Latency={latency:.0f}ms")
    
    async def run_concurrent_test(
        self,
        engine: str,
        num_requests: int = 20,
        concurrent: int = 4
    ):
        """Run concurrent requests test"""
        print(f"\n{'='*60}")
        print(f"Testing {engine.upper()} Engine")
        print(f"Requests: {num_requests}, Concurrent: {concurrent}")
        print(f"{'='*60}")
        
        semaphore = asyncio.Semaphore(concurrent)
        
        async def limited_request(i: int):
            async with semaphore:
                return await self.test_completion(
                    engine=engine,
                    messages=[{"role": "user", "content": f"Test request {i}"}]
                )
        
        tasks = [limited_request(i) for i in range(num_requests)]
        results = await asyncio.gather(*tasks)
        
        # Process results
        for success, latency, chosen_engine in results:
            if success:
                self.results[engine]["success"] += 1
                self.results[engine]["latencies"].append(latency)
                
                # Track engine distribution
                if chosen_engine not in self.engine_distribution:
                    self.engine_distribution[chosen_engine] = 0
                self.engine_distribution[chosen_engine] += 1
            else:
                self.results[engine]["errors"] += 1
        
        # Print summary
        if self.results[engine]["latencies"]:
            latencies = self.results[engine]["latencies"]
            print(f"✓ Success: {self.results[engine]['success']}/{num_requests}")
            print(f"✗ Errors: {self.results[engine]['errors']}")
            print(f"📊 Latency - P50: {statistics.median(latencies):.0f}ms, "
                  f"P95: {statistics.quantiles(latencies, n=20)[18]:.0f}ms, "
                  f"P99: {statistics.quantiles(latencies, n=100)[98]:.0f}ms")
    
    async def run_ab_test(self, duration_seconds: int = 30):
        """Run A/B test between custom and vLLM"""
        print(f"\n{'='*60}")
        print(f"A/B Test: Custom vs vLLM ({duration_seconds}s)")
        print(f"{'='*60}")
        
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < duration_seconds:
            # Alternate between engines
            engine = "custom" if request_count % 2 == 0 else "vllm"
            
            success, latency, chosen = await self.test_completion(
                engine=engine,
                max_tokens=50
            )
            
            if success:
                self.results[engine]["success"] += 1
                self.results[engine]["latencies"].append(latency)
            else:
                self.results[engine]["errors"] += 1
            
            request_count += 1
            
            # Print progress every 10 requests
            if request_count % 10 == 0:
                elapsed = time.time() - start_time
                qps = request_count / elapsed
                print(f"Progress: {request_count} requests, {qps:.2f} QPS")
        
        # Print A/B results
        print(f"\n{'='*60}")
        print("A/B Test Results")
        print(f"{'='*60}")
        
        for engine in ["custom", "vllm"]:
            if self.results[engine]["latencies"]:
                latencies = self.results[engine]["latencies"]
                success_rate = (self.results[engine]["success"] / 
                              (self.results[engine]["success"] + self.results[engine]["errors"]) * 100)
                
                print(f"\n{engine.upper()} Engine:")
                print(f"  Success Rate: {success_rate:.1f}%")
                print(f"  P50 Latency: {statistics.median(latencies):.0f}ms")
                print(f"  P95 Latency: {statistics.quantiles(latencies, n=20)[18]:.0f}ms")
                print(f"  Mean Latency: {statistics.mean(latencies):.0f}ms")
    
    async def run_soak_test(self, duration_seconds: int = 300, qps_target: float = 2.0):
        """Run soak test with target QPS"""
        print(f"\n{'='*60}")
        print(f"Soak Test: {duration_seconds}s at {qps_target} QPS")
        print(f"{'='*60}")
        
        start_time = time.time()
        request_count = 0
        interval = 1.0 / qps_target
        
        # Metrics windows
        window_size = 60  # 1 minute windows
        windows = []
        current_window = {"start": start_time, "requests": 0, "errors": 0, "latencies": []}
        
        while time.time() - start_time < duration_seconds:
            # Check if we need a new window
            if time.time() - current_window["start"] >= window_size:
                windows.append(current_window)
                current_window = {"start": time.time(), "requests": 0, "errors": 0, "latencies": []}
            
            # Send request
            request_start = time.time()
            success, latency, chosen = await self.test_completion(
                engine="auto",
                max_tokens=50
            )
            
            current_window["requests"] += 1
            if success:
                current_window["latencies"].append(latency)
            else:
                current_window["errors"] += 1
            
            request_count += 1
            
            # Maintain target QPS
            elapsed = time.time() - request_start
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)
            
            # Print progress
            if request_count % 50 == 0:
                total_elapsed = time.time() - start_time
                actual_qps = request_count / total_elapsed
                recent_errors = sum(w["errors"] for w in windows[-3:]) if len(windows) >= 3 else 0
                
                print(f"[{int(total_elapsed)}s] Requests: {request_count}, "
                      f"QPS: {actual_qps:.2f}, Recent Errors: {recent_errors}")
        
        # Add final window
        windows.append(current_window)
        
        # Analyze windows for stability
        print(f"\n{'='*60}")
        print("Soak Test Analysis")
        print(f"{'='*60}")
        
        stable_windows = 0
        for i, window in enumerate(windows):
            if window["latencies"]:
                p95 = statistics.quantiles(window["latencies"], n=20)[18]
                error_rate = window["errors"] / window["requests"] if window["requests"] > 0 else 0
                
                # Check if window meets SLO
                is_stable = p95 < 20000 and error_rate < 0.005
                if is_stable:
                    stable_windows += 1
                
                print(f"Window {i+1}: P95={p95:.0f}ms, Errors={error_rate:.1%}, "
                      f"Stable={'✓' if is_stable else '✗'}")
        
        stability_rate = stable_windows / len(windows) * 100 if windows else 0
        print(f"\nOverall Stability: {stability_rate:.1f}% of windows met SLO")
        
        return stability_rate >= 95  # Pass if 95% of windows are stable
    
    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*60}")
        print("Test Summary")
        print(f"{'='*60}")
        
        # Engine distribution
        if self.engine_distribution:
            print("\nEngine Distribution:")
            total = sum(self.engine_distribution.values())
            for engine, count in sorted(self.engine_distribution.items()):
                percentage = count / total * 100
                print(f"  {engine}: {count} ({percentage:.1f}%)")
        
        # Performance by engine
        print("\nPerformance by Engine Type:")
        for engine in ["custom", "vllm", "auto"]:
            if self.results[engine]["latencies"]:
                latencies = self.results[engine]["latencies"]
                success = self.results[engine]["success"]
                errors = self.results[engine]["errors"]
                total = success + errors
                
                if total > 0:
                    print(f"\n{engine.upper()}:")
                    print(f"  Total Requests: {total}")
                    print(f"  Success Rate: {success/total*100:.1f}%")
                    print(f"  P50 Latency: {statistics.median(latencies):.0f}ms")
                    if len(latencies) > 1:
                        print(f"  P95 Latency: {statistics.quantiles(latencies, n=20)[18]:.0f}ms")
                    print(f"  Mean Latency: {statistics.mean(latencies):.0f}ms")

async def main():
    parser = argparse.ArgumentParser(description="Test v4.5 server")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--test", choices=["quick", "routing", "ab", "soak", "full"],
                       default="quick", help="Test type")
    parser.add_argument("--duration", type=int, default=30, help="Test duration (seconds)")
    parser.add_argument("--concurrent", type=int, default=4, help="Concurrent requests")
    parser.add_argument("--requests", type=int, default=20, help="Total requests")
    
    args = parser.parse_args()
    
    tester = V45Tester(args.url)
    
    # Check server health first
    print("Checking server health...")
    try:
        health = await tester.test_health()
        print(f"Server Status: {health['status']}")
        print(f"Health Score: {health['score']}")
        print(f"Available Engines: {list(health['engines'].keys())}")
    except Exception as e:
        print(f"❌ Server health check failed: {e}")
        sys.exit(1)
    
    # Run tests based on type
    if args.test == "quick":
        # Quick test of each engine
        for engine in ["custom", "vllm", "auto"]:
            await tester.run_concurrent_test(engine, args.requests, args.concurrent)
    
    elif args.test == "routing":
        # Test routing logic
        await tester.test_engine_routing()
    
    elif args.test == "ab":
        # A/B test
        await tester.run_ab_test(args.duration)
    
    elif args.test == "soak":
        # Soak test
        passed = await tester.run_soak_test(args.duration, qps_target=2.0)
        if not passed:
            print("⚠️ Soak test did not meet stability criteria")
    
    elif args.test == "full":
        # Full test suite
        await tester.test_engine_routing()
        await tester.run_concurrent_test("custom", args.requests, args.concurrent)
        await tester.run_concurrent_test("vllm", args.requests, args.concurrent)
        await tester.run_concurrent_test("auto", args.requests, args.concurrent)
        await tester.run_ab_test(min(args.duration, 60))
        
        if args.duration >= 300:
            await tester.run_soak_test(args.duration)
    
    # Print summary
    tester.print_summary()

if __name__ == "__main__":
    asyncio.run(main())