#!/usr/bin/env python3
"""
Release Gate Validation Script for v4.5
Automated testing with SLO checks and gate criteria
"""

import asyncio
import aiohttp
import time
import json
import sys
import argparse
import statistics
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

class TestPhase(Enum):
    STEP = "step"
    SPIKE = "spike"
    SOAK = "soak"

@dataclass
class SLOTargets:
    """Service Level Objectives"""
    p95_ttft_ms: float = 7000  # Time to first token
    p95_e2e_ms: float = 20000  # End-to-end latency
    error_rate: float = 0.005  # 0.5% error rate
    target_qps: float = 2.0  # Target queries per second
    gpu_memory_pct: float = 85  # GPU memory utilization
    queue_length: int = 10  # Max queue length

@dataclass
class TestResult:
    """Test phase result"""
    phase: TestPhase
    duration_seconds: float
    requests_total: int
    requests_success: int
    requests_failed: int
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    actual_qps: float
    error_rate: float
    slo_violations: List[str]
    passed: bool
    
    def print_summary(self):
        """Print test result summary"""
        status = "✅ PASSED" if self.passed else "❌ FAILED"
        print(f"\n{'='*60}")
        print(f"{self.phase.value.upper()} Test Results - {status}")
        print(f"{'='*60}")
        print(f"Duration: {self.duration_seconds:.1f}s")
        print(f"Total Requests: {self.requests_total}")
        print(f"Success Rate: {(1-self.error_rate)*100:.1f}%")
        print(f"Actual QPS: {self.actual_qps:.2f}")
        print(f"P50 Latency: {self.p50_latency_ms:.0f}ms")
        print(f"P95 Latency: {self.p95_latency_ms:.0f}ms")
        print(f"P99 Latency: {self.p99_latency_ms:.0f}ms")
        
        if self.slo_violations:
            print(f"\nSLO Violations:")
            for violation in self.slo_violations:
                print(f"  ⚠️ {violation}")

class ReleaseGateValidator:
    """Release gate validation orchestrator"""
    
    def __init__(self, base_url: str = "http://localhost:8000", slo: SLOTargets = None):
        self.base_url = base_url
        self.slo = slo or SLOTargets()
        self.session = None
        self.canary_percentage = 10  # Start with 10% canary
        
    async def _ensure_session(self):
        if self.session is None:
            self.session = aiohttp.ClientSession()
    
    async def _make_request(self, payload: dict) -> Tuple[bool, float, Optional[dict]]:
        """Make a single request to the server"""
        await self._ensure_session()
        
        start = time.time()
        try:
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as resp:
                latency = (time.time() - start) * 1000
                if resp.status == 200:
                    data = await resp.json()
                    return True, latency, data
                else:
                    return False, latency, None
        except Exception as e:
            latency = (time.time() - start) * 1000
            return False, latency, None
    
    async def check_health(self) -> dict:
        """Check server health"""
        await self._ensure_session()
        
        try:
            async with self.session.get(f"{self.base_url}/health") as resp:
                return await resp.json()
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def run_step_test(self, steps: List[int] = None) -> TestResult:
        """Step load test: gradually increase load"""
        if steps is None:
            steps = [1, 2, 4, 8, 16]  # Concurrent requests per step
        
        print(f"\n{'='*60}")
        print(f"STEP Test: Testing load steps {steps}")
        print(f"{'='*60}")
        
        all_latencies = []
        total_requests = 0
        total_errors = 0
        start_time = time.time()
        
        for step in steps:
            print(f"\nStep: {step} concurrent requests")
            
            # Run requests for this step
            tasks = []
            for i in range(step * 5):  # 5 requests per concurrent level
                payload = {
                    "model": "gpt-oss-20b",
                    "messages": [{"role": "user", "content": f"Step test {step}-{i}"}],
                    "max_tokens": 50,
                    "temperature": 0.7
                }
                tasks.append(self._make_request(payload))
            
            # Limit concurrency
            semaphore = asyncio.Semaphore(step)
            async def limited_request(task):
                async with semaphore:
                    return await task
            
            results = await asyncio.gather(*[limited_request(t) for t in tasks])
            
            # Process results
            step_latencies = []
            step_errors = 0
            for success, latency, _ in results:
                total_requests += 1
                if success:
                    all_latencies.append(latency)
                    step_latencies.append(latency)
                else:
                    total_errors += 1
                    step_errors += 1
            
            # Print step summary
            if step_latencies:
                p95 = statistics.quantiles(step_latencies, n=20)[18]
                error_rate = step_errors / len(results)
                print(f"  P95: {p95:.0f}ms, Errors: {error_rate:.1%}")
                
                # Check if we should stop (too many errors)
                if error_rate > 0.1:  # >10% errors
                    print(f"  ⚠️ High error rate, stopping step test")
                    break
        
        # Calculate overall metrics
        duration = time.time() - start_time
        error_rate = total_errors / total_requests if total_requests > 0 else 0
        qps = total_requests / duration if duration > 0 else 0
        
        if all_latencies:
            p50 = statistics.median(all_latencies)
            p95 = statistics.quantiles(all_latencies, n=20)[18]
            p99 = statistics.quantiles(all_latencies, n=100)[98]
        else:
            p50 = p95 = p99 = 0
        
        # Check SLO violations
        violations = []
        if p95 > self.slo.p95_e2e_ms:
            violations.append(f"P95 latency {p95:.0f}ms > {self.slo.p95_e2e_ms}ms")
        if error_rate > self.slo.error_rate:
            violations.append(f"Error rate {error_rate:.1%} > {self.slo.error_rate:.1%}")
        
        passed = len(violations) == 0
        
        return TestResult(
            phase=TestPhase.STEP,
            duration_seconds=duration,
            requests_total=total_requests,
            requests_success=total_requests - total_errors,
            requests_failed=total_errors,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            actual_qps=qps,
            error_rate=error_rate,
            slo_violations=violations,
            passed=passed
        )
    
    async def run_spike_test(self, spike_qps: float = 10.0, duration: int = 30) -> TestResult:
        """Spike test: sudden high load"""
        print(f"\n{'='*60}")
        print(f"SPIKE Test: {spike_qps} QPS for {duration}s")
        print(f"{'='*60}")
        
        all_latencies = []
        total_requests = 0
        total_errors = 0
        start_time = time.time()
        
        # Calculate request interval
        interval = 1.0 / spike_qps
        
        while time.time() - start_time < duration:
            request_start = time.time()
            
            # Send request
            payload = {
                "model": "gpt-oss-20b",
                "messages": [{"role": "user", "content": f"Spike test {total_requests}"}],
                "max_tokens": 30,
                "temperature": 0.7
            }
            
            success, latency, _ = await self._make_request(payload)
            total_requests += 1
            
            if success:
                all_latencies.append(latency)
            else:
                total_errors += 1
            
            # Progress update
            if total_requests % 50 == 0:
                elapsed = time.time() - start_time
                actual_qps = total_requests / elapsed
                print(f"  [{int(elapsed)}s] Requests: {total_requests}, QPS: {actual_qps:.2f}")
            
            # Maintain target QPS
            elapsed = time.time() - request_start
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)
        
        # Calculate metrics
        duration = time.time() - start_time
        error_rate = total_errors / total_requests if total_requests > 0 else 0
        qps = total_requests / duration if duration > 0 else 0
        
        if all_latencies:
            p50 = statistics.median(all_latencies)
            p95 = statistics.quantiles(all_latencies, n=20)[18]
            p99 = statistics.quantiles(all_latencies, n=100)[98]
        else:
            p50 = p95 = p99 = 0
        
        # Check SLO violations (relaxed for spike)
        violations = []
        if p95 > self.slo.p95_e2e_ms * 1.5:  # Allow 50% higher latency during spike
            violations.append(f"P95 latency {p95:.0f}ms > {self.slo.p95_e2e_ms*1.5}ms (spike threshold)")
        if error_rate > self.slo.error_rate * 2:  # Allow 2x error rate during spike
            violations.append(f"Error rate {error_rate:.1%} > {self.slo.error_rate*2:.1%} (spike threshold)")
        
        passed = len(violations) == 0
        
        return TestResult(
            phase=TestPhase.SPIKE,
            duration_seconds=duration,
            requests_total=total_requests,
            requests_success=total_requests - total_errors,
            requests_failed=total_errors,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            actual_qps=qps,
            error_rate=error_rate,
            slo_violations=violations,
            passed=passed
        )
    
    async def run_soak_test(self, target_qps: float = 2.0, duration: int = 1800) -> TestResult:
        """Soak test: sustained load for extended period"""
        print(f"\n{'='*60}")
        print(f"SOAK Test: {target_qps} QPS for {duration}s ({duration/60:.0f} minutes)")
        print(f"{'='*60}")
        
        all_latencies = []
        window_results = []  # Track per-minute windows
        total_requests = 0
        total_errors = 0
        start_time = time.time()
        
        # Window tracking
        window_size = 60  # 1 minute windows
        current_window = {"start": start_time, "latencies": [], "errors": 0, "requests": 0}
        
        # Calculate request interval
        interval = 1.0 / target_qps
        
        while time.time() - start_time < duration:
            # Check if we need a new window
            if time.time() - current_window["start"] >= window_size:
                window_results.append(current_window)
                current_window = {"start": time.time(), "latencies": [], "errors": 0, "requests": 0}
            
            request_start = time.time()
            
            # Send request
            payload = {
                "model": "gpt-oss-20b",
                "messages": [{"role": "user", "content": f"Soak test {total_requests}"}],
                "max_tokens": 50,
                "temperature": 0.7
            }
            
            success, latency, _ = await self._make_request(payload)
            total_requests += 1
            current_window["requests"] += 1
            
            if success:
                all_latencies.append(latency)
                current_window["latencies"].append(latency)
            else:
                total_errors += 1
                current_window["errors"] += 1
            
            # Progress update
            if total_requests % 100 == 0:
                elapsed = time.time() - start_time
                actual_qps = total_requests / elapsed
                recent_errors = sum(w["errors"] for w in window_results[-5:]) if len(window_results) >= 5 else 0
                print(f"  [{int(elapsed)}s] Requests: {total_requests}, QPS: {actual_qps:.2f}, "
                      f"Recent Errors (5min): {recent_errors}")
                
                # Early termination if too many errors
                if current_window["errors"] > 20:
                    print("  ⚠️ High error rate in current window, terminating soak test")
                    break
            
            # Maintain target QPS
            elapsed = time.time() - request_start
            if elapsed < interval:
                await asyncio.sleep(interval - elapsed)
        
        # Add final window
        if current_window["requests"] > 0:
            window_results.append(current_window)
        
        # Calculate overall metrics
        duration = time.time() - start_time
        error_rate = total_errors / total_requests if total_requests > 0 else 0
        qps = total_requests / duration if duration > 0 else 0
        
        if all_latencies:
            p50 = statistics.median(all_latencies)
            p95 = statistics.quantiles(all_latencies, n=20)[18]
            p99 = statistics.quantiles(all_latencies, n=100)[98]
        else:
            p50 = p95 = p99 = 0
        
        # Analyze window stability
        stable_windows = 0
        for window in window_results:
            if window["latencies"]:
                window_p95 = statistics.quantiles(window["latencies"], n=20)[18]
                window_error_rate = window["errors"] / window["requests"] if window["requests"] > 0 else 0
                
                # Check if window meets SLO
                if window_p95 < self.slo.p95_e2e_ms and window_error_rate < self.slo.error_rate:
                    stable_windows += 1
        
        stability_rate = stable_windows / len(window_results) if window_results else 0
        
        # Check SLO violations
        violations = []
        if p95 > self.slo.p95_e2e_ms:
            violations.append(f"P95 latency {p95:.0f}ms > {self.slo.p95_e2e_ms}ms")
        if error_rate > self.slo.error_rate:
            violations.append(f"Error rate {error_rate:.1%} > {self.slo.error_rate:.1%}")
        if qps < self.slo.target_qps * 0.9:  # Allow 10% variance
            violations.append(f"QPS {qps:.2f} < {self.slo.target_qps*0.9:.2f} (90% of target)")
        if stability_rate < 0.95:
            violations.append(f"Stability {stability_rate:.1%} < 95% (only {stable_windows}/{len(window_results)} stable windows)")
        
        passed = len(violations) == 0
        
        print(f"\n  Window Analysis: {stable_windows}/{len(window_results)} windows met SLO ({stability_rate:.1%})")
        
        return TestResult(
            phase=TestPhase.SOAK,
            duration_seconds=duration,
            requests_total=total_requests,
            requests_success=total_requests - total_errors,
            requests_failed=total_errors,
            p50_latency_ms=p50,
            p95_latency_ms=p95,
            p99_latency_ms=p99,
            actual_qps=qps,
            error_rate=error_rate,
            slo_violations=violations,
            passed=passed
        )
    
    async def run_canary_promotion(self, test_results: List[TestResult]):
        """Simulate canary promotion based on test results"""
        print(f"\n{'='*60}")
        print(f"CANARY PROMOTION DECISION")
        print(f"{'='*60}")
        
        all_passed = all(r.passed for r in test_results)
        
        if all_passed:
            print(f"✅ All tests passed!")
            print(f"\nCanary Promotion Path:")
            print(f"  Current: {self.canary_percentage}% → 50% → 100%")
            
            # Simulate promotion
            if self.canary_percentage == 10:
                print(f"  Action: Promote to 50% canary")
                self.canary_percentage = 50
            elif self.canary_percentage == 50:
                print(f"  Action: Promote to 100% (full rollout)")
                self.canary_percentage = 100
            else:
                print(f"  Action: Already at full rollout")
        else:
            print(f"❌ Some tests failed!")
            print(f"  Action: Hold at {self.canary_percentage}% canary")
            print(f"  Recommendation: Fix issues before promotion")
    
    async def run_full_gate(self, quick: bool = False) -> bool:
        """Run full release gate validation"""
        print(f"\n{'='*60}")
        print(f"RELEASE GATE VALIDATION - v4.5")
        print(f"Mode: {'Quick' if quick else 'Full'}")
        print(f"{'='*60}")
        
        # Check health first
        health = await self.check_health()
        print(f"\nServer Health: {health.get('status', 'unknown')}")
        print(f"Health Score: {health.get('score', 0)}")
        
        if health.get('status') == 'unhealthy':
            print("❌ Server is unhealthy, cannot proceed with tests")
            return False
        
        results = []
        
        # Step test
        step_result = await self.run_step_test([1, 2, 4] if quick else [1, 2, 4, 8, 16])
        step_result.print_summary()
        results.append(step_result)
        
        # Spike test
        spike_duration = 30 if quick else 60
        spike_result = await self.run_spike_test(spike_qps=5.0, duration=spike_duration)
        spike_result.print_summary()
        results.append(spike_result)
        
        # Soak test
        soak_duration = 60 if quick else 300  # 1 min quick, 5 min normal
        soak_result = await self.run_soak_test(target_qps=2.0, duration=soak_duration)
        soak_result.print_summary()
        results.append(soak_result)
        
        # Canary promotion decision
        await self.run_canary_promotion(results)
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"FINAL GATE DECISION")
        print(f"{'='*60}")
        
        all_passed = all(r.passed for r in results)
        if all_passed:
            print("✅ RELEASE GATE: PASSED")
            print("The release is approved for deployment")
        else:
            print("❌ RELEASE GATE: FAILED")
            print("The release needs fixes before deployment")
            print("\nFailed Tests:")
            for r in results:
                if not r.passed:
                    print(f"  - {r.phase.value}: {', '.join(r.slo_violations)}")
        
        return all_passed
    
    async def cleanup(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()

async def main():
    parser = argparse.ArgumentParser(description="Release Gate Validation")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument("--quick", action="store_true", help="Run quick validation")
    parser.add_argument("--phase", choices=["step", "spike", "soak", "full"],
                       default="full", help="Test phase to run")
    
    # SLO overrides
    parser.add_argument("--slo-p95-ttft", type=float, default=7000,
                       help="SLO P95 TTFT in ms")
    parser.add_argument("--slo-p95-e2e", type=float, default=20000,
                       help="SLO P95 E2E latency in ms")
    parser.add_argument("--slo-error-rate", type=float, default=0.005,
                       help="SLO error rate (0.005 = 0.5%)")
    parser.add_argument("--slo-qps", type=float, default=2.0,
                       help="SLO target QPS")
    
    args = parser.parse_args()
    
    # Create SLO targets
    slo = SLOTargets(
        p95_ttft_ms=args.slo_p95_ttft,
        p95_e2e_ms=args.slo_p95_e2e,
        error_rate=args.slo_error_rate,
        target_qps=args.slo_qps
    )
    
    # Create validator
    validator = ReleaseGateValidator(args.url, slo)
    
    try:
        if args.phase == "full":
            passed = await validator.run_full_gate(quick=args.quick)
            sys.exit(0 if passed else 1)
        else:
            # Run specific phase
            if args.phase == "step":
                result = await validator.run_step_test()
            elif args.phase == "spike":
                result = await validator.run_spike_test()
            elif args.phase == "soak":
                result = await validator.run_soak_test()
            
            result.print_summary()
            sys.exit(0 if result.passed else 1)
    finally:
        await validator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())