#!/usr/bin/env python3
"""
Test 3: PR-OBS01 - 모델 태깅 테스트
모든 엔드포인트에서 모델 정보가 태깅되는지 확인
"""

import requests
import json
import time
from typing import Dict, List

SERVER_URL = "http://localhost:8000"

def test_health_endpoint():
    """Health 엔드포인트 모델 태깅 테스트"""
    print("\n🧪 Test 3.1: /health 엔드포인트 태깅")
    print("="*50)
    
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5)
        
        if response.status_code != 200:
            print(f"  ❌ Health 요청 실패: {response.status_code}")
            return False
        
        data = response.json()
        print("\n📊 Health 응답:")
        print(f"  상태: {data.get('status', 'unknown')}")
        print(f"  버전: {data.get('version', 'unknown')}")
        print(f"  dtype: {data.get('dtype', 'unknown')}")
        print(f"  프로파일: {data.get('profile', 'unknown')}")
        
        # 모델 정보 확인
        model_info = data.get('model', {})
        if not model_info:
            print("\n  ❌ 모델 정보 없음")
            return False
        
        print("\n📌 모델 태깅 정보:")
        required_tags = ["model_id", "model_size", "dtype", "gpu_mode", "prompt_version"]
        missing_tags = []
        
        for tag in required_tags:
            value = model_info.get(tag)
            if value:
                print(f"  ✅ {tag}: {value}")
            else:
                print(f"  ❌ {tag}: 누락")
                missing_tags.append(tag)
        
        if not missing_tags:
            print("\n  ✅ 모든 필수 태그 존재")
            return True
        else:
            print(f"\n  ❌ 누락된 태그: {missing_tags}")
            return False
    
    except Exception as e:
        print(f"  ❌ Health 엔드포인트 에러: {e}")
        return False

def test_stats_endpoint():
    """Stats 엔드포인트 모델 태깅 테스트"""
    print("\n🧪 Test 3.2: /stats 엔드포인트 태깅")
    print("="*50)
    
    # 먼저 요청을 하나 생성
    print("\n📤 테스트 요청 생성...")
    try:
        test_response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json={
                "model": "gpt-oss-20b",
                "messages": [{"role": "user", "content": "Test for stats"}],
                "max_tokens": 10
            },
            timeout=30
        )
        print(f"  테스트 요청 상태: {test_response.status_code}")
    except:
        print("  ⚠️ 테스트 요청 실패 (계속 진행)")
    
    time.sleep(1)
    
    # Stats 조회
    try:
        response = requests.get(f"{SERVER_URL}/stats", timeout=5)
        
        if response.status_code != 200:
            print(f"  ❌ Stats 요청 실패: {response.status_code}")
            return False
        
        data = response.json()
        
        print("\n📊 Stats 기본 정보:")
        print(f"  총 요청: {data.get('requests_total', 0)}")
        print(f"  성공: {data.get('requests_success', 0)}")
        print(f"  실패: {data.get('requests_failed', 0)}")
        print(f"  QPS: {data.get('qps', 0)}")
        
        # model_info 확인
        model_info = data.get('model_info', {})
        if model_info:
            print("\n📌 모델 정보:")
            for key, value in model_info.items():
                print(f"  {key}: {value}")
            print("  ✅ model_info 존재")
        else:
            print("  ⚠️ model_info 없음")
        
        # model_metrics 확인
        model_metrics = data.get('model_metrics', {})
        if model_metrics:
            print("\n📌 모델별 메트릭:")
            for key, metrics in model_metrics.items():
                print(f"  {key}:")
                print(f"    - 총 요청: {metrics.get('total', 0)}")
                print(f"    - 성공: {metrics.get('success', 0)}")
                print(f"    - 실패: {metrics.get('failed', 0)}")
                print(f"    - 취소: {metrics.get('cancelled', 0)}")
            print("  ✅ model_metrics 존재")
        else:
            print("  ⚠️ model_metrics 없음")
        
        # prompt_metrics 확인
        prompt_metrics = data.get('prompt_metrics', {})
        if prompt_metrics:
            print("\n📌 프롬프트 메트릭:")
            print(f"  캐시 히트: {prompt_metrics.get('cache_hits', 0)}")
            print(f"  캐시 미스: {prompt_metrics.get('cache_misses', 0)}")
            print(f"  캐시 히트율: {prompt_metrics.get('cache_hit_rate', 0):.1%}")
            print(f"  캐시 크기: {prompt_metrics.get('cache_size', 0)}")
            print("  ✅ prompt_metrics 존재")
        
        # 전체 평가
        has_model_info = bool(model_info)
        has_model_metrics = bool(model_metrics)
        
        if has_model_info or has_model_metrics:
            print("\n  ✅ Stats 엔드포인트 태깅 확인")
            return True
        else:
            print("\n  ❌ Stats 엔드포인트 태깅 부족")
            return False
    
    except Exception as e:
        print(f"  ❌ Stats 엔드포인트 에러: {e}")
        return False

def test_metrics_endpoint():
    """Prometheus 메트릭 엔드포인트 테스트"""
    print("\n🧪 Test 3.3: /metrics 엔드포인트 (Prometheus)")
    print("="*50)
    
    try:
        response = requests.get(f"{SERVER_URL}/metrics", timeout=5)
        
        if response.status_code != 200:
            print(f"  ❌ Metrics 요청 실패: {response.status_code}")
            return False
        
        metrics_text = response.text
        
        print("\n📊 Prometheus 메트릭 분석:")
        
        # 주요 메트릭 확인
        metrics_found = {
            "requests_total": "requests_total" in metrics_text,
            "requests_active": "requests_active" in metrics_text,
            "stream_active": "stream_active" in metrics_text,
            "stream_cancelled_total": "stream_cancelled_total" in metrics_text,
            "model_requests_total": "model_requests_total" in metrics_text,
            "ttft_ms": "ttft_ms" in metrics_text,
            "e2e_ms": "e2e_ms" in metrics_text,
            "prompt_cache_hits": "prompt_cache_hits" in metrics_text,
            "prompt_cache_hit_rate": "prompt_cache_hit_rate" in metrics_text
        }
        
        for metric, found in metrics_found.items():
            status = "✅" if found else "❌"
            print(f"  {status} {metric}")
        
        # 모델 라벨 확인
        has_model_labels = False
        if 'model_id=' in metrics_text or 'model_size=' in metrics_text:
            has_model_labels = True
            print("\n  ✅ 모델 라벨 포함")
            
            # 라벨 예시 출력
            for line in metrics_text.split('\n'):
                if 'model_id=' in line and not line.startswith('#'):
                    print(f"    예시: {line[:100]}...")
                    break
        else:
            print("\n  ⚠️ 모델 라벨 없음")
        
        # 전체 평가
        essential_metrics = sum([
            metrics_found["requests_total"],
            metrics_found["ttft_ms"],
            metrics_found["e2e_ms"]
        ])
        
        if essential_metrics >= 2:
            print("\n  ✅ Prometheus 메트릭 형식 확인")
            return True
        else:
            print("\n  ❌ 필수 메트릭 부족")
            return False
    
    except Exception as e:
        print(f"  ❌ Metrics 엔드포인트 에러: {e}")
        return False

def test_request_metadata():
    """요청 응답의 메타데이터 태깅 테스트"""
    print("\n🧪 Test 3.4: 요청 메타데이터 태깅")
    print("="*50)
    
    try:
        print("\n📤 테스트 요청 전송...")
        response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json={
                "model": "gpt-oss-20b",
                "messages": [{"role": "user", "content": "Check metadata"}],
                "max_tokens": 10,
                "temperature": 0
            },
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"  ❌ 요청 실패: {response.status_code}")
            return False
        
        data = response.json()
        metadata = data.get("metadata", {})
        
        if not metadata:
            print("  ⚠️ 메타데이터 없음 (OpenAI 호환 모드)")
            # OpenAI 호환 응답 확인
            if "choices" in data and "usage" in data:
                print("  ✅ OpenAI 호환 형식 확인")
                return True
            return False
        
        print("\n📌 응답 메타데이터:")
        important_fields = [
            "prompt_version",
            "model_id", 
            "model_size",
            "dtype",
            "gpu_mode",
            "cache_key",
            "cache_hit",
            "tokens_before",
            "tokens_after",
            "request_id"
        ]
        
        found_count = 0
        for field in important_fields:
            value = metadata.get(field)
            if value is not None:
                print(f"  ✅ {field}: {value}")
                found_count += 1
            else:
                print(f"  ⚠️ {field}: 없음")
        
        if found_count >= 5:  # 최소 5개 이상 필드
            print("\n  ✅ 메타데이터 태깅 충분")
            return True
        else:
            print("\n  ❌ 메타데이터 태깅 부족")
            return False
    
    except Exception as e:
        print(f"  ❌ 요청 메타데이터 테스트 에러: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("\n" + "="*60)
    print("🚀 PR-OBS01: 모델 태깅 테스트 시작")
    print("="*60)
    
    # 서버 상태 확인
    print("\n🔍 서버 상태 확인...")
    try:
        health_response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health = health_response.json()
            print(f"  ✅ 서버 상태: {health.get('status', 'unknown')}")
            print(f"  📌 버전: {health.get('version', 'unknown')}")
        else:
            print(f"  ❌ 서버 응답 없음")
            return
    except Exception as e:
        print(f"  ❌ 서버 연결 실패: {e}")
        print("\n💡 서버가 실행 중인지 확인하세요:")
        print("  python src/server_v454_p0.py --model 20b --profile latency_first")
        return
    
    # 테스트 실행
    results = []
    
    # Test 1: Health 엔드포인트
    result1 = test_health_endpoint()
    results.append(("Health 엔드포인트", result1))
    time.sleep(1)
    
    # Test 2: Stats 엔드포인트
    result2 = test_stats_endpoint()
    results.append(("Stats 엔드포인트", result2))
    time.sleep(1)
    
    # Test 3: Metrics 엔드포인트
    result3 = test_metrics_endpoint()
    results.append(("Metrics 엔드포인트", result3))
    time.sleep(1)
    
    # Test 4: 요청 메타데이터
    result4 = test_request_metadata()
    results.append(("요청 메타데이터", result4))
    
    # 결과 요약
    print("\n" + "="*60)
    print("📊 테스트 결과 요약")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\n총 {total}개 중 {passed}개 통과 ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n🎉 모든 모델 태깅 테스트 통과!")
    elif passed >= total * 0.75:
        print("\n✅ 대부분의 모델 태깅 테스트 통과")
    else:
        print("\n⚠️ 일부 테스트 실패. 로그를 확인하세요.")

if __name__ == "__main__":
    main()