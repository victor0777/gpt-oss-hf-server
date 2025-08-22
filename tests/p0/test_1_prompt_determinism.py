#!/usr/bin/env python3
"""
Test 1: PR-PF01 - 프롬프트 결정론 테스트
동일한 입력이 동일한 프롬프트를 생성하는지 확인
"""

import requests
import json
import hashlib
import time
from typing import List, Dict

SERVER_URL = "http://localhost:8000"

def test_prompt_determinism():
    """동일 입력 → 동일 프롬프트 테스트"""
    print("\n🧪 Test 1: 프롬프트 결정론 테스트")
    print("="*50)
    
    # 테스트 메시지
    messages = [
        {"role": "user", "content": "What is 2+2?"}
    ]
    
    # 동일한 파라미터로 3번 요청
    request_data = {
        "model": "gpt-oss-20b",
        "messages": messages,
        "max_tokens": 10,
        "temperature": 0,  # 결정론적 생성
        "seed": 42  # 시드 고정
    }
    
    cache_keys = []
    responses = []
    
    print("\n📤 동일한 요청 3번 전송...")
    for i in range(3):
        try:
            response = requests.post(
                f"{SERVER_URL}/v1/chat/completions",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                metadata = data.get("metadata", {})
                cache_key = metadata.get("cache_key", "")
                
                if cache_key:
                    cache_keys.append(cache_key)
                    print(f"  요청 {i+1}: cache_key = {cache_key[:16]}...")
                    responses.append(data)
                else:
                    print(f"  ❌ 요청 {i+1}: cache_key 없음")
            else:
                print(f"  ❌ 요청 {i+1} 실패: {response.status_code}")
        
        except Exception as e:
            print(f"  ❌ 요청 {i+1} 에러: {e}")
            return False
    
    # 결과 분석
    print("\n📊 결과 분석:")
    
    # 1. 모든 cache_key가 동일한지 확인
    unique_keys = set(cache_keys)
    if len(unique_keys) == 1 and len(cache_keys) == 3:
        print(f"  ✅ 결정론적 생성 확인: 모든 요청이 동일한 cache_key 생성")
    else:
        print(f"  ❌ 결정론 실패: {len(unique_keys)}개의 서로 다른 cache_key")
        return False
    
    # 2. 프롬프트 버전 확인
    prompt_versions = [r.get("metadata", {}).get("prompt_version") for r in responses]
    if all(v for v in prompt_versions):
        print(f"  ✅ 프롬프트 버전 태깅: {prompt_versions[0]}")
    else:
        print(f"  ❌ 프롬프트 버전 누락")
    
    # 3. 캐시 히트 확인
    cache_hits = [r.get("metadata", {}).get("cache_hit", False) for r in responses]
    hit_count = sum(1 for hit in cache_hits if hit)
    print(f"  ℹ️ 캐시 히트: {hit_count}/3 요청")
    
    return True

def test_cache_hit_rate():
    """캐시 히트율 테스트"""
    print("\n🧪 Test 1.2: 캐시 히트율 테스트")
    print("="*50)
    
    # 동일한 요청 10번 반복
    messages = [{"role": "user", "content": "Hello, how are you?"}]
    request_data = {
        "model": "gpt-oss-20b",
        "messages": messages,
        "max_tokens": 20,
        "temperature": 0,
        "seed": 1
    }
    
    print("\n📤 동일한 요청 10번 전송...")
    cache_hits = 0
    
    for i in range(10):
        try:
            response = requests.post(
                f"{SERVER_URL}/v1/chat/completions",
                json=request_data,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("metadata", {}).get("cache_hit", False):
                    cache_hits += 1
                print(f"  요청 {i+1:2d}: {'✅ 캐시 히트' if data.get('metadata', {}).get('cache_hit') else '⭕ 캐시 미스'}")
        
        except Exception as e:
            print(f"  ❌ 요청 {i+1} 에러: {e}")
    
    # 통계 확인
    print("\n📊 캐시 통계 확인...")
    try:
        stats_response = requests.get(f"{SERVER_URL}/stats")
        if stats_response.status_code == 200:
            stats = stats_response.json()
            prompt_metrics = stats.get("prompt_metrics", {})
            
            if prompt_metrics:
                cache_hit_rate = prompt_metrics.get("cache_hit_rate", 0)
                print(f"  📈 전체 캐시 히트율: {cache_hit_rate:.1%}")
                print(f"  📊 캐시 히트: {prompt_metrics.get('cache_hits', 0)}")
                print(f"  📊 캐시 미스: {prompt_metrics.get('cache_misses', 0)}")
                print(f"  📊 캐시 크기: {prompt_metrics.get('cache_size', 0)}")
                
                if cache_hit_rate >= 0.3:
                    print(f"  ✅ 캐시 히트율 목표 달성 (≥30%)")
                    return True
                else:
                    print(f"  ⚠️ 캐시 히트율 부족 (<30%)")
                    return False
            else:
                print(f"  ⚠️ prompt_metrics 없음")
                return False
    
    except Exception as e:
        print(f"  ❌ 통계 조회 실패: {e}")
        return False
    
    print(f"  ⚠️ 통계 조회 실패")
    return False

def test_prompt_versioning():
    """프롬프트 버전 시스템 테스트"""
    print("\n🧪 Test 1.3: 프롬프트 버전 시스템")
    print("="*50)
    
    messages = [{"role": "user", "content": "Test version"}]
    
    # 기본 버전 확인
    print("\n📤 기본 프롬프트 버전 확인...")
    try:
        response = requests.post(
            f"{SERVER_URL}/v1/chat/completions",
            json={
                "model": "gpt-oss-20b",
                "messages": messages,
                "max_tokens": 10
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            version = data.get("metadata", {}).get("prompt_version")
            print(f"  ✅ 프롬프트 버전: {version}")
            
            # 모델 정보도 확인
            model_id = data.get("metadata", {}).get("model_id")
            if model_id:
                print(f"  ✅ 모델 ID: {model_id}")
            
            return True
        else:
            print(f"  ❌ 요청 실패: {response.status_code}")
    
    except Exception as e:
        print(f"  ❌ 에러: {e}")
    
    return False

def main():
    """메인 테스트 실행"""
    print("\n" + "="*60)
    print("🚀 PR-PF01: 프롬프트 빌더 테스트 시작")
    print("="*60)
    
    # 서버 상태 확인
    print("\n🔍 서버 상태 확인...")
    try:
        health_response = requests.get(f"{SERVER_URL}/health", timeout=5)
        if health_response.status_code == 200:
            health = health_response.json()
            print(f"  ✅ 서버 상태: {health.get('status', 'unknown')}")
            print(f"  📌 버전: {health.get('version', 'unknown')}")
            model_info = health.get('model', {})
            if model_info:
                print(f"  📌 모델: {model_info.get('model_id', 'unknown')}")
                print(f"  📌 dtype: {model_info.get('dtype', 'unknown')}")
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
    
    # Test 1: 결정론 테스트
    result1 = test_prompt_determinism()
    results.append(("프롬프트 결정론", result1))
    time.sleep(1)
    
    # Test 2: 캐시 히트율
    result2 = test_cache_hit_rate()
    results.append(("캐시 히트율", result2))
    time.sleep(1)
    
    # Test 3: 버전 시스템
    result3 = test_prompt_versioning()
    results.append(("프롬프트 버전", result3))
    
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
        print("\n🎉 모든 프롬프트 빌더 테스트 통과!")
    else:
        print("\n⚠️ 일부 테스트 실패. 로그를 확인하세요.")

if __name__ == "__main__":
    main()