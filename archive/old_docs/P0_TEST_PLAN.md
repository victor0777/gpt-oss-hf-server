# P0 구현 테스트 계획서

## 📋 테스트 개요

**목적**: GPT-OSS HF Server v4.5.4-P0의 모든 기능 검증  
**버전**: v4.5.4-P0  
**테스트 일자**: 2025-08-21

## 🎯 테스트 목표

1. PromptBuilder의 결정론적 동작 확인
2. SSE 스트리밍 안정성 검증
3. 모델 태깅 완전성 확인
4. 성능 목표 달성 여부 검증

## 📊 테스트 매트릭스

| 영역 | 테스트 항목 | 우선순위 | 예상 시간 |
|------|------------|----------|-----------|
| PR-PF01 | 프롬프트 결정론 | P0 | 1분 |
| PR-PF01 | 캐시 히트율 | P0 | 2분 |
| PR-PF01 | 프롬프트 버전 태깅 | P0 | 1분 |
| PR-ST01 | SSE 포맷 검증 | P0 | 2분 |
| PR-ST01 | 스트림 취소 | P0 | 3분 |
| PR-OBS01 | 모델 태깅 (health) | P0 | 1분 |
| PR-OBS01 | 모델 태깅 (stats) | P0 | 1분 |
| PR-OBS01 | Prometheus 메트릭 | P0 | 1분 |
| 성능 | LATENCY_FIRST 프로파일 | P0 | 5분 |
| 성능 | 메모리 누수 확인 | P1 | 10분 |

## 🚀 테스트 준비

### 1. 환경 설정
```bash
cd /home/ktl/gpt-oss-hf-server
source .venv/bin/activate  # 가상환경 활성화

# 필요 패키지 확인
pip install aiohttp fastapi uvicorn transformers torch

# GPU 상태 확인
nvidia-smi
```

### 2. 서버 시작
```bash
# 터미널 1: 서버 실행
python src/server_v454_p0.py --model 20b --profile latency_first --port 8000

# 또는 백그라운드 실행
nohup python src/server_v454_p0.py --model 20b --profile latency_first --port 8000 > server.log 2>&1 &
```

### 3. 서버 상태 확인
```bash
# 헬스 체크
curl http://localhost:8000/health | python -m json.tool
```

## 📝 테스트 시나리오

### Test 1: PR-PF01 - 프롬프트 결정론 테스트

**목적**: 동일 입력이 동일한 프롬프트를 생성하는지 확인

**수용 기준**:
- 동일 입력 → 동일 cache_key
- 메타데이터에 prompt_version 포함
- 캐시 히트율 ≥ 30%

**테스트 스크립트**: `test_1_prompt_determinism.py`

---

### Test 2: PR-ST01 - SSE 스트리밍 테스트

**목적**: 스트리밍이 올바른 SSE 포맷으로 동작하는지 확인

**수용 기준**:
- SSE 이벤트: start, token, done
- 취소 시 리소스 정리
- 메모리 누수 없음

**테스트 스크립트**: `test_2_sse_streaming.py`

---

### Test 3: PR-OBS01 - 모델 태깅 테스트

**목적**: 모든 엔드포인트에서 모델 정보가 태깅되는지 확인

**수용 기준**:
- /health에 model_id, model_size, dtype 표시
- /stats에 model_metrics 포함
- /metrics에 라벨 포함

**테스트 스크립트**: `test_3_model_tagging.py`

---

### Test 4: 성능 테스트

**목적**: LATENCY_FIRST 프로파일 성능 목표 달성 확인

**수용 기준**:
- p95 TTFT ≤ 7초
- p95 E2E ≤ 20초
- 에러율 < 0.5%

**테스트 스크립트**: `test_4_performance.py`

---

### Test 5: 부하 테스트 (선택)

**목적**: 연속 요청 처리 및 안정성 확인

**테스트 스크립트**: `test_5_load.py`

## 🔍 수동 테스트 체크리스트

### 기본 기능 테스트
- [ ] 서버 시작 성공
- [ ] /health 응답 정상
- [ ] 간단한 요청 처리
- [ ] 스트리밍 요청 처리

### 프롬프트 빌더 테스트
- [ ] 동일 입력 → 동일 결과
- [ ] 캐시 동작 확인
- [ ] 버전 태깅 확인

### 스트리밍 테스트
- [ ] SSE 포맷 정확성
- [ ] 브라우저 탭 닫기 시 취소
- [ ] 긴 응답 중간 취소

### 모니터링 테스트
- [ ] /stats에 모든 메트릭 표시
- [ ] /metrics Prometheus 호환
- [ ] 모델별 메트릭 분리

## 📈 예상 결과

### 성공 기준
- 모든 P0 테스트 통과
- 캐시 히트율 ≥ 30%
- p95 응답 시간 < 7초
- 에러율 < 0.5%

### 실패 시 대응
1. 서버 로그 확인: `tail -f server.log`
2. GPU 메모리 확인: `nvidia-smi`
3. 프로세스 확인: `ps aux | grep server_v454`
4. 포트 확인: `netstat -an | grep 8000`

## 💾 결과 기록

테스트 완료 후 다음 정보 기록:

```markdown
테스트 일시: [날짜/시간]
서버 버전: v4.5.4-P0
환경:
- GPU: [모델]
- CUDA: [버전]
- PyTorch: [버전]
- NumPy: [버전]

결과:
- Test 1 (프롬프트): [PASS/FAIL] - [메시지]
- Test 2 (스트리밍): [PASS/FAIL] - [메시지]
- Test 3 (태깅): [PASS/FAIL] - [메시지]
- Test 4 (성능): [PASS/FAIL] - [메시지]
- Test 5 (부하): [PASS/FAIL] - [메시지]

특이사항:
[기록]
```

## 🚨 문제 해결 가이드

### 서버 시작 실패
```bash
# 포트 사용 중인지 확인
lsof -i :8000
# 기존 프로세스 종료
pkill -f server_v454

# 권한 문제
chmod +x run_p0_tests.sh
```

### 캐시 히트율 낮음
```bash
# 동일한 요청 반복 필요
# temperature=0, seed 고정 필요
```

### 스트리밍 실패
```bash
# curl에 -N 옵션 필요
curl -N -X POST ...
```

## 📞 지원

문제 발생 시:
1. server.log 확인
2. GPU 상태 확인
3. 메모리 사용량 확인
4. 네트워크 연결 확인