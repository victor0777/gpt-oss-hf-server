# GPT-OSS HF Server v4.5.1 - 테스트 절차

## 📋 테스트 전 준비사항

### 환경 확인
```bash
# GPU 상태 확인
nvidia-smi

# 포트 사용 확인
lsof -i :8000
lsof -i :8001

# 기존 프로세스 정리
pkill -f "server_v451"
pkill -f "server.py"
```

## 🧪 Test Suite 1: 20B 모델 테스트

### Step 1. 서버 시작
```bash
cd /home/ktl/gpt-oss-hf-server
python src/server_v451.py --engine custom --profile latency_first --model 20b --port 8000
```

### Step 2. 헬스 체크
```bash
curl -s http://localhost:8000/health | python -m json.tool
```

**확인 항목**:
- `status`: "healthy"
- `model_info.current_model`: "20b"
- `model_info.gpu_mode`: "pipeline"
- `model_info.default_profile`: "latency_first"

### Step 3. 기본 요청 테스트
```bash
# Non-streaming 요청
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Write a Python hello world function"}],
    "max_tokens": 100,
    "temperature": 0.7
  }' | python -m json.tool
```

**기대 결과**:
- 응답 시간: 4-8초
- `chosen_engine`: "custom"
- `profile`: "latency_first"

### Step 4. 프로파일 전환 테스트
```bash
# QUALITY_FIRST 프로파일로 전환
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Explain recursion with examples"}],
    "max_tokens": 200,
    "profile": "quality_first",
    "temperature": 0.8
  }' | python -m json.tool
```

**기대 결과**:
- `profile`: "quality_first"
- `max_tokens`: 최대 2048까지 가능

### Step 5. 스트리밍 테스트 (버그 확인)
```bash
# 스트리밍 요청 - 현재 버그로 500 에러 예상
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Count from 1 to 5"}],
    "max_tokens": 30,
    "stream": true
  }'
```

**예상 결과**:
- HTTP 500 에러
- 에러 메시지: "object async_generator can't be used in 'await' expression"

### Step 6. 메트릭 확인
```bash
# 통계 조회
curl -s http://localhost:8000/stats | python -m json.tool
```

**확인 항목**:
- `requests_total`: 요청 횟수
- `error_rate`: <0.3 (스트리밍 제외)
- `p95_ttft_ms`: <10000
- `model_usage`: "gpt-oss-20b" 카운트

### Step 7. GPU 메모리 확인
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
```

**기대값**:
- 각 GPU당 ~12.8GB 사용

## 🧪 Test Suite 2: 120B 모델 테스트

### Step 1. 서버 재시작 (120B)
```bash
# 기존 서버 종료
pkill -f "server_v451"

# 120B 모델로 시작
cd /home/ktl/gpt-oss-hf-server
python src/server_v451.py --engine custom --profile quality_first --model 120b --port 8000
```

### Step 2. 헬스 체크
```bash
curl -s http://localhost:8000/health | python -m json.tool
```

**확인 항목**:
- `model_info.current_model`: "120b"
- `model_info.gpu_mode`: "tensor"
- `model_info.default_profile`: "quality_first"

### Step 3. 품질 테스트
```bash
# 복잡한 추론 요청
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "Explain the differences between TCP and UDP protocols with real-world examples"}],
    "max_tokens": 300,
    "temperature": 0.7,
    "profile": "quality_first"
  }' | python -m json.tool
```

**기대 결과**:
- 응답 시간: 6-15초
- 더 상세하고 구조화된 답변
- `profile`: "quality_first"

### Step 4. 프로파일 비교
```bash
# LATENCY_FIRST로 전환
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "What is Docker?"}],
    "max_tokens": 100,
    "profile": "latency_first"
  }' --max-time 20 | python -m json.tool
```

### Step 5. GPU 메모리 확인
```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

**기대값**:
- 각 GPU당 ~13.8GB 사용 (4비트 양자화)

## 🧪 Test Suite 3: 릴리스 게이트 테스트

### 개인 모드 릴리스 게이트
```bash
cd /home/ktl/gpt-oss-hf-server
python scripts/release_gate_personal.py --endpoint http://localhost:8000 --verbose
```

**Pass 기준**:
- QPS: ≥0.3
- P95 TTFT: ≤10초
- P95 E2E: ≤30초
- Error Rate: ≤1%

## 📊 성능 벤치마크

### 간단한 성능 테스트
```bash
# 연속 요청 테스트 (20B)
for i in {1..5}; do
  time curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "gpt-oss-20b",
      "messages": [{"role": "user", "content": "Hello"}],
      "max_tokens": 10
    }' > /dev/null
  echo "Request $i completed"
done
```

## 🔍 문제 해결

### 포트 충돌 시
```bash
lsof -i :8000
kill -9 [PID]
```

### 서버 로그 확인
```bash
# 실시간 로그
tail -f server_v451_*.log

# 에러만 필터링
grep ERROR server_v451_*.log
```

### GPU 메모리 부족 시
```bash
# GPU 프로세스 확인
nvidia-smi
fuser -v /dev/nvidia*

# 캐시 정리
rm -rf ~/.cache/huggingface/hub/.locks/*
```

## ✅ 테스트 체크리스트

### 20B 모델
- [ ] 서버 시작 성공
- [ ] 헬스 체크 정상
- [ ] Non-streaming 요청 성공 (4-8초)
- [ ] 프로파일 전환 정상
- [ ] 스트리밍 버그 확인
- [ ] 메트릭 태깅 확인
- [ ] GPU 메모리 ~12.8GB

### 120B 모델
- [ ] 서버 시작 성공
- [ ] 헬스 체크 정상 (tensor mode)
- [ ] Non-streaming 요청 성공 (6-15초)
- [ ] 품질 향상 확인
- [ ] GPU 메모리 ~13.8GB
- [ ] MoE 구조 확인

### 릴리스 게이트
- [ ] 개인 모드 게이트 통과
- [ ] QPS ≥0.3
- [ ] P95 응답시간 달성
- [ ] 에러율 ≤1%

## 📝 테스트 결과 기록

테스트 완료 후 다음 정보를 기록:

```markdown
테스트 일시: YYYY-MM-DD HH:MM
테스트 모델: 20B / 120B
성공률: X/Y
평균 응답시간: Xs
특이사항: 
```

---

**참고**: 스트리밍 버그는 알려진 이슈이며, `server_v451.py`의 475번 라인 수정으로 해결 가능합니다.