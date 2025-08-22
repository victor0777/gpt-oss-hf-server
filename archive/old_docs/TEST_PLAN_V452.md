# GPT-OSS HF Server v4.5.2 테스트 계획

## 📋 테스트 목표

v4.5.2 버전의 모든 기능을 검증하고 안정성을 확인합니다.

### 주요 개선사항 확인
- ✅ NumPy 2.x 호환성
- ✅ 엔진 시스템 완전 제거 
- ✅ 실제 모델 로딩 및 추론
- ✅ 스트리밍 지원
- ✅ 프로파일 시스템

## 🚀 사전 준비

### 1. 환경 설정
```bash
cd /home/ktl/gpt-oss-hf-server
source .venv/bin/activate

# NumPy 버전 확인 (2.x 또는 1.x 모두 가능)
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"

# GPU 확인
nvidia-smi
```

### 2. 서버 실행 가능 여부 확인
```bash
# 임포트 테스트
python -c "from src.server_v452 import *; print('✅ Import successful')"
```

## 📝 테스트 시나리오

### Phase 1: 기본 기능 테스트 (Mock Mode)

#### Test 1.1: 서버 시작 (Mock)
```bash
# transformers 없이도 시작되는지 확인
python src/server_v452.py --model 20b --profile latency_first --port 8000
```

**검증 항목**:
- [ ] 서버 시작 성공
- [ ] NumPy 2.x 경고 없음
- [ ] Mock 모드 메시지 확인

#### Test 1.2: API 엔드포인트 (Mock)
```bash
# 헬스 체크
curl -s http://localhost:8000/health | python -m json.tool

# 통계 확인
curl -s http://localhost:8000/stats | python -m json.tool

# Mock 응답 테스트
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }' | python -m json.tool
```

**검증 항목**:
- [ ] 모든 엔드포인트 정상 응답
- [ ] Mock 응답 생성 확인

### Phase 2: 20B 모델 실제 테스트

#### Test 2.1: 20B 모델 로딩
```bash
# LATENCY_FIRST 프로파일로 시작
python src/server_v452.py --model 20b --profile latency_first --port 8000
```

**검증 항목**:
- [ ] 모델 로딩 성공 메시지
- [ ] GPU 메모리 사용 (~12.8GB)
- [ ] 에러 없이 시작

#### Test 2.2: 20B 기본 추론
```bash
# 간단한 질문
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "What is Python?"}],
    "max_tokens": 100,
    "temperature": 0.7
  }' | python -m json.tool

# 응답 시간 측정
time curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Write a hello world function"}],
    "max_tokens": 150
  }' > /dev/null
```

**검증 항목**:
- [ ] 실제 모델 응답 생성
- [ ] 응답 시간 4-8초
- [ ] 응답 품질 확인

#### Test 2.3: 프로파일 전환
```bash
# QUALITY_FIRST 프로파일로 전환
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Explain recursion in detail"}],
    "max_tokens": 300,
    "profile": "quality_first"
  }' | python -m json.tool
```

**검증 항목**:
- [ ] 프로파일 전환 성공
- [ ] 더 긴 응답 생성 (max_tokens 증가)

#### Test 2.4: 스트리밍 테스트
```bash
# 스트리밍 요청
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Count from 1 to 10"}],
    "max_tokens": 50,
    "stream": true
  }'
```

**검증 항목**:
- [ ] SSE 형식으로 스트리밍
- [ ] 점진적 출력
- [ ] [DONE] 메시지로 종료

### Phase 3: 120B 모델 테스트

#### Test 3.1: 서버 재시작 (120B)
```bash
# 기존 서버 종료 (Ctrl+C)
# 120B 모델로 시작
python src/server_v452.py --model 120b --profile quality_first --port 8000
```

**검증 항목**:
- [ ] 120B 모델 로딩 시도
- [ ] GPU 메모리 분산 (~13.8GB per GPU)
- [ ] Tensor parallelism 활성화

#### Test 3.2: 120B 복잡한 추론
```bash
# 복잡한 질문
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-120b",
    "messages": [{"role": "user", "content": "Explain the differences between machine learning and deep learning with examples"}],
    "max_tokens": 500,
    "temperature": 0.8
  }' | python -m json.tool
```

**검증 항목**:
- [ ] 고품질 응답 생성
- [ ] 응답 시간 6-15초
- [ ] 20B보다 상세한 답변

#### Test 3.3: GPU 활용도 확인
```bash
# 별도 터미널에서
watch -n 1 nvidia-smi

# 요청 중 GPU 사용률 모니터링
```

**검증 항목**:
- [ ] 4개 GPU 모두 사용
- [ ] 메모리 균등 분산

### Phase 4: 안정성 테스트

#### Test 4.1: 연속 요청
```bash
# 5회 연속 요청
for i in {1..5}; do
  echo "Request $i"
  time curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "gpt-oss-20b",
      "messages": [{"role": "user", "content": "Hello"}],
      "max_tokens": 20
    }' > /dev/null
  echo "---"
done
```

**검증 항목**:
- [ ] 모든 요청 성공
- [ ] 일관된 응답 시간
- [ ] 메모리 누수 없음

#### Test 4.2: 통계 확인
```bash
# 최종 통계
curl -s http://localhost:8000/stats | python -m json.tool
```

**검증 항목**:
- [ ] QPS >= 0.3
- [ ] Error rate < 1%
- [ ] P95 latency < 10s

### Phase 5: 에러 처리

#### Test 5.1: 잘못된 요청
```bash
# 빈 메시지
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [],
    "max_tokens": 100
  }'

# 너무 큰 max_tokens
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 10000
  }'
```

**검증 항목**:
- [ ] 적절한 에러 메시지
- [ ] 서버 크래시 없음

## 📊 성능 벤치마크

### 측정 스크립트
```bash
# benchmark.sh 생성
cat > benchmark.sh << 'EOF'
#!/bin/bash
echo "Starting benchmark..."
START=$(date +%s)
SUCCESS=0
FAIL=0

for i in {1..10}; do
  if curl -s -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "gpt-oss-20b",
      "messages": [{"role": "user", "content": "Hello"}],
      "max_tokens": 50
    }' > /dev/null 2>&1; then
    SUCCESS=$((SUCCESS+1))
  else
    FAIL=$((FAIL+1))
  fi
done

END=$(date +%s)
DURATION=$((END-START))

echo "Results:"
echo "- Duration: ${DURATION}s"
echo "- Success: $SUCCESS/10"
echo "- QPS: $(echo "scale=2; 10/$DURATION" | bc)"
EOF

chmod +x benchmark.sh
./benchmark.sh
```

## ✅ 테스트 체크리스트

### 기능 테스트
- [ ] NumPy 2.x 호환성 확인
- [ ] Mock 모드 동작
- [ ] 20B 모델 실제 로딩
- [ ] 120B 모델 실제 로딩
- [ ] 프로파일 시스템 동작
- [ ] 스트리밍 응답
- [ ] 에러 처리

### 성능 테스트
- [ ] 20B: 4-8초 응답 시간
- [ ] 120B: 6-15초 응답 시간
- [ ] QPS >= 0.3
- [ ] 에러율 < 1%
- [ ] GPU 메모리 효율

### 안정성 테스트
- [ ] 연속 요청 처리
- [ ] 메모리 누수 없음
- [ ] 장시간 실행 안정성

## 🐛 문제 해결

### NumPy 에러 발생 시
```bash
# NumPy 버전 확인
python -c "import numpy; print(numpy.__version__)"

# v4.5.2는 NumPy 2.x 지원하므로 문제 없어야 함
# 그래도 문제시: pip install --upgrade scipy scikit-learn
```

### 모델 로딩 실패 시
```bash
# 캐시 확인
ls -la ~/.cache/huggingface/hub/models--openai--gpt-oss-*/

# 디스크 공간 확인
df -h

# GPU 메모리 확인
nvidia-smi
```

### 스트리밍 에러 시
```bash
# 스트리밍 비활성화 테스트
export ENABLE_STREAMING=false
python src/server_v452.py --model 20b --port 8000
```

## 📝 테스트 결과 기록

테스트 완료 후 다음 정보를 기록:

```markdown
테스트 일시: 2025-08-21
버전: v4.5.2
환경:
- NumPy: [버전]
- PyTorch: [버전]
- GPU: [모델 및 개수]

20B 모델:
- 로딩 시간: [초]
- 평균 응답 시간: [초]
- 성공률: [%]
- 메모리 사용: [GB]

120B 모델:
- 로딩 시간: [초]
- 평균 응답 시간: [초]
- 성공률: [%]
- 메모리 사용: [GB]

특이사항:
[기록]
```

---

**테스트 준비 완료!** 이제 단계별로 테스트를 진행하실 수 있습니다.