# 작업 지시서: OOM 방지 & 4-GPU 활용 고도화 (v4.6.0)

## 📋 Executive Summary

**목표**: 단일 GPU(20B) OOM 완전 차단 + 대규모 요청 자동 4-GPU 라우팅  
**범위**: v4.5.4 → v4.6.0  
**우선순위**: P0(필수) → P1(권장) → P2(선택) → P3(관측)

## 🎯 핵심 목표

1. **단일 GPU(20B)에서 OOM '발생 전' 차단**
2. **큰/긴 요청만 자동으로 4-GPU 경로로 라우팅**
3. **성능 저하 최소화** (개인용 SLO: TTFT p95 ≤ 7s, E2E p95 ≤ 20s, 에러 < 0.5%)

---

## P0 — 즉시 구현 (필수): 메모리 가드레일

### PR-MEM01: 사전 메모리 추정 + 입장제어 (Admission Control)

**목적**: 토크나이즈 직후, KV 캐시 예상 메모리와 배치/프리필 활성 메모리를 합산하여 입장 여부 결정

**구현 방안**:
```python
# 메모리 예측 공식 (모델 config 기반)
kv_bytes ≈ tokens * num_layers * (num_kv_heads * head_dim) * 2(K&V) * dtype_bytes

# 입장 제어 로직
if predicted_bytes + safety_reserve > gpu_free * THRESHOLD:
    1. 컨텍스트 축약 (요약/헤더 제거)
    2. max_new_tokens 축소
    3. 대기열 지연
    4. 라지-패스(4GPU) 라우팅
    5. 최종 429 응답
```

**검증 기준**: 
- 인위적 OOM 재현 시 5xx=0, 429로 치환
- 메모리 누수 없음

### PR-MEM02: 세션별 KV 상한 + LRU/Idle Evict

**목적**: 세션별 메모리 사용량 제한 및 효율적 메모리 관리

**구현 방안**:
- `session_kv_limit_mb`: 512MB (세션당)
- `max_kv_gb`: 20GB (글로벌)
- Idle 세션 (최근 n초 미사용) 우선 페이징/해제
- LRU 정책 적용

**검증 기준**: 
- 동시 6 세션 장문 반복 시 OOM 미발생
- 캐시 히트 테스트 통과

### PR-MEM03: 메모리 압력 기반 동적 디그레이드

**목적**: GPU 사용률/여유 VRAM에 따라 실시간 파라미터 조정

**동적 조정 항목**:
- `max_new_tokens` ↓
- `temperature` = 0
- `top_p`: 0.9 → 0.7
- `batch_max_size` ↓

**검증 기준**: 
- 메모리 압력 이벤트 중 5xx=0
- p95 증가 ≤ +10%

---

## P1 — 빠른 개선 (권장): 4-GPU 자동 라우팅 & 실행

### PR-MG01: 라지-패스 자동 라우팅 (20B → 4GPU)

**트리거 조건**:
- `predicted_kv_bytes > LARGE_REQ_KB` 
- 또는 `input_tokens > 8k`

**구현 방안**:
- 동일 API 유지
- 엔진/런처를 `hybrid:tp2,pp2`로 자동 선택
- `MICRO_BATCHES=4~8`

**검증 기준**: 
- 라지-패스 요청에서 5xx=0, OOM=0
- p95 E2E ≤ 20s 유지

### PR-MG02: 하이브리드 병렬 런처 (TP2+PP2) 템플릿

**목적**: PCIe 환경 기준 통신 오버헤드 최소화

**구성**:
- Tensor Parallelism = 2
- Pipeline Parallelism = 2
- NCCL 튜닝:
```bash
NCCL_ASYNC_ERROR_HANDLING=1
NCCL_MIN_NCHANNELS=4
NCCL_BUFFSIZE=8388608
```

**검증 기준**: 
- 4장 GPU utilization 균형 (각 ≥60%)
- VRAM ≤85%
- 재시작/리로드 후 일관된 device_map

---

## P2 — 선택 (효율성 향상)

### PR-OPT01: Decode 배치 상한 & 길이 버킷 재조정

**목적**: 긴 컨텍스트+긴 출력 동시 발생 시 최적화

**구현**: 
- `decode_max_batch_tokens` 하향 (예: 8-12k)

**검증 기준**: 
- 피크 시 p95 TTFT 영향 ≤ +5%
- OOM=0

### PR-OPT02: CPU 오프로드 (옵션)

**목적**: 오래된 KV 페이지를 pinned CPU 메모리로 스왑

**구현**:
- 백그라운드 prefetch
- Pinned memory 활용

**검증 기준**: 
- 장기 세션 30분 소크 테스트에서 VRAM 안정
- Swap 왕복 지연 허용 범위 내

### PR-OPT03: INT8/FP8 양자화 경로

**목적**: 20B 단일 GPU 여유 확대

**구현**: 
- Weight-only 양자화 우선
- 품질/호환성 검증 필수

**검증 기준**: 
- 품질 검증 통과
- OOM 마진 ≥ 15% 증가

---

## P3 — 관측 & 회복

### PR-OBS01: 메모리 예측/결정 로깅 & OTel 태깅

**로깅 항목**:
- `pred_kv_mb`: 예측 KV 캐시 크기
- `admission_action`: {accept|shrink|route4|reject}
- `route`: single|tp2pp2

**검증 기준**: 
- OOM 분석 시 단일 요청 경로 재현 가능

### PR-RCV01: OOM 세이프가드

**구현**:
1. CUDA OOM 캐치
2. 즉시 KV/캐시 해제
3. 해당 워커만 재시작
4. 헬스 스코어 하향

**검증 기준**: 
- OOM 주입 테스트에서 프로세스 전체 다운 없음
- 30s 내 정상 복귀

---

## 📊 기본 설정 (권장)

```yaml
# 메모리 가드
GPU_MEMORY_THRESHOLD: 0.85
MEM_SAFETY_RESERVE_MB: 2048
SESSION_KV_LIMIT_MB: 512
MAX_KV_GB: 20

# 라지-패스 트리거
LARGE_REQ_TOKENS: 8000
LARGE_REQ_KV_MB: 6000
LARGE_PATH: "hybrid:tp2,pp2"
MICRO_BATCHES: 6

# 배칭 (지연 우선)
PREFILL_WINDOW_MS: 6
DECODE_WINDOW_MS: 3
DECODE_MAX_BATCH_TOKENS: 8192
```

---

## ✅ 검증 시나리오

### 1. OOM 재현 스크립트
- 긴 컨텍스트(12k) × 동시 4, 출력 1k
- 기대: 5xx=0, 429만 발생

### 2. 라지-패스 라우팅
- 임계 초과 입력이 자동 4GPU로 가는지 확인
- /stats 및 OTel 태그로 검증

### 3. 소크 테스트 (30분)
- 동시 4 세션, 길이 혼합
- 기대: VRAM ≤85%, p95 E2E ≤20s

### 4. OOM 주입 테스트
- 인위적 OOM 트리거
- 기대: 워커만 재시작, 30s 내 정상화

---

## 📈 구현 우선순위

### Phase 1 (즉시)
- [ ] PR-MEM01: Admission Control
- [ ] PR-MEM02: KV 제한
- [ ] PR-MEM03: 동적 디그레이드

### Phase 2 (1주 내)
- [ ] PR-MG01: 자동 라우팅
- [ ] PR-MG02: 하이브리드 병렬

### Phase 3 (선택)
- [ ] PR-OPT01: Decode 최적화
- [ ] PR-OPT02: CPU 오프로드
- [ ] PR-OPT03: 양자화

### Phase 4 (지속)
- [ ] PR-OBS01: 로깅/모니터링
- [ ] PR-RCV01: OOM 세이프가드

---

## 💡 핵심 요약

- **P0 (Admission + KV 제한 + 디그레이드)만으로도 단일 GPU OOM을 즉시 종결**
- **P1 (자동 4-GPU 라우팅)을 더하면 긴 입력·장문에서도 안전하게 여유 확보**
- 나머지는 선택 과제이며, 개인 운영 기준에선 P0+P1이면 충분

---

**작성일**: 2025-08-22  
**버전**: v4.6.0 계획  
**상태**: 🚧 구현 대기