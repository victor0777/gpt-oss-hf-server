# GPT-OSS HF Server Architecture

## 📋 Overview

GPT-OSS HF Server는 개인 사용자를 위한 고성능 추론 서버입니다. v4.5.1부터는 **개인용 단순화 아키텍처**를 채택했습니다.

## 🏗️ Architecture Evolution

### v4.4: Direct Model Processing
```
[Client] → [Server:8000] → [Model Manager] → [GPU Processing]
```
- **단일 서버**: 모든 처리를 하나의 프로세스에서 수행
- **직접 처리**: HuggingFace Transformers 직접 사용
- **간단한 구조**: 설치와 운영이 쉬움

### v4.5.0: Enterprise Engine Architecture
```
[Client] → [Service:8000] → [Engine Router] → [Custom Engine:8001]
                                            → [vLLM Engine:8002]
                                            → [TRT-LLM:8003]
```
- **분산 아키텍처**: 서비스와 추론 분리
- **다중 엔진**: 상황별 최적 엔진 선택
- **복잡한 구조**: 엔터프라이즈급 확장성

### v4.5.1: Personal Optimized Architecture (현재)
```
[Client] → [Server:8000] → [Profile System] → [Model Processing]
                         → [Metrics]       → [20B/120B Models]
```
- **단순화**: 엔진 시스템 제거, 단일 포트
- **프로파일**: LATENCY_FIRST, QUALITY_FIRST, BALANCED
- **개인 최적화**: 개인 사용 시나리오에 집중

## 🎯 Design Decisions for v4.5.1

### 왜 엔진 시스템을 제거했는가?

#### 개인 사용 시나리오
- **단순성 우선**: 복잡한 설정 불필요
- **단일 사용자**: 로드 밸런싱 불필요  
- **리소스 효율**: 추가 프로세스 오버헤드 제거
- **유지보수**: 단일 서버가 관리 용이

#### 엔터프라이즈 시나리오 (향후)
엔진 시스템이 유용한 경우:
- **다수 사용자**: 동시 요청 처리
- **이기종 하드웨어**: 다양한 GPU 최적화
- **고가용성**: 엔진별 페일오버
- **전문 최적화**: vLLM, TensorRT-LLM 활용

### 개인용 vs 엔터프라이즈

| 요소 | 개인용 (v4.5.1) | 엔터프라이즈 (향후) |
|------|----------------|-------------------|
| **아키텍처** | 모놀리식 | 마이크로서비스 |
| **포트** | 8000 | 8000-8003 |
| **프로세스** | 1개 | 여러 개 |
| **복잡도** | 낮음 | 높음 |
| **성능** | 충분 | 최적화 |
| **확장성** | 수직 | 수평 |

## 🔧 v4.5.1 Core Components

### 1. Server Core
```python
ServerV451
├── ProfileSystem      # 사용 시나리오별 설정
├── MetricsCollector   # 상세한 모니터링
├── RequestManager     # 요청 생명주기 관리
└── ModelProcessor     # 직접 모델 처리 (v4.4 기반)
```

### 2. Profile System
```yaml
LATENCY_FIRST:
  model: 20b
  gpu_mode: pipeline
  max_batch: 32
  response_time: 4-8s

QUALITY_FIRST:
  model: 120b  
  gpu_mode: tensor
  max_batch: 8
  response_time: 6-15s

BALANCED:
  model: 20b/120b
  adaptive: true
```

### 3. Model Processing
- **20B Model**: Pipeline parallelism across 4 GPUs
- **120B Model**: Tensor parallelism with 4-bit quantization
- **Direct Processing**: HuggingFace Transformers 직접 사용

## 🚀 Migration Path

### 현재: 개인용 (v4.5.1)
```bash
# 간단한 시작
python src/server_v451.py --profile latency_first --model 20b
```

### 향후: 엔터프라이즈 확장
```bash
# 1단계: 기본 엔진 추가
python src/server_v6.py --enable-engines

# 2단계: vLLM 통합
docker-compose up -d vllm-engine

# 3단계: 로드 밸런서
nginx -c /etc/nginx/multi-engine.conf
```

## 📊 Performance Characteristics

### 개인용 모드 (v4.5.1)
- **QPS**: 0.3-0.5 (충분함)
- **Latency**: 4-15s (프로파일별)
- **동시 사용자**: 1-3명
- **메모리**: 13-14GB/GPU

### 엔터프라이즈 모드 (향후)
- **QPS**: 2-10 (엔진별)
- **Latency**: 1-5s (최적화)
- **동시 사용자**: 100+
- **메모리**: 동적 할당

## 🎯 Future Roadmap

### Phase 1: Personal Excellence (현재)
- ✅ 프로파일 시스템
- ✅ 120B 모델 지원
- ✅ 개인용 SLO
- 🔄 스트리밍 버그 수정

### Phase 2: Hybrid Mode (v5.0)
- 선택적 엔진 활성화
- 로컬/클라우드 하이브리드
- 자동 모델 선택

### Phase 3: Enterprise Ready (v6.0)
- 완전한 엔진 시스템
- Kubernetes 배포
- 멀티테넌트 지원
- SLA 보장

## 💡 Key Insights

### 개인용 최적화의 가치
1. **복잡성 감소**: 설정과 운영이 간단
2. **즉시 사용**: 추가 설정 없이 바로 시작
3. **충분한 성능**: 개인 사용에 충분한 속도
4. **비용 효율**: 단일 서버로 모든 기능

### 엔진 시스템의 미래 가치
1. **확장 준비**: 필요시 엔진 추가 가능
2. **점진적 마이그레이션**: 단계별 전환
3. **하이브리드 옵션**: 로컬+클라우드 조합
4. **엔터프라이즈 준비**: 대규모 서비스 가능

## 🔨 Implementation Status

### v4.5.1 Current State
- ✅ 프로파일 시스템 구현
- ✅ 메트릭 태깅 완성
- ✅ 20B/120B 모델 지원
- ❌ 실제 모델 처리 (mock 상태)
- ❌ 스트리밍 (버그)

### Next Steps
1. **엔진 제거 완료**: 불필요한 코드 정리
2. **v4.4 통합**: 실제 모델 처리 코드 이식
3. **테스트**: 통합 테스트 수행
4. **문서화**: 사용자 가이드 업데이트

---

**결론**: v4.5.1은 개인 사용에 최적화된 단순하고 효율적인 아키텍처를 채택했습니다. 향후 필요시 엔진 시스템을 다시 도입하여 엔터프라이즈급으로 확장 가능합니다.