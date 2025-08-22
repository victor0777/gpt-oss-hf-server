#!/bin/bash
# P0 전체 테스트 실행 스크립트 (개선 버전)
# 서버 프로세스 관리 포함

echo "=============================================="
echo "🚀 GPT-OSS HF Server P0 전체 테스트 (개선판)"
echo "=============================================="
echo ""
echo "테스트 일시: $(date)"
echo "서버 버전: v4.5.4-P0"
echo ""

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 결과 저장 배열
declare -a test_results

# 1. 기존 서버 프로세스 확인 및 종료
echo "🔧 기존 서버 프로세스 확인..."
EXISTING_PIDS=$(ps aux | grep "server_v454_p0.py" | grep -v grep | awk '{print $2}')

if [ ! -z "$EXISTING_PIDS" ]; then
    echo -e "${YELLOW}⚠️ 기존 서버 프로세스 발견:${NC}"
    for pid in $EXISTING_PIDS; do
        echo "  PID: $pid"
        kill -9 $pid 2>/dev/null
    done
    echo "  ✅ 기존 프로세스 종료됨"
    sleep 2
else
    echo "  ✅ 실행 중인 서버 없음"
fi

# GPU 메모리 확인
echo ""
echo "📊 GPU 메모리 상태:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | while read line; do
    echo "  $line"
done

# 2. 새 서버 시작
echo ""
echo "🚀 새 서버 시작..."
cd /home/ktl/gpt-oss-hf-server
nohup python src/server_v454_p0.py --model 20b --profile latency_first --port 8000 > server_test.log 2>&1 &
SERVER_PID=$!
echo "  서버 PID: $SERVER_PID"

# 서버 시작 대기
echo "  서버 초기화 대기 중..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo -e "${GREEN}  ✅ 서버가 준비되었습니다${NC}"
        break
    fi
    if [ $i -eq 30 ]; then
        echo -e "${RED}  ❌ 서버 시작 실패 (30초 타임아웃)${NC}"
        echo "  서버 로그 확인:"
        tail -20 server_test.log
        exit 1
    fi
    sleep 1
done

# 3. 서버 상태 확인
echo ""
echo "🔍 서버 상태 확인..."
curl -s http://localhost:8000/health | python -m json.tool | head -15

# 환경 정보 출력
echo ""
echo "📋 환경 정보:"
echo "  Python: $(python --version 2>&1)"
echo "  NumPy: $(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null)"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo ""

# 테스트 실행 함수
run_test() {
    local test_name=$1
    local test_file=$2
    
    echo "=============================================="
    echo "🧪 $test_name 실행 중..."
    echo "=============================================="
    
    python $test_file
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}✅ $test_name 완료${NC}"
        test_results+=("✅ $test_name")
    else
        echo -e "${RED}❌ $test_name 실패${NC}"
        test_results+=("❌ $test_name")
    fi
    
    echo ""
    sleep 2
    return $exit_code
}

# 4. 각 테스트 실행
echo "🚀 테스트 시작"
echo ""

# Test 1: 프롬프트 빌더
run_test "Test 1: 프롬프트 빌더 (PR-PF01)" "test_1_prompt_determinism.py"
test1_result=$?

# Test 2: SSE 스트리밍
run_test "Test 2: SSE 스트리밍 (PR-ST01)" "test_2_sse_streaming.py"
test2_result=$?

# Test 3: 모델 태깅
run_test "Test 3: 모델 태깅 (PR-OBS01)" "test_3_model_tagging.py"
test3_result=$?

# Test 4: 성능
run_test "Test 4: 성능 테스트" "test_4_performance.py"
test4_result=$?

# 5. 최종 결과 출력
echo ""
echo "=============================================="
echo "📊 P0 테스트 최종 결과"
echo "=============================================="
echo ""

for result in "${test_results[@]}"; do
    echo "  $result"
done

echo ""

# 성공/실패 카운트
success_count=0
total_count=${#test_results[@]}

for result in "${test_results[@]}"; do
    if [[ $result == *"✅"* ]]; then
        ((success_count++))
    fi
done

echo "총 테스트: $total_count개"
echo "성공: $success_count개"
echo "실패: $((total_count - success_count))개"
echo "성공률: $((success_count * 100 / total_count))%"
echo ""

# P0 수용 기준 체크
echo "🎯 P0 수용 기준:"
echo ""

if [ $test1_result -eq 0 ]; then
    echo "  ✅ 프롬프트 결정론 달성"
else
    echo "  ❌ 프롬프트 결정론 미달"
fi

if [ $test2_result -eq 0 ]; then
    echo "  ✅ SSE 스트리밍 안정화"
else
    echo "  ❌ SSE 스트리밍 문제"
fi

if [ $test3_result -eq 0 ]; then
    echo "  ✅ 모델 태깅 완료"
else
    echo "  ❌ 모델 태깅 미완"
fi

if [ $test4_result -eq 0 ]; then
    echo "  ✅ 성능 목표 달성"
else
    echo "  ❌ 성능 목표 미달"
fi

echo ""

# 최종 판정
if [ $success_count -eq $total_count ]; then
    echo -e "${GREEN}🎉 모든 P0 테스트 통과!${NC}"
    echo -e "${GREEN}✅ P0 구현 완료${NC}"
    exit_code=0
elif [ $success_count -ge $((total_count * 3 / 4)) ]; then
    echo -e "${YELLOW}⚠️ 대부분의 P0 테스트 통과${NC}"
    echo "일부 개선 필요"
    exit_code=1
else
    echo -e "${RED}❌ P0 테스트 실패${NC}"
    echo "추가 작업 필요"
    exit_code=2
fi

echo ""

# 6. 서버 종료 (옵션)
echo "🛑 테스트 서버 종료..."
kill $SERVER_PID 2>/dev/null
echo "  서버 PID $SERVER_PID 종료됨"

# 로그 저장
echo ""
echo "📄 테스트 결과를 p0_test_results_improved.txt에 저장합니다..."
{
    echo "P0 테스트 결과 (개선판)"
    echo "====================="
    echo "일시: $(date)"
    echo "서버: v4.5.4-P0"
    echo ""
    echo "결과:"
    for result in "${test_results[@]}"; do
        echo "  $result"
    done
    echo ""
    echo "성공률: $((success_count * 100 / total_count))%"
    echo ""
    echo "서버 로그 (마지막 50줄):"
    echo "========================"
    tail -50 server_test.log
} > p0_test_results_improved.txt

echo "✅ 저장 완료"
echo ""
echo "테스트 완료: $(date)"
echo ""

exit $exit_code