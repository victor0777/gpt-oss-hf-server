#!/bin/bash

# P0 Test Runner Script
# Tests all P0 features for v4.6.0

echo "=================================="
echo "GPT-OSS HF Server v4.6.0 P0 Tests"
echo "=================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if server is running
echo "🔍 Checking server status..."
curl -s http://localhost:8000/health > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Server is not running${NC}"
    echo ""
    echo "Please start the server first:"
    echo "  python src/server.py --model 20b --profile latency_first"
    echo ""
    exit 1
fi

echo -e "${GREEN}✅ Server is running${NC}"
echo ""

# Function to run a test
run_test() {
    local test_name=$1
    local test_file=$2
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Running: $test_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    python $test_file
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ $test_name PASSED${NC}"
    else
        echo -e "${RED}❌ $test_name FAILED${NC}"
    fi
    echo ""
}

# Run individual P0 tests
echo "🚀 Starting P0 test suite..."
echo ""

# Test 1: Prompt Determinism
run_test "Test 1: Prompt Determinism" "tests/p0/test_1_prompt_determinism.py"

# Test 2: SSE Streaming
run_test "Test 2: SSE Streaming" "tests/p0/test_2_sse_streaming.py"

# Test 3: Model Tagging
run_test "Test 3: Model Tagging" "tests/p0/test_3_model_tagging.py"

# Test 4: Performance
run_test "Test 4: Performance" "tests/p0/test_4_performance.py"

# Test 5: Memory Management (NEW)
run_test "Test 5: Memory Management" "tests/p0/test_memory_management.py"

# Test 6: P0 Integration (NEW)
run_test "Test 6: P0 Integration" "tests/p0/test_integration_p0.py"

echo ""
echo "=================================="
echo "P0 Test Suite Complete"
echo "=================================="
echo ""

# Summary
echo "📊 Test Summary:"
echo "  1. Prompt Determinism: Tests prompt caching and consistency"
echo "  2. SSE Streaming: Tests streaming stability and cancellation"
echo "  3. Model Tagging: Tests observability and metrics"
echo "  4. Performance: Tests TTFT and E2E latency targets"
echo "  5. Memory Management: Tests PR-MEM01/02/03 features"
echo "  6. P0 Integration: Tests all features working together"
echo ""

echo -e "${YELLOW}💡 For detailed memory stats during tests:${NC}"
echo "  curl http://localhost:8000/memory_stats | jq"
echo ""

echo -e "${YELLOW}💡 To monitor GPU memory in real-time:${NC}"
echo "  watch -n 1 nvidia-smi"
echo ""