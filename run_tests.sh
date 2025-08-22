#!/bin/bash
# GPT-OSS HF Server v4.5.4 - Comprehensive Test Suite
# Usage: ./run_tests.sh [20b|120b] [all|p0|integration|performance]

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL=${1:-"20b"}
TEST_SUITE=${2:-"all"}
PORT=8000
SERVER_WAIT_TIME=30

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}🚀 GPT-OSS HF Server v4.5.4 Test Suite${NC}"
echo -e "${BLUE}============================================${NC}"
echo ""
echo "📌 Model: $MODEL"
echo "📌 Test Suite: $TEST_SUITE"
echo "📌 Date: $(date)"
echo ""

# Function to check if server is running
check_server() {
    curl -s http://localhost:$PORT/health > /dev/null 2>&1
    return $?
}

# Function to start server
start_server() {
    echo -e "${YELLOW}🔧 Starting server...${NC}"
    
    # Kill any existing server
    pkill -f "python src/server.py" 2>/dev/null || true
    sleep 2
    
    # Start new server
    nohup python src/server.py \
        --model $MODEL \
        --profile latency_first \
        --port $PORT \
        --auto-port \
        > logs/test_server.log 2>&1 &
    
    SERVER_PID=$!
    echo "  Server PID: $SERVER_PID"
    
    # Wait for server to be ready
    echo -n "  Waiting for server startup"
    for i in $(seq 1 $SERVER_WAIT_TIME); do
        if check_server; then
            echo -e "\n  ${GREEN}✅ Server ready!${NC}"
            return 0
        fi
        echo -n "."
        sleep 1
    done
    
    echo -e "\n  ${RED}❌ Server startup timeout${NC}"
    cat logs/test_server.log | tail -20
    return 1
}

# Function to run P0 tests
run_p0_tests() {
    echo -e "\n${BLUE}📋 Running P0 Tests${NC}"
    echo "================================"
    
    local results=()
    
    # Test 1: Prompt Determinism
    echo -e "\n${YELLOW}Test 1: Prompt Determinism${NC}"
    if python tests/p0/test_1_prompt_determinism.py; then
        results+=("✅ Prompt Determinism")
    else
        results+=("❌ Prompt Determinism")
    fi
    
    # Test 2: SSE Streaming
    echo -e "\n${YELLOW}Test 2: SSE Streaming${NC}"
    if python tests/p0/test_2_sse_streaming.py; then
        results+=("✅ SSE Streaming")
    else
        results+=("❌ SSE Streaming")
    fi
    
    # Test 3: Model Tagging
    echo -e "\n${YELLOW}Test 3: Model Tagging${NC}"
    if python tests/p0/test_3_model_tagging.py; then
        results+=("✅ Model Tagging")
    else
        results+=("❌ Model Tagging")
    fi
    
    # Test 4: Performance
    echo -e "\n${YELLOW}Test 4: Performance${NC}"
    if python tests/p0/test_4_performance.py; then
        results+=("✅ Performance")
    else
        results+=("❌ Performance")
    fi
    
    # Print summary
    echo -e "\n${BLUE}📊 P0 Test Results:${NC}"
    for result in "${results[@]}"; do
        echo "  $result"
    done
}

# Function to run integration tests
run_integration_tests() {
    echo -e "\n${BLUE}📋 Running Integration Tests${NC}"
    echo "================================"
    
    python tests/p0/test_integration.py
}

# Function to run all tests
run_all_tests() {
    run_p0_tests
    run_integration_tests
}

# Main execution
main() {
    # Ensure logs directory exists
    mkdir -p logs
    
    # Start server if not running
    if ! check_server; then
        if ! start_server; then
            echo -e "${RED}Failed to start server${NC}"
            exit 1
        fi
    else
        echo -e "${GREEN}✅ Server already running${NC}"
    fi
    
    # Run requested test suite
    case $TEST_SUITE in
        "p0")
            run_p0_tests
            ;;
        "integration")
            run_integration_tests
            ;;
        "performance")
            echo -e "\n${YELLOW}Running performance tests only${NC}"
            python tests/p0/test_4_performance.py
            ;;
        "all")
            run_all_tests
            ;;
        *)
            echo -e "${RED}Unknown test suite: $TEST_SUITE${NC}"
            echo "Usage: $0 [20b|120b] [all|p0|integration|performance]"
            exit 1
            ;;
    esac
    
    # Save test results
    echo -e "\n${BLUE}💾 Saving test results...${NC}"
    {
        echo "Test Report - $(date)"
        echo "Model: $MODEL"
        echo "Test Suite: $TEST_SUITE"
        echo ""
        echo "Server Log:"
        tail -50 logs/test_server.log
    } > "reports/test_report_$(date +%Y%m%d_%H%M%S).txt"
    
    echo -e "${GREEN}✅ Test execution complete!${NC}"
}

# Run main function
main