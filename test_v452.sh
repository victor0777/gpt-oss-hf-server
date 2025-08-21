#!/bin/bash
# GPT-OSS HF Server v4.5.2 Test Script

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SERVER_PORT=8000
BASE_URL="http://localhost:$SERVER_PORT"

# Functions
print_header() {
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===============================================${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

check_environment() {
    print_header "Environment Check"
    
    # Check Python
    python --version
    
    # Check NumPy
    python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
    
    # Check PyTorch
    python -c "import torch; print(f'PyTorch: {torch.__version__}')"
    
    # Check GPUs
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,memory.total --format=csv
    else
        print_error "nvidia-smi not found"
    fi
    
    echo ""
}

test_import() {
    print_header "Testing Import"
    
    if python -c "from src.server_v452 import *" 2>/dev/null; then
        print_success "Import successful"
    else
        print_error "Import failed"
        exit 1
    fi
    echo ""
}

test_health() {
    print_header "Testing Health Endpoint"
    
    response=$(curl -s $BASE_URL/health)
    if [ $? -eq 0 ]; then
        print_success "Health endpoint responded"
        echo "$response" | python -m json.tool | head -20
    else
        print_error "Health endpoint failed"
    fi
    echo ""
}

test_stats() {
    print_header "Testing Stats Endpoint"
    
    response=$(curl -s $BASE_URL/stats)
    if [ $? -eq 0 ]; then
        print_success "Stats endpoint responded"
        echo "$response" | python -m json.tool | head -15
    else
        print_error "Stats endpoint failed"
    fi
    echo ""
}

test_chat_simple() {
    print_header "Testing Simple Chat Completion"
    
    response=$(curl -s -X POST $BASE_URL/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "gpt-oss-20b",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 20
        }')
    
    if [ $? -eq 0 ]; then
        print_success "Chat completion responded"
        echo "$response" | python -m json.tool | grep -E '"content"|"model"|"id"' || echo "$response"
    else
        print_error "Chat completion failed"
    fi
    echo ""
}

test_chat_with_timing() {
    print_header "Testing Chat with Timing"
    
    print_info "Sending request..."
    start_time=$(date +%s.%N)
    
    response=$(curl -s -X POST $BASE_URL/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "gpt-oss-20b",
            "messages": [{"role": "user", "content": "What is Python?"}],
            "max_tokens": 100
        }')
    
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc)
    
    if [ $? -eq 0 ]; then
        print_success "Response received in ${duration}s"
        echo "$response" | python -c "import sys, json; data=json.load(sys.stdin); print(f\"Response: {data.get('choices',[{}])[0].get('message',{}).get('content','')[:100]}...\")" 2>/dev/null || echo "Could not parse response"
    else
        print_error "Request failed"
    fi
    echo ""
}

test_profile_switch() {
    print_header "Testing Profile Switch"
    
    print_info "Testing LATENCY_FIRST profile..."
    response=$(curl -s -X POST $BASE_URL/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "gpt-oss-20b",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "profile": "latency_first"
        }')
    
    if echo "$response" | grep -q "content"; then
        print_success "LATENCY_FIRST profile works"
    else
        print_error "LATENCY_FIRST profile failed"
    fi
    
    print_info "Testing QUALITY_FIRST profile..."
    response=$(curl -s -X POST $BASE_URL/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "gpt-oss-20b",
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 10,
            "profile": "quality_first"
        }')
    
    if echo "$response" | grep -q "content"; then
        print_success "QUALITY_FIRST profile works"
    else
        print_error "QUALITY_FIRST profile failed"
    fi
    echo ""
}

test_streaming() {
    print_header "Testing Streaming"
    
    print_info "Sending streaming request..."
    
    # Test streaming with timeout
    timeout 5 curl -N -X POST $BASE_URL/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "gpt-oss-20b",
            "messages": [{"role": "user", "content": "Count to 5"}],
            "max_tokens": 30,
            "stream": true
        }' 2>/dev/null | head -5
    
    if [ $? -eq 124 ] || [ $? -eq 0 ]; then
        print_success "Streaming test completed"
    else
        print_error "Streaming test failed"
    fi
    echo ""
}

run_benchmark() {
    print_header "Running Performance Benchmark"
    
    SUCCESS=0
    FAIL=0
    TOTAL_TIME=0
    
    for i in {1..5}; do
        print_info "Request $i/5..."
        
        start_time=$(date +%s.%N)
        
        if curl -s -X POST $BASE_URL/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "gpt-oss-20b",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 20
            }' > /dev/null 2>&1; then
            SUCCESS=$((SUCCESS+1))
        else
            FAIL=$((FAIL+1))
        fi
        
        end_time=$(date +%s.%N)
        duration=$(echo "$end_time - $start_time" | bc)
        TOTAL_TIME=$(echo "$TOTAL_TIME + $duration" | bc)
    done
    
    AVG_TIME=$(echo "scale=2; $TOTAL_TIME / 5" | bc)
    QPS=$(echo "scale=2; 5 / $TOTAL_TIME" | bc)
    
    echo ""
    print_success "Benchmark Results:"
    echo "  - Success: $SUCCESS/5"
    echo "  - Average time: ${AVG_TIME}s"
    echo "  - QPS: $QPS"
    echo ""
}

# Main menu
show_menu() {
    echo -e "${BLUE}GPT-OSS HF Server v4.5.2 Test Suite${NC}"
    echo ""
    echo "1) Run all tests"
    echo "2) Check environment only"
    echo "3) Test API endpoints"
    echo "4) Test chat completion"
    echo "5) Test profiles"
    echo "6) Test streaming"
    echo "7) Run benchmark"
    echo "8) Exit"
    echo ""
}

# Parse command line arguments
if [ "$1" == "--all" ]; then
    check_environment
    test_import
    test_health
    test_stats
    test_chat_simple
    test_chat_with_timing
    test_profile_switch
    test_streaming
    run_benchmark
    exit 0
fi

# Interactive mode
while true; do
    show_menu
    read -p "Select option: " choice
    
    case $choice in
        1)
            check_environment
            test_import
            test_health
            test_stats
            test_chat_simple
            test_chat_with_timing
            test_profile_switch
            test_streaming
            run_benchmark
            ;;
        2)
            check_environment
            test_import
            ;;
        3)
            test_health
            test_stats
            ;;
        4)
            test_chat_simple
            test_chat_with_timing
            ;;
        5)
            test_profile_switch
            ;;
        6)
            test_streaming
            ;;
        7)
            run_benchmark
            ;;
        8)
            echo "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid option"
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
done