#!/bin/bash
# Run P0 Tests for GPT-OSS HF Server v4.5.4

echo "🚀 Starting P0 Test Suite for GPT-OSS HF Server"
echo "=============================================="

# Kill any existing server
echo "🔧 Cleaning up existing processes..."
pkill -f "server_v454_p0.py" 2>/dev/null
sleep 2

# Start server in background
echo "🚀 Starting server v4.5.4-P0..."
cd /home/ktl/gpt-oss-hf-server
source .venv/bin/activate 2>/dev/null || true

python src/server_v454_p0.py --model 20b --profile latency_first --port 8000 > server_p0.log 2>&1 &
SERVER_PID=$!

echo "Server PID: $SERVER_PID"
echo "Waiting for server to initialize..."
sleep 10

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Server failed to start. Check server_p0.log for details"
    tail -20 server_p0.log
    exit 1
fi

# Run tests
echo ""
echo "🧪 Running P0 tests..."
python test_p0.py --server http://localhost:8000

TEST_EXIT_CODE=$?

# Capture server stats
echo ""
echo "📊 Final server stats:"
curl -s http://localhost:8000/stats | python -m json.tool | head -30

# Stop server
echo ""
echo "🛑 Stopping server..."
kill $SERVER_PID 2>/dev/null
sleep 2

# Show server log tail
echo ""
echo "📜 Server log (last 20 lines):"
tail -20 server_p0.log

echo ""
echo "✅ P0 test suite completed"

exit $TEST_EXIT_CODE