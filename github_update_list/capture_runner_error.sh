#!/bin/bash
# Run this ON THE VM (test-3@10.25.33.11) to capture the llama runner's CUDA error.
# Usage: ./capture_runner_error.sh
# Or: python3 connect_vm.py 'bash -s' < capture_runner_error.sh

set -e
echo "=== Capturing runner error (run on VM) ==="

# 1. Trigger a generate in background
(
  sleep 1
  curl -s -X POST http://127.0.0.1:11434/api/generate \
    -H 'Content-Type: application/json' \
    -d '{"model":"llama3.2:1b","prompt":"Hi","stream":false}' \
    > /tmp/generate_output.json 2>&1
) &
CURL_PID=$!

# 2. Tail journalctl for ollama while curl runs
sleep 8
sudo journalctl -u ollama -n 200 --no-pager 2>/dev/null \
  | grep -iE "CUDA error|error:|exit status|Load failed|runner|ggml" \
  | tail -30

# 3. Show last error if any
echo ""
echo "=== Generate response ==="
cat /tmp/generate_output.json 2>/dev/null | head -3
echo ""
echo "=== Last 15 ollama log lines ==="
sudo journalctl -u ollama -n 15 --no-pager 2>/dev/null

wait $CURL_PID 2>/dev/null || true
echo "=== Done ==="
