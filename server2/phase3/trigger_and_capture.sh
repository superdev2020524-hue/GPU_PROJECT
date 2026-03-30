#!/bin/bash
# Run on the VM. Triggers generate and captures journalctl for runner errors.
# Usage: sudo ./trigger_and_capture.sh

echo "=== Triggering generate ==="
curl -s -X POST http://127.0.0.1:11434/api/generate \
  -d '{"model":"llama3.2:1b","prompt":"Hi","stream":false}' &
CURL_PID=$!

sleep 15
echo ""
echo "=== Recent ollama journal (errors) ==="
journalctl -u ollama -n 150 --no-pager 2>/dev/null | grep -iE "CUDA error|error:|Load failed|exit status|runner|ggml|LastErr" || echo "(no matches)"

echo ""
echo "=== Full last 30 ollama lines ==="
journalctl -u ollama -n 30 --no-pager 2>/dev/null

wait $CURL_PID 2>/dev/null || true
