#!/bin/bash
# Quick capture: trigger one generate on the VM, then print error logs and last call sequence.
# Run from phase3: bash quick_capture_errors.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

python3 connect_vm.py "rm -f /tmp/ollama_errors_full.log /tmp/ollama_errors_filtered.log; curl -s -m 70 -X POST http://127.0.0.1:11434/api/generate -d '{\"model\":\"tinyllama\",\"prompt\":\"Hi\",\"stream\":false,\"options\":{\"num_predict\":2}}' -o /tmp/gen.json -w '\nHTTP=%{http_code}\n'; echo '=== /tmp/ollama_errors_full.log ==='; cat /tmp/ollama_errors_full.log 2>/dev/null || echo '(empty)'; echo '=== /tmp/ollama_errors_filtered.log ==='; cat /tmp/ollama_errors_filtered.log 2>/dev/null || echo '(empty)'; echo '=== /tmp/vgpu_next_call.log (last 40) ==='; tail -40 /tmp/vgpu_next_call.log 2>/dev/null || echo '(empty)'; echo '=== /tmp/gen.json ==='; cat /tmp/gen.json"
