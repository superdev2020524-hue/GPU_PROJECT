#!/bin/bash
# Run on VM to capture runner's actual stderr and exit code via strace.
# Usage: bash track_runner_error.sh
# Or: python3 connect_vm.py 'bash -s' < phase3/track_runner_error.sh

set -e
export PATH="/usr/local/bin:$PATH"

echo "=== 1. Stop ollama service ==="
sudo systemctl stop ollama 2>/dev/null || true
sudo pkill -f 'ollama.bin serve' 2>/dev/null || true
sudo pkill -f 'ollama serve' 2>/dev/null || true
sleep 3
rm -f /tmp/ollama_run.log /tmp/runner_strace.log

echo "=== 2. Start server in background (patched binary, vGPU env) ==="
LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama \
OLLAMA_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama \
OLLAMA_LLM_LIBRARY=cuda_v12 OLLAMA_NUM_GPU=1 OLLAMA_DEBUG=1 \
nohup /usr/local/bin/ollama.bin serve >> /tmp/ollama_run.log 2>&1 &
sleep 6

SERVER_PID=$(pgrep -f "ollama.bin.*serve" | head -1)
if [ -z "$SERVER_PID" ]; then
  echo "ERROR: Server not found"
  tail -30 /tmp/ollama_run.log
  exit 1
fi
echo "=== 3. Server PID: $SERVER_PID ==="

echo "=== 4. Attach strace (follow forks, write + exit_group, separate file per PID) ==="
rm -f /tmp/runner_strace.log.* 2>/dev/null || true
sudo strace -f -ff -p "$SERVER_PID" -e trace=write,exit_group -s 2000 -o /tmp/runner_strace.log 2>/dev/null &
STRACE_PID=$!
sleep 3

echo "=== 5. Trigger generate (background, 8min timeout) ==="
curl -s -m 480 -X POST http://127.0.0.1:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2:1b","prompt":"Hi","stream":false,"options":{"num_predict":2}}' \
  > /tmp/gen_response.json 2>&1 &
CURL_PID=$!

echo "=== 6. Waiting 7 min for load + inference or timeout ==="
sleep 420

echo "=== 7. Stop strace ==="
sudo kill $STRACE_PID 2>/dev/null || true
wait $CURL_PID 2>/dev/null || true

echo "=== 8. Analyze strace: runner stderr and exit (per-PID files with -ff) ==="
# With -ff we get runner_strace.log.$SERVER_PID and runner_strace.log.$RUNNER_PID
for f in /tmp/runner_strace.log.*; do
  [ -f "$f" ] || continue
  pid="${f##*.}"
  echo "--- File $f (PID $pid) ---"
  if [ "$pid" = "$SERVER_PID" ]; then
    echo "(server process)"
    grep -E "exit_group|write(2," "$f" 2>/dev/null | tail -20
  else
    echo "(runner process - stderr and exit)"
    grep "write(2," "$f" 2>/dev/null | tail -100
    echo "--- exit_group for runner ---"
    grep "exit_group" "$f" 2>/dev/null || true
  fi
  echo ""
done
if [ ! -f /tmp/runner_strace.log ]; then
  echo "No strace log found."
fi

echo ""
echo "=== 9. Server log tail (Load failed / error lines) ==="
grep -iE "Load failed|error|runner|exit|progress" /tmp/ollama_run.log 2>/dev/null | tail -20
echo ""
echo "=== 10. Generate response (first 5 lines) ==="
head -5 /tmp/gen_response.json 2>/dev/null || true

echo ""
echo "=== DONE ==="
