#!/bin/bash
# Short (2 min) run to find WHERE the runner blocks: last RPC sent + last syscall.
# Usage: run via run_track_runner_error_short.py (base64 decode and bash)

set -e
export PATH="/usr/local/bin:$PATH"
WAIT_SEC=${WAIT_SEC:-120}

echo "=== 1. Stop ollama, clear call sequence log ==="
sudo systemctl stop ollama 2>/dev/null || true
sudo pkill -f 'ollama.bin serve' 2>/dev/null || true
sudo pkill -f 'ollama serve' 2>/dev/null || true
sleep 3
rm -f /tmp/ollama_run.log
sudo rm -f /tmp/runner_strace.log.* /tmp/vgpu_call_sequence.log 2>/dev/null || true

echo "=== 2. Start server (patched binary, vGPU env) ==="
LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama \
OLLAMA_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama \
OLLAMA_LLM_LIBRARY=cuda_v12 OLLAMA_NUM_GPU=1 OLLAMA_DEBUG=1 \
nohup /usr/local/bin/ollama.bin serve >> /tmp/ollama_run.log 2>&1 &
sleep 6

SERVER_PID=$(pgrep -f "ollama.bin.*serve" | head -1)
[ -n "$SERVER_PID" ] || { echo "ERROR: Server not found"; tail -20 /tmp/ollama_run.log; exit 1; }
echo "=== 3. Server PID: $SERVER_PID ==="

echo "=== 4. Strace (follow forks, read/poll/write/exit_group to see blocking) ==="
sudo strace -f -ff -p "$SERVER_PID" -e trace=read,poll,poll_poll,write,exit_group -s 500 -o /tmp/runner_strace.log 2>/dev/null &
STRACE_PID=$!
sleep 3

echo "=== 5. Trigger generate (background) ==="
curl -s -m 300 -X POST http://127.0.0.1:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model":"llama3.2:1b","prompt":"Hi","stream":false,"options":{"num_predict":2}}' \
  > /tmp/gen_response.json 2>&1 &
CURL_PID=$!

echo "=== 6. Waiting ${WAIT_SEC}s for load (runner will block or progress) ==="
sleep "$WAIT_SEC"

echo "=== 7. Stop strace ==="
sudo kill $STRACE_PID 2>/dev/null || true
wait $CURL_PID 2>/dev/null || true

echo ""
echo "=== 8. LAST RPC SENT (last lines of vgpu_call_sequence.log) — runner blocks WAITING for this response ==="
if [ -f /tmp/vgpu_call_sequence.log ]; then
  wc -l /tmp/vgpu_call_sequence.log
  tail -60 /tmp/vgpu_call_sequence.log
else
  echo "(no vgpu_call_sequence.log — runner may not have reached transport)"
fi

echo ""
echo "=== 9. LAST SYSCALLS per child (where runner is blocked) ==="
for f in /tmp/runner_strace.log.*; do
  [ -f "$f" ] || continue
  pid="${f##*.}"
  if [ "$pid" = "$SERVER_PID" ]; then continue; fi
  echo "--- PID $pid (last 25 syscalls) ---"
  tail -25 "$f" 2>/dev/null || true
  echo "--- exit_group for PID $pid ---"
  grep "exit_group" "$f" 2>/dev/null || true
  echo ""
done

echo "=== 10. Server log (Load failed / progress) ==="
grep -iE "Load failed|progress|error.*runner" /tmp/ollama_run.log 2>/dev/null | tail -15
echo ""
echo "=== 11. Generate response ==="
head -3 /tmp/gen_response.json 2>/dev/null || true
echo ""
echo "=== DONE (short run) ==="
