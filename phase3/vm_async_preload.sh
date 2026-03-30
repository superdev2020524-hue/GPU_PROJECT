#!/usr/bin/env bash
# Run ON THE VM (e.g. via SSH or connect_vm.py). Decouples slow first load from a single
# blocking curl with a short -m. See PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md
set -euo pipefail

MODEL="${1:-tinyllama:latest}"
OUT="${OLLAMA_PRELOAD_OUT:-/tmp/ollama_preload.out}"
ERR="${OLLAMA_PRELOAD_ERR:-/tmp/ollama_preload.err}"
PIDF="${OLLAMA_PRELOAD_PID:-/tmp/ollama_preload.pid}"

# Empty-body preload per Ollama FAQ + keep model resident after load
PAYLOAD=$(printf '%s' '{"model":"'"$MODEL"'","keep_alive":-1}')

echo "[vm_async_preload] model=$MODEL"
echo "[vm_async_preload] starting background POST /api/generate (preload, no prompt)"
echo "[vm_async_preload] stdout=$OUT stderr=$ERR"

nohup curl -sS -X POST "http://127.0.0.1:11434/api/generate" \
  -d "$PAYLOAD" \
  -o "$OUT" \
  2>>"$ERR" &

echo $! >"$PIDF"
echo "[vm_async_preload] PID=$(cat "$PIDF")"

echo "[vm_async_preload] Poll examples (no long single curl required):"
echo "  journalctl -u ollama -f | grep -E 'model load progress|ERROR|exit status'"
echo "  curl -sS http://127.0.0.1:11434/api/ps"
echo "  tail -f $ERR"
echo "[vm_async_preload] To wait in-shell for that one job: wait \$(cat $PIDF)"
