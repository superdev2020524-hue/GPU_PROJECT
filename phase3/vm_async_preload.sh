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
echo "[vm_async_preload] If GPU generate returns garbage after preload (VM-6 BAR1): run one num_gpu:0 /api/generate then num_gpu:-1 (PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md §7)."
echo "[vm_async_preload] March 29 Test-4 bounded replay (VM-6 / ERROR_TRACKING_STATUS.md 2026-05-15): after /api/ps lists the model,"
echo "  run ONE CPU generate with the same prompt as Test-4 (options.num_gpu:0, same num_predict), then the archival Test-4 POST (omit num_gpu) with curl -m 185 — expect HTTP 200 + readable response."
echo "[vm_async_preload] Resident pair only (no restart): bash run_resident_mar29_test4.sh — §7 then Test-4 (never run strict Test-4 alone; see PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md §7)."
echo "[vm_async_preload] Full cold chain: systemctl restart ollama → this script → poll /api/ps → §7 CPU prime → Test-4 (see PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md §8)."
echo "[vm_async_preload] One-shot automation on the VM: bash run_mar29_section8_chain.sh (LF line endings; repo phase3/ — base64 or scp to guest if needed)."
