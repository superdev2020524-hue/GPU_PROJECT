#!/usr/bin/env bash
# Run from repo: phase3/. Polls VM journal + host mediator.log every 10 min, 13 samples over ~2h.
set -eu
cd "$(dirname "$0")"
RECORD="FATBIN_TRACE_RECORD.md"

{
  echo ""
  echo "---"
  echo "## Poll loop started $(date -u +"%Y-%m-%dT%H:%M:%SZ") (local)"
  echo ""
} >>"$RECORD"

poll_sample() {
  local n=$1
  {
    echo ""
    echo "### Sample $n — $(date -u +"%Y-%m-%dT%H:%M:%SZ") UTC"
    echo "#### VM — ollama journal (filtered tail)"
  } >>"$RECORD"
  CONNECT_VM_COMMAND_TIMEOUT_SEC=60 python3 -u connect_vm.py '
sudo journalctl -u ollama -n 80 --no-pager 2>/dev/null | grep -E "model load progress|load_tensors|ERROR|cuda-transport|runner terminated|context canceled|llama runner|sched.go" | tail -35
echo "---VM unfiltered tail (last 8 lines)---"
sudo journalctl -u ollama -n 8 --no-pager 2>/dev/null
' >>"$RECORD" 2>&1 || echo "(connect_vm failed)" >>"$RECORD"

  {
    echo ""
    echo "#### Host — mediator.log (line count + HtoD / module / 401312)"
  } >>"$RECORD"
  python3 -u connect_host.py '
echo "wc:"; wc -l /tmp/mediator.log 2>/dev/null
grep -E "HtoD progress|401312|INVALID_IMAGE|module-load|CUDA_ERROR" /tmp/mediator.log 2>/dev/null | tail -30
' >>"$RECORD" 2>&1 || echo "(connect_host failed)" >>"$RECORD"

  {
    echo ""
    echo "--- end sample $n ---"
  } >>"$RECORD"
}

for i in $(seq 1 13); do
  poll_sample "$i"
  if [ "$i" -lt 13 ]; then
    sleep 600
  fi
done

{
  echo ""
  echo "## Poll loop finished $(date -u +"%Y-%m-%dT%H:%M:%SZ") UTC"
} >>"$RECORD"
