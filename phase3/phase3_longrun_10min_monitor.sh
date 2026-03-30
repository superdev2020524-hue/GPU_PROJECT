#!/usr/bin/env bash
# Append incremental Markdown snapshots every INTERVAL (default 600s) for a 4h long-run.
# Aligns with INCREMENTAL_RUN_MONITORING.md — VM journal + host mediator + /api/ps + curl alive.
#
# Usage (workstation, phase3/):
#   export PHASE3_LONGRUN_TS=20260326_175917   # must match /tmp/phase3_longrun_<TS>/ on VM
#   export SSHPASS="$(python3 -c 'from vm_config import MEDIATOR_PASSWORD; print(MEDIATOR_PASSWORD)')"
#   nohup ./phase3_longrun_10min_monitor.sh >> /tmp/phase3_monitor_nohup.log 2>&1 &
#
# Output: LONGRUN_SESSION_<TS>.md in this directory (gitignored pattern optional).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

TS="${PHASE3_LONGRUN_TS:?Set PHASE3_LONGRUN_TS to the session id (e.g. 20260326_175917)}"
INTERVAL="${PHASE3_MONITOR_INTERVAL_SEC:-600}"
DURATION="${PHASE3_LONGRUN_DURATION_SEC:-14400}"

MH="$(python3 -c "from vm_config import MEDIATOR_HOST; print(MEDIATOR_HOST)")"
VH="$(python3 -c "from vm_config import VM_HOST; print(VM_HOST)")"
VU="test-4"

OUT="${PHASE3_LONGRUN_MD_OUT:-$SCRIPT_DIR/LONGRUN_SESSION_${TS}.md}"
SSHPASS="${SSHPASS:-$(python3 -c "from vm_config import MEDIATOR_PASSWORD; print(MEDIATOR_PASSWORD)")}"
export SSHPASS

ssh_vm() { sshpass -e ssh -n -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=no -o ConnectTimeout=20 "${VU}@${VH}" "$@"; }
ssh_host() { sshpass -e ssh -n -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=no -o ConnectTimeout=20 "root@${MH}" "$@"; }

tick=0
N_TICKS=$(( DURATION / INTERVAL ))
[[ "$N_TICKS" -ge 1 ]] || N_TICKS=1

if [[ ! -f "$OUT" ]]; then
  {
    echo "# Phase 3 long-run — automated 10-minute monitor"
    echo ""
    echo "- **Session \`TS\`:** \`${TS}\`"
    echo "- **VM artifact dir:** \`/tmp/phase3_longrun_${TS}/\`"
    echo "- **Monitor started (UTC):** $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- **Interval:** ${INTERVAL}s · **Duration cap:** ${DURATION}s (4h default)"
    echo "- **See:** \`INCREMENTAL_RUN_MONITORING.md\`, \`LONG_RUN_4H_LOG_PATHS.md\`"
    echo ""
    echo "---"
    echo ""
  } >> "$OUT"
fi

append_tick() {
  tick=$((tick + 1))
  local now_utc
  now_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  local vm_snip host_snip ps_snip curl_snip alerts
  vm_snip="$(ssh_vm 'journalctl -u ollama -n 80 --no-pager 2>/dev/null | grep -E "model load progress|ERROR|exit status|context canceled|499|401312|INVALID|runner|sched.go|cuda|GEMM|llama" | tail -50' 2>&1 || echo "(ssh vm failed)")"
  host_snip="$(ssh_host "grep -E 'HtoD progress|module-load|401312|INVALID_IMAGE|FAILED|GEMM|cuda-executor|INVALID|cuCtxSynchronize rc=|result\\.status=700' /tmp/mediator.log 2>/dev/null | tail -35" 2>&1 || echo "(ssh host failed)")"
  # E1–E5 focused traces (maximize capture per SYSTEMATIC_ERROR_TRACKING_PLAN.md §3–5)
  vm_err="$(ssh_vm 'journalctl -u ollama -b --no-pager 2>/dev/null | grep -Ei "mmq|mmq_x|3884|GGML_ABORT|ggml_abort|SIGABRT|fatal error|llama_init|INVALID_IMAGE|401312|STATUS_ERROR|CUDA_ERROR|illegal address|rc=700|result\\.status=700|cublasGemm|GEMM_BATCHED|exit status|context canceled|sched\\.go.*error" | tail -80' 2>&1 || echo "(ssh vm err failed)")"
  host_err="$(ssh_host "grep -Ei 'module-load|401312|INVALID_IMAGE|INVALID|cuModuleLoad|FatBinary|fail401312|HtoD progress|FAILED|cuda-executor|GEMM|Batched|cuCtxSynchronize rc=|result\\.status=700|700|ILLEGAL|ERROR' /tmp/mediator.log 2>/dev/null | tail -80" 2>&1 || echo "(ssh host err failed)")"
  ps_snip="$(ssh_vm "curl -sS -m 8 http://127.0.0.1:11434/api/ps 2>&1 | head -c 2000" 2>&1 || echo "(api/ps failed)")"
  curl_snip="$(ssh_vm "pgrep -af 'curl.*127\\.0\\.0\\.1:11434/api/generate' || echo '(no matching curl)'" 2>&1 || true)"
  alerts=""
  # Do not match mediator request_id=700 — use explicit CUDA error patterns only.
  if echo "$vm_snip $host_snip $vm_err $host_err" | grep -qiE 'ERROR|401312|INVALID_IMAGE|FAILED|context canceled|499|cuCtxSynchronize rc=700|result\.status=700|CUDA_ERROR_ILLEGAL|mmq|GGML_ABORT|SIGABRT|fatal error|3884'; then
    alerts="**Alerts (keyword hit in this tick):** review snippets below."
  else
    alerts="*(no ERROR / 401312 / INVALID_IMAGE / cuCtxSynchronize rc=700 / context canceled / MMQ abort in filtered snippets)*"
  fi

  {
    echo "## Tick ${tick} — ${now_utc}"
    echo ""
    echo "${alerts}"
    echo ""
    echo "### VM — long-run curl (still running?)"
    echo '```text'
    echo "$curl_snip"
    echo '```'
    echo ""
    echo "### VM — \`/api/ps\`"
    echo '```json'
    echo "$ps_snip"
    echo '```'
    echo ""
    echo "### VM — journal (ollama, filtered tail)"
    echo '```text'
    echo "$vm_snip"
    echo '```'
    echo ""
    echo "### Host — \`/tmp/mediator.log\` (filtered tail)"
    echo '```text'
    echo "$host_snip"
    echo '```'
    echo ""
    echo "### Error trace — VM (E1–E5 keywords, boot journal)"
    echo '```text'
    echo "$vm_err"
    echo '```'
    echo ""
    echo "### Error trace — Host (mediator, E1–E5 keywords)"
    echo '```text'
    echo "$host_err"
    echo '```'
    echo ""
    echo "---"
    echo ""
  } >> "$OUT"
  echo "phase3_longrun_10min_monitor: tick=$tick wrote $OUT"
}

for ((i = 1; i <= N_TICKS; i++)); do
  append_tick
  (( i < N_TICKS )) && sleep "$INTERVAL"
done

{
  echo ""
  echo "## Monitor finished"
  echo "- **UTC:** $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "- **Ticks written:** ${tick}"
  echo "- **Output file:** \`$OUT\`"
} >> "$OUT"
echo "phase3_longrun_10min_monitor: done → $OUT"
