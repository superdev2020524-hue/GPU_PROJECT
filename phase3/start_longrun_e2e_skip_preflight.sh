#!/usr/bin/env bash
# Same as reset_and_start_longrun_4h.sh but skips run_preflight_gemm_ex_vm.sh (slow / can block mediator).
# Use when you want immediate 4h capture after mediator + ollama reset (optionally run preflight manually first).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

# Always use a fresh timestamp (stale PHASE3_LONGRUN_TS in the shell breaks new sessions).
TS="$(date +%Y%m%d_%H%M%S)"
if [[ -n "${PHASE3_LONGRUN_TS_FORCE:-}" ]]; then TS="$PHASE3_LONGRUN_TS_FORCE"; fi
export PHASE3_LONGRUN_TS="$TS"
echo "=== PHASE3_LONGRUN_TS=$TS ==="

MH="$(python3 -c "from vm_config import MEDIATOR_HOST; print(MEDIATOR_HOST)")"
VH="$(python3 -c "from vm_config import VM_HOST; print(VM_HOST)")"

export SSHPASS="$(python3 -c "from vm_config import MEDIATOR_PASSWORD; print(MEDIATOR_PASSWORD)")"

echo "=== Host: backup mediator.log ==="
sshpass -e ssh -n -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=25 \
  "root@${MH}" \
  "cp -a /tmp/mediator.log /tmp/mediator.log.bak.${TS} 2>/dev/null; ls -la /tmp/mediator.log.bak.${TS} 2>/dev/null || echo '(no prior mediator.log)'; true"

echo "=== Host: truncate + restart mediator ==="
MEDIATOR_TRUNCATE_LOG=1 ./host_restart_mediator.sh

VM_PW="$(python3 -c "from vm_config import VM_PASSWORD; print(VM_PASSWORD)")"
export SSHPASS="$VM_PW"

echo "=== VM: restart ollama ==="
python3 connect_vm.py 'sudo systemctl restart ollama && sleep 8 && systemctl is-active ollama && systemctl --no-pager status ollama | head -14'

echo "=== VM: upload + start 4h capture (PHASE3_LONGRUN_TS=$TS) ==="
sshpass -e scp -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  run_longrun_4h_capture.sh "test-4@${VH}:/tmp/run_longrun_4h_capture.sh"

python3 connect_vm.py "chmod +x /tmp/run_longrun_4h_capture.sh && export PHASE3_LONGRUN_TS='${TS}' && exec /tmp/run_longrun_4h_capture.sh"

echo ""
echo "=== Workstation: 10-minute Markdown monitor (background) ==="
export SSHPASS="$(python3 -c "from vm_config import MEDIATOR_PASSWORD; print(MEDIATOR_PASSWORD)")"
MON_LOG="/tmp/phase3_monitor_${TS}.log"
nohup ./phase3_longrun_10min_monitor.sh >> "$MON_LOG" 2>&1 &
echo "monitor_pid=$!  monitor_log=$MON_LOG  md_out=$SCRIPT_DIR/LONGRUN_SESSION_${TS}.md"

echo ""
echo "=== Done ==="
echo "TS=$TS"
echo "VM artifacts: /tmp/phase3_longrun_${TS}/"
echo "Session log: LONGRUN_SESSION_${TS}.md (includes Error trace E1–E5 sections)"
echo "After run: ./collect_host_longrun_slice.sh"
