#!/usr/bin/env bash
# Run from workstation (phase3/). Backs up + truncates host mediator, restarts mediator,
# restarts VM ollama, starts run_longrun_4h_capture.sh on the VM (4h curl + full logs).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"

TS="$(date +%Y%m%d_%H%M%S)"
echo "=== PHASE3_LONGRUN_TS=$TS ==="

MH="$(python3 -c "from vm_config import MEDIATOR_HOST; print(MEDIATOR_HOST)")"
VH="$(python3 -c "from vm_config import VM_HOST; print(VM_HOST)")"

export SSHPASS="$(python3 -c "from vm_config import MEDIATOR_PASSWORD; print(MEDIATOR_PASSWORD)")"

echo "=== Host: backup mediator.log ==="
sshpass -e ssh -n -o PreferredAuthentications=password -o PubkeyAuthentication=no -o ConnectTimeout=20 \
  "root@${MH}" \
  "cp -a /tmp/mediator.log /tmp/mediator.log.bak.${TS} 2>/dev/null; ls -la /tmp/mediator.log.bak.${TS} 2>/dev/null || echo '(no prior mediator.log)'; true"

echo "=== Host: truncate + restart mediator ==="
MEDIATOR_TRUNCATE_LOG=1 ./host_restart_mediator.sh

VM_PW="$(python3 -c "from vm_config import VM_PASSWORD; print(VM_PASSWORD)")"
export SSHPASS="$VM_PW"

echo "=== VM: restart ollama (connect_vm.py — handles sudo prompt) ==="
python3 connect_vm.py 'sudo systemctl restart ollama && sleep 6 && systemctl is-active ollama && systemctl --no-pager status ollama | head -12'

echo "=== VM: mediated cublasGemmEx preflight (abort long run if not perfect) ==="
export SSHPASS="$VM_PW"
./run_preflight_gemm_ex_vm.sh

echo "=== VM: upload + start 4h capture ==="
sshpass -e scp -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  run_longrun_4h_capture.sh "test-4@${VH}:/tmp/run_longrun_4h_capture.sh"

python3 connect_vm.py "chmod +x /tmp/run_longrun_4h_capture.sh && export PHASE3_LONGRUN_TS='${TS}' && exec /tmp/run_longrun_4h_capture.sh"

echo ""
echo "=== Workstation: 10-minute Markdown monitor (background, same TS) ==="
export PHASE3_LONGRUN_TS="$TS"
export SSHPASS="$(python3 -c "from vm_config import MEDIATOR_PASSWORD; print(MEDIATOR_PASSWORD)")"
MON_LOG="/tmp/phase3_monitor_${TS}.log"
nohup ./phase3_longrun_10min_monitor.sh >> "$MON_LOG" 2>&1 &
echo "monitor_pid=$!  monitor_log=$MON_LOG  md_out=$SCRIPT_DIR/LONGRUN_SESSION_${TS}.md"

echo ""
echo "=== Done starting long run ==="
echo "Session TS: $TS"
echo "After ~4h (or on failure), on workstation:"
echo "  ./collect_host_longrun_slice.sh"
echo "On VM, artifacts: /tmp/phase3_longrun_${TS}/"
echo "10-min snapshots: LONGRUN_SESSION_${TS}.md (tail -f or open in editor)"
