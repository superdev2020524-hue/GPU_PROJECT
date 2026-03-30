#!/usr/bin/env bash
# Run from dev workstation (phase3 dir) AFTER the 4h VM curl completes — captures dom0 mediator slice.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
SSHPASS="${SSHPASS:-$(python3 -c "import sys; sys.path.insert(0,'$SCRIPT_DIR'); from vm_config import MEDIATOR_PASSWORD; print(MEDIATOR_PASSWORD)")}"
export SSHPASS
MEDIATOR_HOST="${MEDIATOR_HOST:-10.25.33.10}"
TS="$(date +%Y%m%d_%H%M%S)"
OUT="${1:-./phase3_host_mediator_slice_${TS}.txt}"

sshpass -e ssh -n \
  -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
  "root@${MEDIATOR_HOST}" \
  "echo '=== mediator.log wc ==='; wc -l /tmp/mediator.log; echo '=== slice (grep; empty if no hits) ==='; grep -E '401312|700|INVALID|GEMM|module-load|HtoD|cuda-executor|INVALID_IMAGE|fail401312' /tmp/mediator.log 2>/dev/null | tail -800 || true; echo '=== fail401312.bin ==='; ls -la /tmp/fail401312.bin 2>&1 || true" \
  > "$OUT"
echo "Wrote $OUT"
