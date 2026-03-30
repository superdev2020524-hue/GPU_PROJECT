#!/usr/bin/env bash
# Reliable dom0 mediator restart from the dev workstation.
# Fixes: ssh without -n can block on stdin; long one-liners hit tool timeouts.
#
# Usage (from repo):
#   cd phase3 && export SSHPASS="$(python3 -c 'from vm_config import MEDIATOR_PASSWORD; print(MEDIATOR_PASSWORD)')"
#   ./host_restart_mediator.sh
#
# Logs:
#   Restart alone does NOT clear /tmp/mediator.log — nohup uses >> (append).
#   Old lines remain until you truncate or rotate. To start with an empty log:
#     MEDIATOR_TRUNCATE_LOG=1 ./host_restart_mediator.sh
#
# Note on cuda_executor.c:
#   That file is the host-side CUDA replay *source* linked into mediator_phase3.
#   It is copied from the workspace only when rebuilding so dom0 matches repo
#   changes (e.g. logging). It is not an "error log" artifact.
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
SSHPASS="${SSHPASS:-$(python3 -c "import sys; sys.path.insert(0,'$SCRIPT_DIR'); from vm_config import MEDIATOR_PASSWORD; print(MEDIATOR_PASSWORD)")}"
export SSHPASS

MEDIATOR_HOST="${MEDIATOR_HOST:-10.25.33.10}"
TRUNC="${MEDIATOR_TRUNCATE_LOG:-0}"

# Remote body: TRUNC expanded locally; \$ escapes remote shell vars.
exec sshpass -e ssh -n \
  -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o ServerAliveInterval=5 \
  "root@${MEDIATOR_HOST}" \
  "cd /root/phase3 && killall -9 mediator_phase3 2>/dev/null || true
sleep 1
if [ \"${TRUNC}\" = \"1\" ]; then : > /tmp/mediator.log; echo '(truncated /tmp/mediator.log)'; fi
nohup ./mediator_phase3 >> /tmp/mediator.log 2>&1 &
echo \"mediator_pid=\$!\"
sleep 2
if pgrep -x mediator_phase3 >/dev/null; then
  pgrep -a mediator_phase3
  echo 'OK: mediator_phase3 running'
  exit 0
fi
echo 'FAIL: mediator_phase3 not running' >&2
tail -30 /tmp/mediator.log >&2 || true
exit 1"
