#!/usr/bin/env bash
set -euo pipefail

RUN_BASE="${1:-/tmp/e3_bounded_20260328_053032}"
OUT_FILE="${2:-/home/david/Downloads/gpu/phase3/SESSION_RESUME_GUIDE/monitor_e3_20260328_053032.log}"
INTERVAL_SEC="${INTERVAL_SEC:-600}"

VM_USER="test-4"
VM_HOST="10.25.33.12"
HOST_USER="root"
HOST_HOST="10.25.33.10"
PASS="Calvin@123"

touch "$OUT_FILE"

while true; do
  {
    echo "===== $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
    echo "RUN_BASE=$RUN_BASE"
    echo
    echo "[vm status]"
    sshpass -p "$PASS" ssh -n -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no "${VM_USER}@${VM_HOST}" \
      "printf 'alive_pids '; ps -fp \$(cat '$RUN_BASE/curl.pid' 2>/dev/null) \$(cat '$RUN_BASE/journal_follow.pid' 2>/dev/null) 2>/dev/null | tail -n +2 || true; echo '--- journal'; tail -n 40 '$RUN_BASE/journal_follow.log' 2>/dev/null || true; echo '--- curl stdout'; tail -n 20 '$RUN_BASE/curl_generate.stdout.log' 2>/dev/null || true; echo '--- curl stderr'; tail -n 20 '$RUN_BASE/curl_generate.stderr.log' 2>/dev/null || true"
    echo
    echo "[host status]"
    sshpass -p "$PASS" ssh -n -o StrictHostKeyChecking=no -o PreferredAuthentications=password -o PubkeyAuthentication=no "${HOST_USER}@${HOST_HOST}" \
      "python3 -c \"import pathlib,re; lines=pathlib.Path('/tmp/mediator.log').read_text(errors='replace').splitlines(); hits=[l for l in lines if re.search(r'HtoD progress|401312|INVALID_IMAGE|rc=700|CUDA_ERROR_ILLEGAL_ADDRESS|FAILED|call_id=0xb5|call_id=0x26', l)]; print('\\n'.join(hits[-60:]))\""
    echo
  } >> "$OUT_FILE" 2>&1
  sleep "$INTERVAL_SEC"
done
