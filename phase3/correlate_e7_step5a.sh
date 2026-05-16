#!/usr/bin/env bash
# Step 5a (SYSTEMATIC_ERROR_TRACKING_PLAN.md) — host log slices for E7 vs E4 correlation.
# Run on dom0 after a guest repro; align wall time with VM journalctl.
#
# Executor note: cuda_executor.c logs "cublasGemmEx rc=" when rc != SUCCESS (or verbose).
# Successful GemmEx may leave no matching line; E7 (rc=13) should log if stderr is in this file.
#
# From a workstation (no sshpass):  CONNECT_HOST_FORCE_PEXPECT=1 python3 connect_host.py 'bash /tmp/correlate_e7_step5a.sh'
# (scp this script to dom0 /tmp first).
# Usage:
#   bash correlate_e7_step5a.sh
#   MEDIATOR_LOG=/var/log/... VM_ID=6 bash correlate_e7_step5a.sh
#
# Log source: cuda_executor.c prints GEMM_EX lines to stderr; typical nohup is 2>&1 to mediator.log.

set -euo pipefail

LOG="${MEDIATOR_LOG:-/tmp/mediator.log}"
VID="${VM_ID:-6}"

if [[ ! -r "$LOG" ]]; then
  echo "cannot read $LOG" >&2
  exit 1
fi

echo "=== Checkpoint C — 401312 / INVALID_IMAGE (tail hits) ==="
grep -F '401312' "$LOG" | tail -5 || true
grep -F 'INVALID_IMAGE' "$LOG" | tail -3 || true

echo
echo "=== vm_id=${VID} — cublasGemmEx rc= line (E7 often rc=13); GEMM_EX dims after sync failures ==="
grep -F "[cuda-executor] vm_id=${VID} cublasGemmEx" "$LOG" | tail -40 || true
grep -F "[cuda-executor] vm_id=${VID} GEMM_EX " "$LOG" | tail -40 || true

echo
echo "=== vm_id=${VID} — after cublasGemmEx: cuCtxSynchronize (E4-like on single GemmEx path) ==="
grep -F "vm_id=${VID} after cublasGemmEx" "$LOG" | tail -25 || true

echo "=== vm_id=${VID} — MEDIATOR call_id=0xb5 (CUDA_CALL_CUBLAS_GEMM_EX); result.status is CUresult, not cublasStatus_t ==="
grep -F "vm_id=${VID}" "$LOG" | grep -F "call_id=0xb5" | tail -30 || true

echo
echo "=== vm_id=${VID} — MEDIATOR call_id=0x26 (CUDA_CALL_CTX_SYNCHRONIZE); non-zero result.status = host sync failure (E7 / guest STATUS_ERROR) ==="
grep -F "vm_id=${VID}" "$LOG" | grep -F "call_id=0x26" | tail -30 || true
echo "(grep non-zero sync:) "
grep -F "vm_id=${VID}" "$LOG" | grep -F "call_id=0x26" | grep -v "result.status=0" | tail -15 || true

echo
echo "=== vm_id=${VID} — correlate guest seq: grep request_id=<seq> for same window as journal STATUS_ERROR ==="
echo "    Example: guest [cuda-transport] seq=841 -> lines below (if logged for vm ${VID})"
grep -F "vm_id=${VID}" "$LOG" | grep -F "request_id=841" | tail -20 || true

echo
echo "=== vm_id=${VID} — batched path (E4 registry) ==="
grep -E "vm_id=${VID}.*cublasGemmBatchedEx|vm_id=${VID}.*GEMM_BATCHED|after cublasGemmBatchedEx.*vm_id=${VID}" "$LOG" | tail -25 || true
