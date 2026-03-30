#!/usr/bin/env bash
# Workstation: copy test_gemm_ex_vm.c to VM, build, run with Ollama-equivalent LD_LIBRARY_PATH.
# Exit 0 only if preflight passes — run this before long model load /api/generate.
#
# Usage:
#   cd phase3 && export SSHPASS="$(python3 -c 'from vm_config import VM_PASSWORD; print(VM_PASSWORD)')"
#   ./run_preflight_gemm_ex_vm.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR${PYTHONPATH:+:$PYTHONPATH}"
VH="$(python3 -c "from vm_config import VM_HOST; print(VM_HOST)")"
VU="test-4"
SSHPASS="${SSHPASS:-$(python3 -c "from vm_config import VM_PASSWORD; print(VM_PASSWORD)")}"
export SSHPASS

REMOTE_BIN="/tmp/test_gemm_ex_vm"
REMOTE_SRC="/tmp/test_gemm_ex_vm.c"

echo "=== scp test_gemm_ex_vm.c -> ${VU}@${VH}:${REMOTE_SRC} ==="
sshpass -e scp -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  "$SCRIPT_DIR/guest-shim/test_gemm_ex_vm.c" "${VU}@${VH}:${REMOTE_SRC}"

echo "=== build on VM ==="
sshpass -e ssh -n -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  "${VU}@${VH}" \
  "gcc -O2 -std=c11 -Wall -o ${REMOTE_BIN} ${REMOTE_SRC} -ldl && echo BUILD_OK"

echo "=== run preflight (mediated libcublas) ==="
sshpass -e ssh -n -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  "${VU}@${VH}" \
  "export LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:\${LD_LIBRARY_PATH:-}; ${REMOTE_BIN}; echo exit_code=\$?"

echo "=== done ==="
