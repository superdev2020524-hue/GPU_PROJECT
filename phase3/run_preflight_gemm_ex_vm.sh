#!/usr/bin/env bash
# Workstation: copy test_gemm_ex_vm.c to VM, build it, and run it under a
# transient systemd context that matches the live Ollama service path.
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
VU="$(python3 -c "from vm_config import VM_USER; print(VM_USER)")"
SSHPASS="${SSHPASS:-$(python3 -c "from vm_config import VM_PASSWORD; print(VM_PASSWORD)")}"
export SSHPASS

REMOTE_BIN="/tmp/test_gemm_ex_vm"
REMOTE_SRC="/tmp/test_gemm_ex_vm.c"
REMOTE_RUNNER="/tmp/test_gemm_ex_vm_runner.sh"

echo "=== scp test_gemm_ex_vm.c -> ${VU}@${VH}:${REMOTE_SRC} ==="
sshpass -e scp -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  "$SCRIPT_DIR/guest-shim/test_gemm_ex_vm.c" "${VU}@${VH}:${REMOTE_SRC}"

echo "=== build on VM ==="
sshpass -e ssh -n -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  "${VU}@${VH}" \
  "printf '%s\n' '${SSHPASS}' | sudo -S bash -lc 'rm -f ${REMOTE_BIN} && gcc -O2 -std=c11 -Wall -o ${REMOTE_BIN} ${REMOTE_SRC} -ldl && echo BUILD_OK'"

echo "=== install service-equivalent runner on VM ==="
sshpass -e ssh -n -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  "${VU}@${VH}" \
  "cat > ${REMOTE_RUNNER} && chmod 755 ${REMOTE_RUNNER}" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

REMOTE_BIN="/tmp/test_gemm_ex_vm"
OLLAMA_ENV="$(systemctl show ollama -p Environment --value)"
OLLAMA_USER="$(systemctl show ollama -p User --value)"
OLLAMA_GROUP="$(systemctl show ollama -p Group --value)"

: "${OLLAMA_USER:=ollama}"
: "${OLLAMA_GROUP:=ollama}"

env_args=()
for key in \
  LD_LIBRARY_PATH \
  OLLAMA_LLM_LIBRARY \
  OLLAMA_LIBRARY_PATH \
  CUDA_TRANSPORT_TIMEOUT_SEC \
  VGPU_SHMEM_MIN_SPAN_KB \
  VGPU_ALLOW_MULTI_PROCESS_SHMEM \
  VGPU_HTOD_BAR1 \
  VGPU_MODULE_BAR1 \
  VGPU_HTOD_BAR1_SHADOW \
  OLLAMA_NO_MMAP
do
  val="$(printf '%s\n' "$OLLAMA_ENV" | tr ' ' '\n' | sed -n "s/^${key}=//p" | tail -n 1)"
  if [[ -n "$val" ]]; then
    env_args+=(-E "${key}=${val}")
  fi
done

echo "runner_user=${OLLAMA_USER}"
echo "runner_group=${OLLAMA_GROUP}"
printf 'runner_env=%s\n' "${env_args[*]}"

systemd-run --wait --collect --pipe \
  --uid="${OLLAMA_USER}" \
  --gid="${OLLAMA_GROUP}" \
  -p "AmbientCapabilities=CAP_SYS_ADMIN CAP_IPC_LOCK" \
  -p "CapabilityBoundingSet=CAP_SYS_ADMIN CAP_IPC_LOCK" \
  -p NoNewPrivileges=no \
  -p LimitMEMLOCK=infinity \
  "${env_args[@]}" \
  "${REMOTE_BIN}"
EOF

echo "=== run preflight (service-equivalent mediated libcublas) ==="
sshpass -e ssh -n -o PreferredAuthentications=password -o PubkeyAuthentication=no \
  "${VU}@${VH}" \
  "printf '%s\n' '${SSHPASS}' | sudo -S bash ${REMOTE_RUNNER}; echo exit_code=\$?"

echo "=== done ==="
