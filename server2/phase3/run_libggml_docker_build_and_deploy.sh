#!/usr/bin/env bash
#
# Phase 3: apply CC-9.0 GGML patch, build libggml-cuda.so (Hopper) via Docker, deploy to VM.
#
# Prerequisites:
#   - Docker (run this script with sudo if your user cannot use `docker` without it)
#   - Network (pull nvidia/cuda image, clone Ollama if needed)
#   - phase3/vm_config.py — VM target for deploy_libggml_cuda_hopper.py
#
# Usage:
#   cd /path/to/gpu/phase3
#   ./run_libggml_docker_build_and_deploy.sh
#
# Optional:
#   OLLAMA_SRC=/path/to/existing/ollama ./run_libggml_docker_build_and_deploy.sh
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OLLAMA_SRC="${OLLAMA_SRC:-$SCRIPT_DIR/ollama-src-phase3}"
PATCH="$SCRIPT_DIR/patches/phase3_ggml_cuda_force_cc90.patch"
OUT_DIR="$SCRIPT_DIR/out"

if [[ ! -f "$PATCH" ]]; then
  echo "Missing patch: $PATCH"
  exit 1
fi

if [[ ! -f "$OLLAMA_SRC/go.mod" ]]; then
  echo "=== Cloning ollama/ollama (shallow) into $OLLAMA_SRC ==="
  rm -rf "$OLLAMA_SRC"
  git clone --depth 1 https://github.com/ollama/ollama.git "$OLLAMA_SRC"
fi

GGML_CU="$OLLAMA_SRC/ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu"
if [[ ! -f "$GGML_CU" ]]; then
  echo "=== Initializing submodules (ggml sources) ==="
  (cd "$OLLAMA_SRC" && git submodule update --init --recursive --depth 1)
fi
if [[ ! -f "$GGML_CU" ]]; then
  echo "ERROR: Expected file missing: $GGML_CU"
  echo "Use OLLAMA_SRC= pointing to a full Ollama tree (e.g. copy from the VM under /home/test-4/ollama)."
  exit 1
fi

echo "=== Applying Phase3 CC patch ==="
if (cd "$OLLAMA_SRC" && patch -p1 --forward --reject-file=- < "$PATCH"); then
  echo "Patch applied (or already applied)."
else
  if grep -q 'dev_ctx->major = 9;' "$GGML_CU" 2>/dev/null; then
    echo "Patch already present (major=9)."
  else
    echo "Patch failed. Edit $GGML_CU manually — replace prop.major/prop.minor assignment with major=9 minor=0"
    echo "See patches/phase3_ggml_cuda_force_cc90.patch and TRACE_E2_COMPUTE_89_ROOT_CAUSE.md"
    exit 1
  fi
fi

echo "=== Docker build (Hopper sm_90) -> $OUT_DIR/libggml-cuda.so ==="
export OLLAMA_SRC
"$SCRIPT_DIR/build_libggml_cuda_hopper_docker.sh" "$OUT_DIR"

echo "=== Deploy to VM ==="
python3 "$SCRIPT_DIR/deploy_libggml_cuda_hopper.py" "$OUT_DIR/libggml-cuda.so"

echo "=== Done. Verify on VM: journalctl -u ollama | grep inference compute ==="
