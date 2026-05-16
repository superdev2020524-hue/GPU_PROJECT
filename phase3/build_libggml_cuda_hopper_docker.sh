#!/usr/bin/env bash
#
# Build libggml-cuda.so with Hopper (sm_90) support using Docker.
# Also sets GGML_CUDA_GRAPHS=OFF and optional -DGGML_CUDA_FORCE_CUBLAS=ON|OFF via
# PHASE3_GGML_CUDA_FORCE_CUBLAS (default ON) to avoid E5 or to bisect SIGFPE (see BUILD_AND_DEPLOY).
# PHASE3_GGML_CUDA_FA: default ON (upstream). Set OFF only for experimental E6 mitigation
# (requires patches/phase3_ollama_cmake_ggml_cuda_fa_overridable.patch); see BUILD_AND_DEPLOY.
# patches/phase3_ggml_fattn_launch_avoid_host_sigfpe.patch: host-side integer div-by-zero in launch_fattn
# (stream_k / wave efficiency) — applied automatically when present; see E6 in SYSTEMATIC_ERROR_TRACKING_PLAN.md.
# patches/phase3_ggml_cuda_init_nsm_fallback.patch: ggml_cuda_init nsm=132 if multiProcessorCount<=0 — see BUILD_AND_DEPLOY §1 item 6.
# patches/phase3_ggml_mul_mat_cublas_chunk_m_phase3.patch: chunk NVIDIA FP16 GemmEx m dimension for mediated GEMM_EX span (E7 OOB) — §1 item 7.
#
# Requires: Docker, and either network (to clone) or an existing Ollama tree.
# Clone is done on the HOST to avoid TLS/gnutls issues inside the container.
#
# Usage:
#   ./build_libggml_cuda_hopper_docker.sh [OUTPUT_DIR]
#   OLLAMA_SRC=/path/to/ollama ./build_libggml_cuda_hopper_docker.sh [OUTPUT_DIR]
# Default OUTPUT_DIR is ./out. If OLLAMA_SRC is set, that dir is mounted (no clone).
# Otherwise we clone on the host into ./ollama-src, then mount that into the container.
#
# If you run with sudo, preserve OLLAMA_SRC and optional PHASE3_GGML_CUDA_FORCE_CUBLAS:
#   sudo -E env OLLAMA_SRC=/path/to/patched/ollama PHASE3_GGML_CUDA_FORCE_CUBLAS=OFF ./build_libggml_cuda_hopper_docker.sh ./out
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR/out}"
IMAGE="${OLLAMA_CUDA_IMAGE:-nvidia/cuda:12.4.0-devel-ubuntu22.04}"
OLLAMA_SRC="${OLLAMA_SRC:-}"
PATCH_FILE="${SCRIPT_DIR}/patches/phase3_ggml_cuda_force_cc90.patch"
PATCH_FA_OVERRIDABLE="${SCRIPT_DIR}/patches/phase3_ollama_cmake_ggml_cuda_fa_overridable.patch"
PATCH_FATTN_SIGFPE="${SCRIPT_DIR}/patches/phase3_ggml_fattn_launch_avoid_host_sigfpe.patch"
PATCH_NSM_FALLBACK="${SCRIPT_DIR}/patches/phase3_ggml_cuda_init_nsm_fallback.patch"
PATCH_MUL_MAT_CHUNK="${SCRIPT_DIR}/patches/phase3_ggml_mul_mat_cublas_chunk_m_phase3.patch"

echo "=== Build libggml-cuda.so (Hopper, graphs OFF) via Docker ==="
echo "Image: $IMAGE"
echo "Output dir: $OUT_DIR"
PHASE3_GGML_CUDA_FORCE_CUBLAS="${PHASE3_GGML_CUDA_FORCE_CUBLAS:-ON}"
PHASE3_GGML_CUDA_FA="${PHASE3_GGML_CUDA_FA:-ON}"
echo "PHASE3_GGML_CUDA_FORCE_CUBLAS=$PHASE3_GGML_CUDA_FORCE_CUBLAS (ON=compile-time cuBLAS matmul; OFF=bisect vs E5 MMQ / SIGFPE)"
echo "PHASE3_GGML_CUDA_FA=$PHASE3_GGML_CUDA_FA (OFF=experimental E6 mitigation — see BUILD_AND_DEPLOY; ON=upstream default)"
echo ""

# Docker must be usable. Typical failure: user not in group 'docker' (permission denied on socket).
if ! docker info >/dev/null 2>&1; then
  echo "ERROR: Cannot talk to the Docker daemon." >&2
  echo "  - If you see 'permission denied' on /var/run/docker.sock:" >&2
  echo "      sudo usermod -aG docker \"\$USER\"   # then log out and back in" >&2
  echo "    or run this script as root (Docker works for root):" >&2
  echo "      sudo $SCRIPT_DIR/build_libggml_cuda_hopper_docker.sh $OUT_DIR" >&2
  echo "  - If Docker is not installed, install docker.io / Docker Engine first." >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Clone on HOST if no OLLAMA_SRC (avoids gnutls_handshake failures inside container)
if [ -n "$OLLAMA_SRC" ]; then
  if [ ! -d "$OLLAMA_SRC" ] || [ ! -f "$OLLAMA_SRC/CMakeLists.txt" ]; then
    echo "Error: OLLAMA_SRC=$OLLAMA_SRC is not a valid ollama repo (no CMakeLists.txt)."
    exit 1
  fi
  echo "Using existing Ollama tree: $OLLAMA_SRC"
  MOUNT_OLLAMA="$OLLAMA_SRC"
else
  OLLAMA_CLONE="$SCRIPT_DIR/ollama-src"
  if [ ! -d "$OLLAMA_CLONE/.git" ]; then
    echo "Cloning ollama/ollama on host (avoids TLS issues in container)..."
    git clone --depth 1 https://github.com/ollama/ollama.git "$OLLAMA_CLONE"
  else
    echo "Using existing clone: $OLLAMA_CLONE"
    (cd "$OLLAMA_CLONE" && git fetch --depth 1 origin main && git checkout -q origin/main 2>/dev/null) || true
  fi
  MOUNT_OLLAMA="$OLLAMA_CLONE"
fi

# Host tree may be read-only for this user (e.g. root-owned clone). Patch is applied *inside* the container.
if [ -f "$PATCH_FILE" ]; then
  echo "Phase3 patch will be applied inside the container: $PATCH_FILE"
else
  echo "Note: no patch file at $PATCH_FILE — CC=9.0 shim patch skipped."
fi
if [ -f "$PATCH_FA_OVERRIDABLE" ]; then
  echo "FA overridable patch: $PATCH_FA_OVERRIDABLE"
else
  echo "Note: no patch at $PATCH_FA_OVERRIDABLE — -DGGML_CUDA_FA=OFF may be ignored by Ollama CMakeLists."
fi
if [ -f "$PATCH_FATTN_SIGFPE" ]; then
  echo "E6 fattn host SIGFPE patch: $PATCH_FATTN_SIGFPE"
fi
if [ -f "$PATCH_NSM_FALLBACK" ]; then
  echo "E6 nsm fallback patch: $PATCH_NSM_FALLBACK"
fi
if [ -f "$PATCH_MUL_MAT_CHUNK" ]; then
  echo "E7 mul_mat GemmEx m-chunk patch: $PATCH_MUL_MAT_CHUNK"
fi

# Build inside container; /tmp/ollama is the mounted source (read-only mount → copy inside).
# Mount patch files separately so we do not need write access to the ollama tree on the host.
PATCH_MOUNT=()
if [ -f "$PATCH_FILE" ]; then
  PATCH_MOUNT+=( -v "$PATCH_FILE:/phase3/patch_cc90.patch:ro" )
fi
if [ -f "$PATCH_FA_OVERRIDABLE" ]; then
  PATCH_MOUNT+=( -v "$PATCH_FA_OVERRIDABLE:/phase3/patch_fa_cmake.patch:ro" )
fi
if [ -f "$PATCH_FATTN_SIGFPE" ]; then
  PATCH_MOUNT+=( -v "$PATCH_FATTN_SIGFPE:/phase3/patch_fattn_sigfpe.patch:ro" )
fi
if [ -f "$PATCH_NSM_FALLBACK" ]; then
  PATCH_MOUNT+=( -v "$PATCH_NSM_FALLBACK:/phase3/patch_nsm_fallback.patch:ro" )
fi
if [ -f "$PATCH_MUL_MAT_CHUNK" ]; then
  PATCH_MOUNT+=( -v "$PATCH_MUL_MAT_CHUNK:/phase3/patch_mul_mat_chunk.patch:ro" )
fi

DOCKER_INNER=$(cat << 'EOS'
set -euo pipefail
FC="${PHASE3_GGML_CUDA_FORCE_CUBLAS:-ON}"
case "$FC" in
  OFF|off|0|no|false|FALSE) CUBLAS=-DGGML_CUDA_FORCE_CUBLAS=OFF ;;
  *) CUBLAS=-DGGML_CUDA_FORCE_CUBLAS=ON ;;
esac
echo "Container: PHASE3_GGML_CUDA_FORCE_CUBLAS=$FC -> $CUBLAS"
FA="${PHASE3_GGML_CUDA_FA:-ON}"
case "$FA" in
  OFF|off|0|no|false|FALSE) FAFLAG=-DGGML_CUDA_FA=OFF ;;
  *) FAFLAG="" ;;
esac
echo "Container: PHASE3_GGML_CUDA_FA=$FA -> ${FAFLAG:-<omit -D, use CMake default>}"
apt-get update -qq && apt-get install -y -qq git cmake ninja-build golang-go build-essential patch > /dev/null
if [ ! -f /tmp/ollama/CMakeLists.txt ]; then echo "ERROR: /tmp/ollama not valid"; exit 1; fi
rm -rf /tmp/ollama-build
cp -a /tmp/ollama /tmp/ollama-build
cd /tmp/ollama-build
if [ -f /phase3/patch_cc90.patch ]; then
  echo "Applying Phase3 patch inside container (best-effort)..."
  patch -p1 -N --dry-run -i /phase3/patch_cc90.patch >/dev/null 2>&1 && patch -p1 -i /phase3/patch_cc90.patch || \
    echo "  (patch skipped or already applied)"
fi
if [ -f /phase3/patch_fa_cmake.patch ]; then
  echo "Applying Phase3 GGML_CUDA_FA overridable patch (best-effort)..."
  patch -p1 -N --dry-run -i /phase3/patch_fa_cmake.patch >/dev/null 2>&1 && patch -p1 -i /phase3/patch_fa_cmake.patch || \
    echo "  (FA CMake patch skipped or already applied)"
fi
if [ -f /phase3/patch_fattn_sigfpe.patch ]; then
  echo "Applying Phase3 launch_fattn host SIGFPE guard patch (best-effort)..."
  patch -p1 -N --dry-run -i /phase3/patch_fattn_sigfpe.patch >/dev/null 2>&1 && patch -p1 -i /phase3/patch_fattn_sigfpe.patch || \
    echo "  (fattn SIGFPE patch skipped or already applied)"
fi
if [ -f /phase3/patch_nsm_fallback.patch ]; then
  echo "Applying Phase3 ggml_cuda_init nsm fallback patch (best-effort)..."
  patch -p1 -N --dry-run -i /phase3/patch_nsm_fallback.patch >/dev/null 2>&1 && patch -p1 -i /phase3/patch_nsm_fallback.patch || \
    echo "  (nsm fallback patch skipped or already applied)"
fi
if [ -f /phase3/patch_mul_mat_chunk.patch ]; then
  echo "Applying Phase3 mul_mat cublas GemmEx m-chunk patch (best-effort)..."
  # Use -l: Ollama main sometimes drifts whitespace vs hunks; a strict dry-run caused silent skips
  # and unpatched row_diff Gemm (E7 — May 2026).
  patch -p1 -l -i /phase3/patch_mul_mat_chunk.patch || \
    echo "  (mul_mat chunk patch failed — see patch output above)"
fi
rm -rf build
mkdir -p build
cd build
cmake .. \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=90 \
  -DGGML_CUDA_GRAPHS=OFF \
  ${FAFLAG} \
  ${CUBLAS}
if ! cmake --build . -j "$(nproc)" --target ggml-cuda; then
  echo "WARN: target ggml-cuda failed; building default targets..."
  cmake --build . -j "$(nproc)"
fi
F=""
if [ -f lib/ollama/libggml-cuda.so ]; then
  F=lib/ollama/libggml-cuda.so
else
  F=$(find . -name "libggml-cuda.so" -type f 2>/dev/null | head -1)
fi
if [ -z "$F" ] || [ ! -f "$F" ]; then
  echo "ERROR: libggml-cuda.so not found after cmake build"
  find . -name "*.so" -type f 2>/dev/null | head -30
  exit 1
fi
cp "$F" /out/libggml-cuda.so
echo "Built: /out/libggml-cuda.so"
EOS
)

docker run --rm \
  -e "DEBIAN_FRONTEND=noninteractive" \
  -e "PHASE3_GGML_CUDA_FORCE_CUBLAS=${PHASE3_GGML_CUDA_FORCE_CUBLAS}" \
  -e "PHASE3_GGML_CUDA_FA=${PHASE3_GGML_CUDA_FA}" \
  -v "$OUT_DIR:/out" \
  -v "$MOUNT_OLLAMA:/tmp/ollama:ro" \
  "${PATCH_MOUNT[@]}" \
  "$IMAGE" \
  bash -lc "$DOCKER_INNER"

if [ -f "$OUT_DIR/libggml-cuda.so" ]; then
  echo ""
  echo "Success. Library: $OUT_DIR/libggml-cuda.so"
  ls -la "$OUT_DIR/libggml-cuda.so"
  echo ""
  echo "Deploy to VM (vm_config):"
  echo "  cd phase3 && python3 deploy_libggml_cuda_hopper.py $OUT_DIR/libggml-cuda.so"
else
  echo "Build failed: $OUT_DIR/libggml-cuda.so not found"
  exit 1
fi
