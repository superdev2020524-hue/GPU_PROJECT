#!/usr/bin/env bash
#
# Build libggml-cuda.so with Hopper (sm_90) support using Docker.
# Use this when the VM has no disk space for CUDA 12 or you prefer not to
# install the full toolkit on the build machine.
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
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${1:-$SCRIPT_DIR/out}"
IMAGE="${OLLAMA_CUDA_IMAGE:-nvidia/cuda:12.4.0-devel-ubuntu22.04}"
OLLAMA_SRC="${OLLAMA_SRC:-}"

echo "=== Build libggml-cuda.so (Hopper) via Docker ==="
echo "Image: $IMAGE"
echo "Output dir: $OUT_DIR"
echo ""

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

# Build inside container; /tmp/ollama is the mounted source (no git inside container).
docker run --rm \
  -e "CMAKE_CUDA_ARCHITECTURES=90" \
  -e "DEBIAN_FRONTEND=noninteractive" \
  -v "$OUT_DIR:/out" \
  -v "$MOUNT_OLLAMA:/tmp/ollama:ro" \
  "$IMAGE" \
  bash -c '
    set -e
    apt-get update -qq && apt-get install -y -qq git cmake golang-go > /dev/null
    if [ ! -f /tmp/ollama/CMakeLists.txt ]; then echo "ERROR: /tmp/ollama not valid"; exit 1; fi
    cp -a /tmp/ollama /tmp/ollama-build
    cd /tmp/ollama-build
    export CMAKE_CUDA_ARCHITECTURES=90
    # Try top-level make first (Ollama may use Makefile or go generate)
    make -j "$(nproc)" 2>/dev/null || true
    F=$(find . -name "libggml-cuda.so" -type f 2>/dev/null | head -1)
    # Fallback: run go generate to trigger native build, then go build
    if [ -z "$F" ] && [ -f go.mod ]; then
      go generate ./... 2>/dev/null || true
      go build -o /dev/null . 2>/dev/null || true
      F=$(find . -name "libggml-cuda.so" -type f 2>/dev/null | head -1)
    fi
    # Fallback: configure and build with cmake directly
    if [ -z "$F" ] && [ -f CMakeLists.txt ]; then
      mkdir -p build && cd build
      cmake .. -DCMAKE_CUDA_ARCHITECTURES=90 2>/dev/null || true
      cmake --build . -j "$(nproc)" 2>/dev/null || true
      F=$(find . -name "libggml-cuda.so" -type f 2>/dev/null | head -1)
    fi
    if [ -n "$F" ]; then
      cp "$F" /out/libggml-cuda.so
      echo "Built: /out/libggml-cuda.so"
    else
      echo "ERROR: libggml-cuda.so not found after make/go generate/cmake"
      find . -name "*.so" -type f 2>/dev/null | head -20
      exit 1
    fi
  '

if [ -f "$OUT_DIR/libggml-cuda.so" ]; then
  echo ""
  echo "Success. Library: $OUT_DIR/libggml-cuda.so"
  ls -la "$OUT_DIR/libggml-cuda.so"
  echo ""
  echo "Deploy to test-3 VM:"
  echo "  cd phase3 && python3 deploy_libggml_cuda_hopper.py $OUT_DIR/libggml-cuda.so"
else
  echo "Build failed: $OUT_DIR/libggml-cuda.so not found"
  exit 1
fi
