#!/bin/bash
# Run this script ON THE VM (as root or with sudo) to make Ollama's runner
# subprocess load our vGPU shims so discovery sees one GPU and Ollama selects GPU mode.
#
# Root cause: the runner may not inherit LD_PRELOAD and uses LD_LIBRARY_PATH
# (e.g. /usr/local/lib/ollama then .../cuda_v12). If it finds real libcuda/libnvidia-ml
# there, it gets 0 devices. Symlink both the parent ollama dir and cuda_v12 to our shims.

set -e

# Where our vGPU shims live on the VM (adjust if you use /usr/lib64 instead)
VGPU_LIB="${VGPU_LIB:-/opt/vgpu/lib}"
OLLAMA_LIB="${OLLAMA_LIB:-/usr/local/lib/ollama}"
OLLAMA_CUDA_V12="${OLLAMA_CUDA_V12:-/usr/local/lib/ollama/cuda_v12}"

echo "=== Symlink Ollama libs to vGPU shims (for GPU mode) ==="
echo "  VGPU_LIB=$VGPU_LIB"
echo "  OLLAMA_LIB=$OLLAMA_LIB"
echo "  OLLAMA_CUDA_V12=$OLLAMA_CUDA_V12"

if [[ ! -d "$OLLAMA_CUDA_V12" ]]; then
  echo "ERROR: Ollama cuda_v12 dir not found: $OLLAMA_CUDA_V12"
  exit 1
fi

for shim in libcuda.so.1 libcudart.so.12 libnvidia-ml.so.1; do
  if [[ ! -e "$VGPU_LIB/$shim" ]]; then
    echo "  Skip $shim (not found in $VGPU_LIB)"
    continue
  fi
  # Ollama may ship libcudart as libcudart.so.12.8.90 etc.; symlink the one that exists
  if [[ "$shim" == libcudart.so.12 ]]; then
    found=
    for f in "$OLLAMA_CUDA_V12"/libcudart.so.12*; do
      [[ -e "$f" ]] || continue
      found=1
      name=$(basename "$f")
      if [[ -L "$f" ]]; then
        sudo rm -f "$f"
      else
        sudo mv -n "$f" "${f}.backup"
      fi
      sudo ln -sf "$VGPU_LIB/$shim" "$OLLAMA_CUDA_V12/$name"
      echo "  Linked $name -> $VGPU_LIB/$shim"
    done
    if [[ -z "$found" ]]; then
      sudo ln -sf "$VGPU_LIB/$shim" "$OLLAMA_CUDA_V12/$shim"
      echo "  Linked $shim -> $VGPU_LIB/$shim"
    fi
    continue
  fi
  target="$OLLAMA_CUDA_V12/$shim"
  if [[ -e "$target" ]]; then
    if [[ -L "$target" ]]; then
      sudo rm -f "$target"
    else
      sudo mv -n "$target" "${target}.backup"
    fi
  fi
  sudo ln -sf "$VGPU_LIB/$shim" "$target"
  echo "  Linked $shim -> $VGPU_LIB/$shim"
done

# Also symlink in parent ollama dir (runner often looks here first)
for shim in libcuda.so.1 libnvidia-ml.so.1; do
  if [[ ! -e "$VGPU_LIB/$shim" ]]; then continue; fi
  target="$OLLAMA_LIB/$shim"
  if [[ -e "$target" && ! -L "$target" ]]; then
    sudo mv -n "$target" "${target}.backup"
  fi
  sudo ln -sf "$VGPU_LIB/$shim" "$target"
  echo "  Linked (parent) $shim -> $VGPU_LIB/$shim"
done

echo "Done. Restart Ollama and check: journalctl -u ollama -n 80 --no-pager | grep -iE 'inference compute|total_vram|library='"
