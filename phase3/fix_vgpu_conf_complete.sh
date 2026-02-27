#!/bin/bash
# Complete fix for vgpu.conf:
# 1. Remove libvgpu-syscall.so from LD_PRELOAD (file doesn't exist)
# 2. Add OLLAMA_LIBRARY_PATH correctly

VGPU_CONF="/etc/systemd/system/ollama.service.d/vgpu.conf"

# Remove libvgpu-syscall.so from LD_PRELOAD
sed -i 's|:/usr/lib64/libvgpu-syscall.so||g' "$VGPU_CONF"
sed -i 's|libvgpu-syscall.so:||g' "$VGPU_CONF"

# Fix LD_PRELOAD order (should be: exec, cuda, nvml, cudart)
sed -i 's|LD_PRELOAD=.*|LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"|g' "$VGPU_CONF"

# Add OLLAMA_LIBRARY_PATH if not present
if ! grep -q "OLLAMA_LIBRARY_PATH" "$VGPU_CONF"; then
    echo "" >> "$VGPU_CONF"
    echo "# OLLAMA_LIBRARY_PATH - tells scanner where to find backend libraries" >> "$VGPU_CONF"
    echo "Environment=\"OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12\"" >> "$VGPU_CONF"
fi

echo "✓ vgpu.conf fixed"
