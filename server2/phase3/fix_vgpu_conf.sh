#!/bin/bash
# Fix vgpu.conf configuration

set -e

VGPU_CONF="/etc/systemd/system/ollama.service.d/vgpu.conf"

# Remove problematic libraries from LD_PRELOAD
sed -i "s|libvgpu-exec.so:||g" "$VGPU_CONF"
sed -i "s|libvgpu-syscall.so:||g" "$VGPU_CONF"

# Add OLLAMA_LIBRARY_PATH if missing
if ! grep -q "OLLAMA_LIBRARY_PATH" "$VGPU_CONF"; then
    echo 'Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"' >> "$VGPU_CONF"
fi

# Reload systemd
systemctl daemon-reload

# Restart Ollama
systemctl restart ollama

echo "Configuration fixed and Ollama restarted"
