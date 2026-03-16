#!/bin/bash
# Complete guest setup for Phase 3 vGPU (Ollama + shims).
# Run ON THE VM (e.g. after copying phase3 tree and logging in as test_4).
# Prereq: VGPU-STUB already attached; gcc, make installed; Ollama installed.
# Usage: cd /home/test_4/phase3 && bash setup_vm_complete.sh
#   Or from host: python3 connect_vm.py "cd /home/test_4/phase3 && bash setup_vm_complete.sh"

set -e
echo "=== Phase 3 vGPU guest setup ==="

# 1) Build guest shims
if [ ! -f guest-shim/Makefile ] && [ -f Makefile ]; then
    echo "1. Building guest shims (make guest)..."
    make guest
else
    echo "1. Building guest shims..."
    (cd guest-shim && make) 2>/dev/null || make guest
fi
echo "   Build done."

# 2) Install to /opt/vgpu/lib
echo "2. Installing to /opt/vgpu/lib..."
sudo mkdir -p /opt/vgpu/lib
PHASE3="${PHASE3:-.}"
[ -d "$PHASE3/guest-shim" ] && SHIM_DIR="$PHASE3/guest-shim" || SHIM_DIR="guest-shim"
sudo cp "$SHIM_DIR/libvgpu-cuda.so.1" /opt/vgpu/lib/
sudo cp "$SHIM_DIR/libvgpu-cudart.so" /opt/vgpu/lib/ 2>/dev/null || true
sudo cp "$SHIM_DIR/libvgpu-nvml.so" /opt/vgpu/lib/ 2>/dev/null || true
sudo ln -sf /opt/vgpu/lib/libvgpu-cuda.so.1 /opt/vgpu/lib/libcuda.so.1
sudo ln -sf /opt/vgpu/lib/libvgpu-cudart.so /opt/vgpu/lib/libcudart.so.12 2>/dev/null || true
sudo ln -sf /opt/vgpu/lib/libvgpu-nvml.so /opt/vgpu/lib/libnvidia-ml.so.1 2>/dev/null || true
if [ -f "$SHIM_DIR/libvgpu-cublas.so.12" ]; then
    sudo cp "$SHIM_DIR/libvgpu-cublas.so.12" /opt/vgpu/lib/
    sudo ln -sf /opt/vgpu/lib/libvgpu-cublas.so.12 /opt/vgpu/lib/libcublas.so.12
fi
echo "   Install done."

# 3) Ollama systemd drop-in
echo "3. Configuring Ollama service..."
sudo mkdir -p /etc/systemd/system/ollama.service.d
if [ -f "ollama.service.d_vgpu.conf" ]; then
    sudo cp ollama.service.d_vgpu.conf /etc/systemd/system/ollama.service.d/vgpu.conf
else
    sudo tee /etc/systemd/system/ollama.service.d/vgpu.conf << 'EOF'
[Service]
ExecStart=
ExecStart=/usr/local/bin/ollama.bin serve
Environment=LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama
Environment=OLLAMA_NUM_GPU=1
Environment=OLLAMA_LLM_LIBRARY=cuda_v12
Environment=OLLAMA_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama
Environment=OLLAMA_LOAD_TIMEOUT=20m
EOF
fi
sudo systemctl daemon-reload
echo "   Done."

# 4) udev + vgpu-devices for BAR access (optional if already done)
if [ ! -f /etc/udev/rules.d/99-vgpu-nvidia.rules ]; then
    echo "4. Installing udev rule and vgpu-devices.service..."
    echo 'SUBSYSTEM=="pci", ATTR{vendor}=="0x10de", ATTR{device}=="0x2331", RUN+="/bin/chmod 0666 /sys%p/resource0 /sys%p/resource1"' | sudo tee /etc/udev/rules.d/99-vgpu-nvidia.rules
    sudo udevadm control --reload-rules
    sudo udevadm trigger
fi
echo "5. Restarting Ollama..."
sudo systemctl restart ollama
echo "=== Setup complete. Check: systemctl status ollama; journalctl -u ollama -n 20 ==="
