#!/usr/bin/env python3
"""Apply ollama service override on VM: LD_LIBRARY_PATH with /opt/vgpu/lib first, NO LD_PRELOAD.
Run from phase3: python3 apply_vgpu_service_no_ldpreload.py
"""
import os
import sys
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD

CONF = """[Service]
ExecStart=
ExecStart=/usr/local/bin/ollama serve
Environment=LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama
Environment=OLLAMA_NUM_GPU=1
Environment=OLLAMA_LLM_LIBRARY=cuda_v12
Environment=OLLAMA_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama
Environment=OLLAMA_LOAD_TIMEOUT=20m
LimitMEMLOCK=infinity
CapabilityBoundingSet=CAP_SYS_ADMIN CAP_IPC_LOCK
AmbientCapabilities=CAP_SYS_ADMIN CAP_IPC_LOCK
NoNewPrivileges=no
ProtectKernelTunables=no
PrivateDevices=no
ReadWritePaths=/sys/bus/pci/devices/
"""

UDEV_RULES = """# VGPU shim persistent access for guest PCI BARs
SUBSYSTEM=="pci", ATTR{vendor}=="0x10de", ATTR{device}=="0x2331", RUN+="/bin/chmod 0666 /sys%p/resource0 /sys%p/resource1"
"""

BOOT_SERVICE = """[Unit]
Description=Grant vGPU BAR access before Ollama
After=systemd-udev-settle.service
Before=ollama.service

[Service]
Type=oneshot
ExecStart=/bin/bash -c 'for dev in /sys/bus/pci/devices/*/; do v=$(cat "$dev/vendor" 2>/dev/null); d=$(cat "$dev/device" 2>/dev/null); if [ "$v" = "0x10de" ] && [ "$d" = "0x2331" ]; then chmod 0666 "${dev}resource0" "${dev}resource1" 2>/dev/null || true; fi; done'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"""

def run_vm(cmd, timeout=60):
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True, text=True, timeout=timeout, cwd=SCRIPT_DIR,
    )
    return r.returncode == 0, r.stdout or "", r.stderr or ""

def main():
    # Write conf to /tmp on VM via base64
    import base64
    files = [
        (CONF, "/tmp/vgpu.conf", "/tmp/vgpu_conf_b64"),
        (UDEV_RULES, "/tmp/99-vgpu-nvidia.rules", "/tmp/vgpu_udev_b64"),
        (BOOT_SERVICE, "/tmp/vgpu-devices.service", "/tmp/vgpu_boot_b64"),
    ]
    chunk_size = 500
    for content, remote_path, remote_b64 in files:
        b64 = base64.b64encode(content.encode()).decode()
        ok, _, _ = run_vm(f"rm -f {remote_b64}")
        for i in range(0, len(b64), chunk_size):
            chunk = b64[i:i+chunk_size].replace("'", "'\"'\"'")
            ok, o, e = run_vm(f"echo -n '{chunk}' >> {remote_b64}")
            if not ok:
                print("Chunk write failed:", e or o)
                return 1
        ok, o, e = run_vm(f"base64 -d {remote_b64} > {remote_path}")
        if not ok:
            print("Decode failed:", e or o)
            return 1
    # Install with sudo
    pw = VM_PASSWORD
    ok, o, e = run_vm(
        f"echo {pw} | sudo -S mkdir -p /etc/systemd/system/ollama.service.d && "
        f"echo {pw} | sudo -S cp /tmp/vgpu.conf /etc/systemd/system/ollama.service.d/vgpu.conf && "
        f"echo {pw} | sudo -S cp /tmp/99-vgpu-nvidia.rules /etc/udev/rules.d/99-vgpu-nvidia.rules && "
        f"echo {pw} | sudo -S cp /tmp/vgpu-devices.service /etc/systemd/system/vgpu-devices.service && "
        f"echo {pw} | sudo -S systemctl daemon-reload && "
        f"echo {pw} | sudo -S systemctl enable vgpu-devices.service && "
        f"echo {pw} | sudo -S udevadm control --reload && "
        f"echo {pw} | sudo -S udevadm trigger && "
        f"echo {pw} | sudo -S systemctl start vgpu-devices.service && "
        f"echo {pw} | sudo -S systemctl restart ollama"
    )
    if not ok:
        print("Install/restart failed:", e or o)
        return 1
    print("Service override, udev rule, and boot service applied. Ollama restarted.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
