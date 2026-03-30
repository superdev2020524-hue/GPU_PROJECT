#!/usr/bin/env python3
"""Deploy HEXACORE vH100 CAP display name: transfer gpu_properties.h + libvgpu_cuda.c, build, install."""
import os
import sys
import base64

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD, REMOTE_PHASE3
from connect_vm import connect_and_run_command

def run(cmd, timeout=120):
    r = connect_and_run_command(cmd)
    return r

def main():
    # 1. Transfer gpu_properties.h (small)
    print("=== Transfer gpu_properties.h ===")
    with open(os.path.join(SCRIPT_DIR, "guest-shim", "gpu_properties.h"), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    run("rm -f /tmp/gpu_properties.b64")
    for i in range(0, len(b64), 32000):
        chunk = b64[i:i+32000].replace("'", "'\"'\"'")
        run(f"echo -n '{chunk}' >> /tmp/gpu_properties.b64", timeout=30)
    run("base64 -d /tmp/gpu_properties.b64 > /tmp/gpu_properties.h.new && rm /tmp/gpu_properties.b64")
    run(f"cp /tmp/gpu_properties.h.new {REMOTE_PHASE3}/guest-shim/gpu_properties.h || cp /tmp/gpu_properties.h.new ~/phase3/guest-shim/gpu_properties.h")
    run("rm /tmp/gpu_properties.h.new")
    print("  Done")

    # 2. Transfer libvgpu_cuda.c via transfer script
    print("\n=== Transfer libvgpu_cuda.c ===")
    import subprocess
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "transfer_libvgpu_cuda.py")],
        capture_output=True, text=True, timeout=600
    )
    print(r.stdout or "")
    if r.returncode != 0:
        print("STDERR:", r.stderr or "")
        return 1

    # 3. transfer script builds to /opt/vgpu/lib - check if we need /usr/lib64
    print("\n=== Verify symlinks / install location ===")
    out = run("ls -la /usr/lib64/libvgpu-cuda* /opt/vgpu/lib/libvgpu-cuda* 2>/dev/null; cat /etc/ld.so.preload 2>/dev/null")
    print(out or "")

    return 0

if __name__ == "__main__":
    sys.exit(main())
