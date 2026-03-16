#!/usr/bin/env python3
"""Transfer cuda_transport.c and cuda_transport.h to VM, rebuild libvgpu-cuda."""
import base64
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_PASSWORD, REMOTE_PHASE3

def run_vm(cmd, timeout=300):
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True, text=True, timeout=timeout, cwd=SCRIPT_DIR,
    )
    return r.returncode == 0, r.stdout or "", r.stderr or ""

def esc(s):
    return s.replace("'", "'\"'\"'")

def main():
    for name in ["cuda_transport.c", "cuda_transport.h"]:
        path = os.path.join(SCRIPT_DIR, "guest-shim", name)
        data = open(path, "rb").read()
        b64 = base64.b64encode(data).decode("ascii")
        run_vm("rm -f /tmp/ct.b64")
        for i in range(0, len(b64), 40000):
            chunk = b64[i : i + 40000]
            ok, _, _ = run_vm("echo -n '" + esc(chunk) + "' >> /tmp/ct.b64")
            if not ok:
                print(f"Chunk failed for {name}")
                return 1
        ok, out, _ = run_vm(f"base64 -d /tmp/ct.b64 > {REMOTE_PHASE3}/guest-shim/{name}")
        if not ok:
            print(f"Decode failed for {name}")
            return 1
        print(f"Transferred {name}")

    build = (
        f"cd {REMOTE_PHASE3} && "
        "gcc -shared -fPIC -O2 -std=c11 -D_GNU_SOURCE -Iinclude -Iguest-shim "
        "-o /tmp/libvgpu-cuda.so.1 guest-shim/libvgpu_cuda.c guest-shim/cuda_transport.c -ldl -lpthread 2>&1; "
        "echo BUILD_EXIT=$?"
    )
    ok, out, _ = run_vm(build, timeout=120)
    print(out)
    if "BUILD_EXIT=0" not in out:
        return 1

    ok, _, _ = run_vm(
        f"echo {VM_PASSWORD} | sudo -S cp /tmp/libvgpu-cuda.so.1 /opt/vgpu/lib/libvgpu-cuda.so.1 && "
        f"echo {VM_PASSWORD} | sudo -S systemctl restart ollama"
    )
    print("Installed and restarted.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
