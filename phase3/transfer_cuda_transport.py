#!/usr/bin/env python3
"""Transfer cuda_transport.c and cuda_transport.h to VM via SCP, rebuild libvgpu-cuda.

Before transfer we stop Ollama on the VM to free memory and reduce load so the VM
stays responsive for SCP and the gcc build. See TRANSFER_LIBVGPU_CUDA_SCRIPT_INVESTIGATION.md
and PHASE3_TEST3_DEPLOY.md (use SCP; avoid chunked transfer for shim sources).

If SCP fails with "timeout waiting for password" or connection errors, the machine
running this script cannot reach the VM. Run this script FROM the mediator host
(where phase3 is at e.g. /root/phase3); see TRANSFER_FROM_HOST.md and CONNECT_VM_README.md.
"""
import os
import shlex
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_PASSWORD, VM_USER, VM_HOST, REMOTE_PHASE3

def run_vm(cmd, timeout=300):
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True, text=True, timeout=timeout, cwd=SCRIPT_DIR,
    )
    return r.returncode == 0, r.stdout or "", r.stderr or ""

def scp_file(local_path, remote_path, timeout=300):
    """Copy file to VM via scp with pexpect (password auth). Returns True on success."""
    import pexpect
    dest = f"{VM_USER}@{VM_HOST}:{remote_path}"
    scp_cmd = "scp -o StrictHostKeyChecking=no -o ConnectTimeout=15 " + shlex.quote(local_path) + " " + shlex.quote(dest)
    try:
        c = pexpect.spawn(scp_cmd, timeout=timeout, encoding="utf-8")
        idx = c.expect(["password:", "Password:", pexpect.EOF, pexpect.TIMEOUT], timeout=60)
        if idx == 2:
            # EOF before password - connection failed or completed without auth
            c.close()
            print(f"scp: got EOF before password (check connectivity). before={c.before!r}", file=sys.stderr)
            return False
        if idx == 3:
            c.close()
            print(f"scp: timeout waiting for password. before={c.before!r}", file=sys.stderr)
            return False
        if idx in (0, 1):
            c.sendline(VM_PASSWORD)
        idx2 = c.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=timeout)
        if idx2 == 1:
            print(f"scp: timeout during transfer. before={c.before!r}", file=sys.stderr)
        c.close()
        ok = idx2 == 0 and (c.exitstatus is None or c.exitstatus == 0)
        if not ok:
            print(f"scp: idx2={idx2} exitstatus={getattr(c, 'exitstatus', None)} before={c.before!r}", file=sys.stderr)
        return ok
    except Exception as e:
        print(f"scp error: {e}", file=sys.stderr)
        return False

def main():
    # Stop Ollama first so VM has free memory and is responsive for SCP/build (see
    # TRANSFER_LIBVGPU_CUDA_SCRIPT_INVESTIGATION.md).
    print("Stopping Ollama on VM to free memory...")
    ok, out, err = run_vm(f"echo {VM_PASSWORD} | sudo -S systemctl stop ollama", timeout=60)
    if not ok:
        print("Warning: could not stop ollama:", err or out, file=sys.stderr)
    else:
        print("Ollama stopped.")

    guest_shim = os.path.join(SCRIPT_DIR, "guest-shim")
    remote_guest_shim = f"{REMOTE_PHASE3}/guest-shim"
    for name in ["cuda_transport.c", "cuda_transport.h"]:
        local = os.path.join(guest_shim, name)
        remote = f"{remote_guest_shim}/{name}"
        if not scp_file(local, remote):
            print(f"SCP failed for {name}")
            return 1
        print(f"Transferred {name}")

    build = (
        f"cd {REMOTE_PHASE3} && "
        "gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -Iinclude -Iguest-shim "
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
