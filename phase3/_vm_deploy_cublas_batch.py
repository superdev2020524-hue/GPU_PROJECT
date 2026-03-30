#!/usr/bin/env python3
"""
Deploy batched GEMM shim sources to the VM, build, install, run smoke test.

Uses the same credentials as connect_vm.py: vm_config.VM_USER / VM_HOST / VM_PASSWORD.
(Local PC sudo password is not used here — that is only for your workstation.)
"""
import os
import sys
import shlex
import pexpect

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_HOST, VM_USER, VM_PASSWORD  # noqa: E402

REMOTE = f"/home/{VM_USER}/vgpu-cublas-build"
LOCAL_BASE = SCRIPT_DIR

FILES = [
    ("include/cuda_protocol.h", f"{REMOTE}/include/cuda_protocol.h"),
    ("guest-shim/libvgpu_cublas.c", f"{REMOTE}/libvgpu_cublas.c"),
    ("guest-shim/cuda_transport.c", f"{REMOTE}/cuda_transport.c"),
    ("guest-shim/cuda_transport.h", f"{REMOTE}/cuda_transport.h"),
    ("guest-shim/gpu_properties.h", f"{REMOTE}/gpu_properties.h"),
    ("guest-shim/test_gemm_batched_ex_vm.c", f"{REMOTE}/test_gemm_batched_ex_vm.c"),
]


def run_scp(local_path: str, remote_path: str) -> None:
    lp = os.path.join(LOCAL_BASE, local_path)
    if not os.path.isfile(lp):
        print(f"MISSING: {lp}", file=sys.stderr)
        sys.exit(1)
    cmd = f"scp -o StrictHostKeyChecking=no -o ConnectTimeout=20 {lp} {VM_USER}@{VM_HOST}:{remote_path}"
    print(cmd)
    child = pexpect.spawn(cmd, encoding="utf-8", timeout=120)
    child.logfile = sys.stdout
    i = child.expect(["password:", "Password:", pexpect.EOF], timeout=30)
    if i in (0, 1):
        child.sendline(VM_PASSWORD)
        child.expect(pexpect.EOF, timeout=120)
    child.close()
    if child.exitstatus not in (0, None):
        if child.exitstatus != 0 and child.exitstatus is not None:
            sys.exit(child.exitstatus)


def run_ssh(cmd: str, timeout: int = 300, force_tty: bool = False) -> None:
    # sudo on many systems requires a TTY; use -tt so password prompts work over pexpect.
    tty = "-tt " if force_tty or "sudo" in cmd else ""
    full = f"ssh {tty}-o StrictHostKeyChecking=no -o ConnectTimeout=20 {VM_USER}@{VM_HOST} {shlex.quote(cmd)}"
    print(full)
    child = pexpect.spawn(full, encoding="utf-8", timeout=timeout)
    child.logfile = sys.stdout
    while True:
        try:
            i = child.expect(
                ["password:", "Password:", r"\[sudo\] password", pexpect.EOF],
                timeout=timeout,
            )
            if i in (0, 1, 2):
                child.sendline(VM_PASSWORD)
            else:
                break
        except pexpect.TIMEOUT:
            print("TIMEOUT", file=sys.stderr)
            break
    child.close()


def main() -> None:
    run_ssh(f"mkdir -p {REMOTE}/include", timeout=60)
    for rel, dest in FILES:
        run_scp(rel, dest)

    build = (
        f"cd {REMOTE} && "
        "gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -Iinclude -I. "
        "-o libvgpu-cublas.so.12 libvgpu_cublas.c cuda_transport.c -ldl -lpthread && "
        "gcc -O2 -Wall -Wextra -o /tmp/test_gemm_batched_ex_vm test_gemm_batched_ex_vm.c -ldl"
    )
    run_ssh(build, timeout=120)

    install = (
        "sudo cp /opt/vgpu/lib/libvgpu-cublas.so.12 "
        f"/opt/vgpu/lib/libvgpu-cublas.so.12.bak.$(date +%Y%m%d_%H%M%S) && "
        f"sudo cp {REMOTE}/libvgpu-cublas.so.12 /opt/vgpu/lib/libvgpu-cublas.so.12 && "
        "sudo chmod 755 /opt/vgpu/lib/libvgpu-cublas.so.12"
    )
    run_ssh(install, timeout=60, force_tty=True)

    test_cmd = (
        "LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama "
        "/tmp/test_gemm_batched_ex_vm; echo exit=$?"
    )
    run_ssh(test_cmd, timeout=120)
    print("Done.")


if __name__ == "__main__":
    main()
