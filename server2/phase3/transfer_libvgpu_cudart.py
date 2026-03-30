#!/usr/bin/env python3
"""Transfer only libvgpu_cudart.c to VM, build, install, restart ollama."""
import base64
import hashlib
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_PASSWORD, REMOTE_PHASE3

CHUNK_SIZE = 40000
SOURCE = os.path.join(SCRIPT_DIR, "guest-shim", "libvgpu_cudart.c")


def run_vm(cmd: str, timeout_sec: int = 300):
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return result.returncode == 0, (result.stdout or ""), (result.stderr or "")


def esc_single(s: str) -> str:
    return s.replace("'", "'\"'\"'")


def main() -> int:
    if not os.path.isfile(SOURCE):
        print(f"Source not found: {SOURCE}")
        return 1

    data = open(SOURCE, "rb").read()
    local_sha = hashlib.sha256(data).hexdigest()
    b64 = base64.b64encode(data).decode("ascii")
    print(f"Local SHA256: {local_sha}")

    ok, out, err = run_vm("rm -f /tmp/combined.b64 /tmp/libvgpu_cudart_new.c")
    if not ok:
        print(err or out)
        return 1

    sent = 0
    for i in range(0, len(b64), CHUNK_SIZE):
        chunk = b64[i : i + CHUNK_SIZE]
        ok, out, err = run_vm(f"echo -n '{esc_single(chunk)}' >> /tmp/combined.b64")
        if not ok:
            print(f"Chunk failed at {i}:")
            print(err or out)
            return 1
        sent += 1
    print(f"Chunks sent: {sent}")

    ok, out, err = run_vm(
        "base64 -d /tmp/combined.b64 > /tmp/libvgpu_cudart_new.c && wc -c /tmp/libvgpu_cudart_new.c"
    )
    print(out)
    if not ok:
        print(err or out)
        return 1

    ok, out, err = run_vm("sha256sum /tmp/libvgpu_cudart_new.c | awk '{print $1}'")
    if not ok:
        print(err or out)
        return 1
    remote_sha = ""
    for line in out.splitlines():
        line = line.strip()
        if len(line) == 64 and all(c in "0123456789abcdef" for c in line):
            remote_sha = line
            break
    print(f"Remote SHA256: {remote_sha}")
    if remote_sha != local_sha:
        print("SHA mismatch; aborting.")
        return 1

    ok, out, err = run_vm(
        f"cp /tmp/libvgpu_cudart_new.c {REMOTE_PHASE3}/guest-shim/libvgpu_cudart.c 2>/dev/null || "
        "cp /tmp/libvgpu_cudart_new.c ~/phase3/guest-shim/libvgpu_cudart.c"
    )
    if not ok:
        print(err or out)
        return 1

    build_cmd = (
        f"cd {REMOTE_PHASE3} && "
        "gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -Iinclude -Iguest-shim "
        "-o /tmp/libvgpu-cudart.so guest-shim/libvgpu_cudart.c -ldl 2>/tmp/cudart_err.txt; "
        "echo BUILD_EXIT=$?; ls -la /tmp/libvgpu-cudart.so 2>/dev/null"
    )
    ok, out, err = run_vm(build_cmd, timeout_sec=600)
    print(out)
    if (not ok) or ("BUILD_EXIT=0" not in out):
        _, o, _ = run_vm("tail -n 200 /tmp/cudart_err.txt 2>/dev/null || true")
        print(o)
        return 1

    ok, out, err = run_vm(
        f"echo {VM_PASSWORD} | sudo -S cp /tmp/libvgpu-cudart.so /opt/vgpu/lib/libvgpu-cudart.so && "
        f"echo {VM_PASSWORD} | sudo -S systemctl restart ollama"
    )
    print(out)
    if not ok:
        print(err or out)
        return 1

    print("Transfer/build/install complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
