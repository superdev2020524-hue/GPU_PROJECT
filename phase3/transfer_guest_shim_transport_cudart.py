#!/usr/bin/env python3
"""Transfer only cuda_transport.c and libvgpu_cudart.c to VM, build both shims, install, restart ollama.

Uses the same pattern as transfer_libvgpu_cuda.py: connect_vm.py for commands,
chunked base64 transfer with SHA256 verify. Only these two files are copied;
libvgpu_cuda.c is unchanged and stays as on VM. Builds libvgpu-cuda.so.1 (uses
updated cuda_transport.c) and libvgpu-cudart.so (uses updated libvgpu_cudart.c),
installs to /opt/vgpu/lib, restarts ollama.
"""
import os
import sys
import base64
import hashlib
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD, REMOTE_PHASE3

CHUNK_SIZE = 40000
FILES = [
    ("guest-shim/cuda_transport.c", "/tmp/cuda_transport_new.c", "guest-shim/cuda_transport.c"),
    ("guest-shim/libvgpu_cudart.c", "/tmp/libvgpu_cudart_new.c", "guest-shim/libvgpu_cudart.c"),
]


def run_vm(cmd, timeout_sec=900):
    """Run command on VM via connect_vm.py."""
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        cwd=SCRIPT_DIR,
    )
    return result.returncode == 0, result.stdout or "", result.stderr or ""


def transfer_file(rel_path, remote_tmp_path, remote_dest_rel):
    """Transfer one file: chunked base64, decode on VM, SHA256 verify, cp to phase3 tree."""
    local_path = os.path.join(SCRIPT_DIR, rel_path)
    if not os.path.isfile(local_path):
        print(f"Source not found: {local_path}")
        return False
    with open(local_path, "rb") as f:
        data = f.read()
    local_sha = hashlib.sha256(data).hexdigest()
    print(f"  {rel_path}: local SHA256 {local_sha[:16]}...")
    b64 = base64.b64encode(data).decode("ascii")

    def escape(s):
        return s.replace("'", "'\"'\"'")

    ok, out, err = run_vm("rm -f /tmp/combined.b64")
    if not ok:
        print(f"  Failed to clear remote file: {err or out}")
        return False
    for i in range(0, len(b64), CHUNK_SIZE):
        chunk = b64[i : i + CHUNK_SIZE]
        cmd = "echo -n '" + escape(chunk) + "' >> /tmp/combined.b64"
        ok, out, err = run_vm(cmd)
        if not ok:
            print(f"  Chunk failed: {err or out}")
            return False
    ok, out, err = run_vm(
        f"base64 -d /tmp/combined.b64 > {remote_tmp_path} && wc -c {remote_tmp_path}"
    )
    if not ok:
        print(f"  Decode failed: {err or out}")
        return False
    ok, out, err = run_vm(
        f"sha256sum {remote_tmp_path} | awk '{{print \"REMOTE_SHA256=\" $1}}'"
    )
    if not ok:
        print(f"  Remote SHA256 failed: {err or out}")
        return False
    remote_sha = None
    for line in (out or "").splitlines():
        line = line.strip()
        if line.startswith("REMOTE_SHA256="):
            remote_sha = line.split("=", 1)[1].strip()
    if not remote_sha or remote_sha != local_sha:
        print(f"  ERROR: SHA256 mismatch for {rel_path}. Aborting.")
        return False
    dest = f"{REMOTE_PHASE3}/{remote_dest_rel}"
    ok, out, err = run_vm(
        f"cp {remote_tmp_path} {dest} 2>/dev/null || cp {remote_tmp_path} ~/phase3/{remote_dest_rel}"
    )
    if not ok:
        print(f"  Copy to phase3 failed: {err or out}")
        return False
    print(f"  {rel_path}: verified and copied to VM {remote_dest_rel}")
    return True


def main():
    print("Transferring cuda_transport.c and libvgpu_cudart.c to VM (chunked base64 + SHA256)...")
    for rel_path, remote_tmp, remote_dest in FILES:
        if not transfer_file(rel_path, remote_tmp, remote_dest):
            return 1

    print("Building libvgpu-cuda.so.1 and libvgpu-cudart.so on VM...")
    # Build both and install in one session; write result to file (connect_vm truncates long output)
    build_and_install = (
        f"cd {REMOTE_PHASE3} && "
        "rm -f /tmp/libvgpu-cuda.so.1 /tmp/libvgpu-cudart.so /tmp/transfer_result.txt && "
        "gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -Iinclude -Iguest-shim "
        "-o /tmp/libvgpu-cuda.so.1 guest-shim/libvgpu_cuda.c guest-shim/cuda_transport.c -ldl -lpthread 2>/tmp/cuda_err.txt && "
        "gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -Iinclude -Iguest-shim "
        "-o /tmp/libvgpu-cudart.so guest-shim/libvgpu_cudart.c -ldl 2>/tmp/cudart_err.txt && "
        f"echo {VM_PASSWORD} | sudo -S cp /tmp/libvgpu-cuda.so.1 /opt/vgpu/lib/libvgpu-cuda.so.1 && "
        f"echo {VM_PASSWORD} | sudo -S cp /tmp/libvgpu-cuda.so.1 /usr/local/lib/ollama/libcuda.so.1 && "
        f"echo {VM_PASSWORD} | sudo -S cp /tmp/libvgpu-cuda.so.1 /usr/local/lib/ollama/cuda_v12/libcuda.so.1 && "
        f"echo {VM_PASSWORD} | sudo -S cp /tmp/libvgpu-cuda.so.1 /usr/lib64/libvgpu-cuda.so && "
        f"echo {VM_PASSWORD} | sudo -S cp /tmp/libvgpu-cudart.so /opt/vgpu/lib/libvgpu-cudart.so && "
        f"echo {VM_PASSWORD} | sudo -S systemctl restart ollama && "
        "echo SUCCESS > /tmp/transfer_result.txt || echo FAIL > /tmp/transfer_result.txt"
    )
    ok, out, err = run_vm(build_and_install, timeout_sec=300)
    # Check result file (output may be truncated)
    check_ok, check_out, check_err = run_vm("cat /tmp/transfer_result.txt 2>/dev/null || echo MISSING")
    if "SUCCESS" not in (check_out or ""):
        print("Build or install failed. VM stderr (cuda):")
        _, o, _ = run_vm("cat /tmp/cuda_err.txt 2>/dev/null")
        print(o or "(empty)")
        print("VM stderr (cudart):")
        _, o2, _ = run_vm("cat /tmp/cudart_err.txt 2>/dev/null")
        print(o2 or "(empty)")
        return 1
    print("Done: cuda_transport.c and libvgpu_cudart.c deployed; both libs installed; ollama restarted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
