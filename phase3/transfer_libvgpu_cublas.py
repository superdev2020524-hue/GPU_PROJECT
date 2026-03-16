#!/usr/bin/env python3
"""Transfer only libvgpu_cublas.c to VM (checksum-verified), build and install.

The vGPU path relies on activating the shim as `/opt/vgpu/lib/libcublas.so.12`
so Ollama resolves CUBLAS through `LD_LIBRARY_PATH=/opt/vgpu/lib:...`.
"""
import os
import sys
import hashlib
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD, REMOTE_PHASE3

CHUNK_SIZE = 40000
SOURCE = os.path.join(SCRIPT_DIR, "guest-shim", "libvgpu_cublas.c")
PROTOCOL_HEADER = os.path.join(SCRIPT_DIR, "include", "cuda_protocol.h")


def run_vm(cmd, timeout_sec=900):
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True, text=True, timeout=timeout_sec,
    )
    return result.returncode == 0, result.stdout or "", result.stderr or ""


def main():
    def escape(s):
        return s.replace("'", "'\"'\"'")

    def transfer_file(local_path, remote_tmp_path, remote_dest_path):
        if not os.path.isfile(local_path):
            print(f"Source not found: {local_path}")
            return False
        with open(local_path, "rb") as f:
            data = f.read()
        local_sha = hashlib.sha256(data).hexdigest()
        print(f"Local SHA256 ({os.path.basename(local_path)}): {local_sha}")
        import base64
        b64 = base64.b64encode(data).decode("ascii")

        ok, out, err = run_vm("rm -f /tmp/combined.b64")
        if not ok:
            print("Failed to clear remote file:", err or out)
            return False
        n = 0
        for i in range(0, len(b64), CHUNK_SIZE):
            chunk = b64[i : i + CHUNK_SIZE]
            n += 1
            cmd = "echo -n '" + escape(chunk) + "' >> /tmp/combined.b64"
            ok, out, err = run_vm(cmd)
            if not ok:
                print(f"Chunk {n} failed:", err or out)
                return False
            print(f"{os.path.basename(local_path)} chunk {n} sent ({len(chunk)} chars)")

        ok, out, err = run_vm(f"base64 -d /tmp/combined.b64 > {remote_tmp_path} && wc -c {remote_tmp_path}")
        print(out)
        if not ok:
            print("Decode failed:", err or out)
            return False

        ok, out, err = run_vm(f"sha256sum {remote_tmp_path} | awk '{{print \"REMOTE_SHA256=\" $1}}'")
        if not ok:
            print("Remote SHA256 failed:", err or out)
            return False
        remote_sha = None
        for line in (out or "").splitlines():
            line = line.strip()
            if line.startswith("REMOTE_SHA256="):
                remote_sha = line.split("=", 1)[1].strip()
        if not remote_sha:
            print("Could not find REMOTE_SHA256")
            return False
        print(f"Remote SHA256 ({os.path.basename(local_path)}): {remote_sha}")
        if remote_sha != local_sha:
            print("ERROR: SHA256 mismatch. Aborting.")
            return False
        print("SHA256 verified.")

        ok, out, err = run_vm(
            f"cp {remote_tmp_path} {remote_dest_path} 2>/dev/null || "
            f"cp {remote_tmp_path} ~/phase3/{remote_dest_path.split('/phase3/', 1)[1]}"
        )
        if not ok:
            print(f"Copy to {remote_dest_path} failed:", err or out)
            return False
        return True

    if not transfer_file(SOURCE, "/tmp/libvgpu_cublas_new.c", f"{REMOTE_PHASE3}/guest-shim/libvgpu_cublas.c"):
        return 1
    if not transfer_file(PROTOCOL_HEADER, "/tmp/cuda_protocol_new.h", f"{REMOTE_PHASE3}/include/cuda_protocol.h"):
        return 1

    build_cmd = (
        f"cd {REMOTE_PHASE3} && "
        "gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE -Iinclude -Iguest-shim "
        "-o /tmp/libvgpu-cublas.so.12 guest-shim/libvgpu_cublas.c guest-shim/cuda_transport.c -ldl -lpthread 2>&1; "
        "echo BUILD_EXIT=$?; ls -la /tmp/libvgpu-cublas.so.12 2>&1"
    )
    ok, out, err = run_vm(build_cmd, timeout_sec=900)
    print(out)
    if not ok:
        print("Build failed:", err or out)
        return 1
    if "BUILD_EXIT=0" not in (out or ""):
        print("Build failed. Output:", out)
        return 1

    ok, out, err = run_vm(
        f"echo {VM_PASSWORD} | sudo -S cp /tmp/libvgpu-cublas.so.12 /opt/vgpu/lib/libvgpu-cublas.so.12 && "
        f"echo {VM_PASSWORD} | sudo -S ln -sf /opt/vgpu/lib/libvgpu-cublas.so.12 /opt/vgpu/lib/libcublas.so.12 && "
        f"echo {VM_PASSWORD} | sudo -S ldconfig && "
        f"echo {VM_PASSWORD} | sudo -S systemctl restart ollama"
    )
    print(out)
    if not ok:
        print("Install/restart failed:", err or out)
        return 1
    print("Transfer, build, install, activated libcublas.so.12 shim symlink, and restarted ollama.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
