#!/usr/bin/env python3
"""Transfer libvgpu_cublas.c (+ cuda_transport) to VM, build and install.

Deploy (Mar 2026): install **`/opt/vgpu/lib/libvgpu-cublas.so.12`**, then point
**`/usr/local/lib/ollama/cuda_v12/libcublas.so.12`** at the shim (NOT
`/opt/vgpu/lib/libcublas.so.12`, which can break discovery). Vendor math stays
as **`libcublas.so.12.3.2.9`**. See **`CUBLAS_VENDOR_SYMLINK_DEPLOY.md`**.
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
TRANSPORT = os.path.join(SCRIPT_DIR, "guest-shim", "cuda_transport.c")
PROTOCOL_HEADER = os.path.join(SCRIPT_DIR, "include", "cuda_protocol.h")
TRANSPORT_HEADER = os.path.join(SCRIPT_DIR, "guest-shim", "cuda_transport.h")
GPU_PROPS = os.path.join(SCRIPT_DIR, "guest-shim", "gpu_properties.h")
TEST_CUBLAS_VM = os.path.join(SCRIPT_DIR, "guest-shim", "test_cublas_vm.c")


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
            f"mkdir -p $(dirname {remote_dest_path}) && cp {remote_tmp_path} {remote_dest_path}"
        )
        if not ok:
            print(f"Copy to {remote_dest_path} failed:", err or out)
            return False
        return True

    ok, out, err = run_vm(f"mkdir -p {REMOTE_PHASE3}/guest-shim {REMOTE_PHASE3}/include")
    if not ok:
        print("mkdir phase3 failed:", err or out)
        return 1

    if not transfer_file(SOURCE, "/tmp/libvgpu_cublas_new.c", f"{REMOTE_PHASE3}/guest-shim/libvgpu_cublas.c"):
        return 1
    if not transfer_file(TRANSPORT, "/tmp/cuda_transport_new.c", f"{REMOTE_PHASE3}/guest-shim/cuda_transport.c"):
        return 1
    if not transfer_file(TRANSPORT_HEADER, "/tmp/cuda_transport_new.h", f"{REMOTE_PHASE3}/guest-shim/cuda_transport.h"):
        return 1
    if not transfer_file(GPU_PROPS, "/tmp/gpu_properties_new.h", f"{REMOTE_PHASE3}/guest-shim/gpu_properties.h"):
        return 1
    if not transfer_file(PROTOCOL_HEADER, "/tmp/cuda_protocol_new.h", f"{REMOTE_PHASE3}/include/cuda_protocol.h"):
        return 1
    if not transfer_file(TEST_CUBLAS_VM, "/tmp/test_cublas_vm_src.c", f"{REMOTE_PHASE3}/guest-shim/test_cublas_vm.c"):
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

    pw = VM_PASSWORD.replace("'", "'\"'\"'")
    install = (
        f"echo '{pw}' | sudo -S cp /tmp/libvgpu-cublas.so.12 /opt/vgpu/lib/libvgpu-cublas.so.12 && "
        f"echo '{pw}' | sudo -S chmod 755 /opt/vgpu/lib/libvgpu-cublas.so.12 && "
        # Do NOT add libcublas.so.12 under /opt/vgpu/lib (breaks GPU discovery).
        f"echo '{pw}' | sudo -S rm -f /opt/vgpu/lib/libcublas.so.12 && "
        # Ollama/GGML resolves libcublas from cuda_v12 — point at RPC shim.
        f"echo '{pw}' | sudo -S ln -sf /opt/vgpu/lib/libvgpu-cublas.so.12 "
        f"/usr/local/lib/ollama/cuda_v12/libcublas.so.12 && "
        f"echo '{pw}' | sudo -S ldconfig && "
        f"test -f /usr/local/lib/ollama/cuda_v12/libcublas.so.12.3.2.9 && echo VENDOR_OK=1 || echo VENDOR_OK=0; "
        f"echo '{pw}' | sudo -S systemctl restart ollama && echo OLLAMA_RESTARTED=1"
    )
    ok, out, err = run_vm(install, timeout_sec=120)
    print(out)
    if not ok:
        print("Install/restart failed:", err or out)
        return 1
    if "VENDOR_OK=1" not in (out or ""):
        print("WARNING: vendor libcublas.so.12.3.2.9 missing — init_real_cublas fallback may fail.")
    print("Installed libvgpu-cublas; cuda_v12/libcublas.so.12 -> shim; removed /opt/vgpu/lib/libcublas.so.12; restarted ollama.")

    verify_cmd = (
        "echo '--- readlink libcublas.so.12 ---'; readlink -f /usr/local/lib/ollama/cuda_v12/libcublas.so.12; "
        "echo '--- ldd libggml-cuda (cublas line) ---'; "
        "LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama "
        "ldd /usr/local/lib/ollama/cuda_v12/libggml-cuda.so 2>/dev/null | grep -E cublas || true; "
        "echo '--- test_cublas_vm ---'; "
        f"gcc -O2 -o /tmp/test_cublas_vm {REMOTE_PHASE3}/guest-shim/test_cublas_vm.c -ldl 2>&1 && "
        "LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama "
        "/tmp/test_cublas_vm 2>&1 | grep -E '^===|^  |cublasCreate|SGEMM|FAIL' | head -25"
    )
    ok, out, err = run_vm(verify_cmd, timeout_sec=120)
    print("--- post-deploy verify ---")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
