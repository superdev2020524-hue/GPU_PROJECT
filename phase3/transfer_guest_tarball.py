#!/usr/bin/env python3
"""Transfer guest-shim + include + Makefile to VM as tarball (base64 chunks)."""
import base64
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD, REMOTE_PHASE3

def run_vm(cmd, timeout_sec=120):
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True, text=True, timeout=timeout_sec, cwd=SCRIPT_DIR,
    )
    return r.returncode == 0, r.stdout or "", r.stderr or ""

def main():
    tarball = os.path.join(SCRIPT_DIR, "phase3_guest.tar")
    if not os.path.isfile(tarball):
        print("Creating tarball...")
        import tarfile
        with tarfile.open(tarball, "w") as tar:
            tar.add(os.path.join(SCRIPT_DIR, "guest-shim"), "guest-shim")
            tar.add(os.path.join(SCRIPT_DIR, "include"), "include")
            tar.add(os.path.join(SCRIPT_DIR, "Makefile"), "Makefile")
    with open(tarball, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("ascii")
    chunk_size = 50000
    chunks = [b64[i : i + chunk_size] for i in range(0, len(b64), chunk_size)]

    def escape(s):
        return s.replace("'", "'\"'\"'")

    print(f"Transferring {len(chunks)} chunks to {VM_USER}@{VM_HOST}...")
    ok, _, _ = run_vm(f"mkdir -p {REMOTE_PHASE3} && rm -f {REMOTE_PHASE3}/combined.b64", timeout_sec=30)
    if not ok:
        print("Failed to prepare remote dir")
        return 1
    for i, chunk in enumerate(chunks):
        cmd = "echo -n '" + escape(chunk) + "' >> " + REMOTE_PHASE3 + "/combined.b64"
        ok, _, _ = run_vm(cmd, timeout_sec=60)
        if not ok:
            print(f"Chunk {i+1} failed")
            return 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(chunks)}")
    print("Decoding and extracting...")
    ok, out, err = run_vm(
        f"cd {REMOTE_PHASE3} && base64 -d combined.b64 > phase3_guest.tar && tar xf phase3_guest.tar && rm combined.b64 phase3_guest.tar && ls guest-shim include Makefile",
        timeout_sec=60,
    )
    print(out or err)
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())
