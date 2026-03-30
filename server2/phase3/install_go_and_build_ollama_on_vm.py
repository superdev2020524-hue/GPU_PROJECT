#!/usr/bin/env python3
"""
Install Go 1.26.1 on the VM and build the patched Ollama binary there.

The VM already has the patched Ollama source at REMOTE_OLLAMA (device.go, server.go,
discover/runner.go were transferred by transfer_ollama_go_patches.py). This script:

1. Downloads Go 1.26.1 linux-amd64 tarball on the VM (from go.dev/dl), or
   if DOWNLOAD_ON_VM=0, uses a locally downloaded tarball transferred to the VM.
2. Extracts it to /usr/local/go (requires sudo).
3. Builds ollama.bin in the VM's Ollama tree with /usr/local/go/bin/go.
4. Installs ollama.bin and restarts the ollama service.

Usage:
  python3 install_go_and_build_ollama_on_vm.py

  # Use a tarball already on the VM (e.g. you copied it there):
  python3 install_go_and_build_ollama_on_vm.py --tarball /path/on/vm/go1.26.1.linux-amd64.tar.gz

  # Transfer local tarball to VM then install (if VM cannot reach go.dev):
  python3 install_go_and_build_ollama_on_vm.py --transfer-tarball /local/path/go1.26.1.linux-amd64.tar.gz

Requires: connect_vm.py, vm_config.py (VM_USER, VM_HOST, VM_PASSWORD, REMOTE_HOME).
"""
import os
import sys
import subprocess
import hashlib
import base64
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from vm_config import VM_USER, VM_HOST, VM_PASSWORD, REMOTE_HOME

REMOTE_OLLAMA = os.path.join(REMOTE_HOME, "ollama")


def sudo_cmd(cmd_after_sudo: str) -> str:
    """Run a command with sudo -S (password from stdin). Escapes password for shell."""
    passwd = VM_PASSWORD.replace("'", "'\"'\"'")  # escape single quotes for shell
    return f"echo '{passwd}' | sudo -S {cmd_after_sudo}"
GO_VERSION = "1.26.1"
GO_TARBALL_NAME = f"go{GO_VERSION}.linux-amd64.tar.gz"
GO_DOWNLOAD_URL = f"https://go.dev/dl/{GO_TARBALL_NAME}"
# SHA256 for go1.26.1.linux-amd64.tar.gz (from https://go.dev/dl/)
GO_TARBALL_SHA256 = "031f088e5d955bab8657ede27ad4e3bc5b7c1ba281f05f245bcc304f327c987a"
CHUNK_SIZE = 40000


def run_vm(cmd, timeout_sec=300):
    result = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        cwd=SCRIPT_DIR,
    )
    return result.returncode == 0, result.stdout or "", result.stderr or ""


def transfer_file_to_vm(local_path: str, remote_path: str, label: str) -> bool:
    """Transfer a local file to the VM in base64 chunks (same pattern as transfer_ollama_go_patches)."""
    with open(local_path, "rb") as f:
        data = f.read()
    local_sha = hashlib.sha256(data).hexdigest()
    b64 = base64.b64encode(data).decode("ascii")

    def escape(s):
        return s.replace("'", "'\"'\"'")

    ok, _, _ = run_vm("rm -f /tmp/combined.b64")
    for i in range(0, len(b64), CHUNK_SIZE):
        chunk = b64[i : i + CHUNK_SIZE]
        cmd = "echo -n '" + escape(chunk) + "' >> /tmp/combined.b64"
        ok, out, err = run_vm(cmd)
        if not ok:
            print(f"{label}: chunk write failed:", err or out)
            return False
    ok, out, err = run_vm(f"base64 -d /tmp/combined.b64 > {remote_path} && wc -c {remote_path}")
    if not ok:
        print(f"{label}: decode failed:", err or out)
        return False
    print(f"{label}: transferred")
    return True


def main():
    parser = argparse.ArgumentParser(description="Install Go on VM and build Ollama")
    parser.add_argument(
        "--tarball",
        metavar="PATH_ON_VM",
        help="Use this tarball path on the VM (e.g. /tmp/go1.26.1.linux-amd64.tar.gz) instead of downloading",
    )
    parser.add_argument(
        "--transfer-tarball",
        metavar="LOCAL_PATH",
        help="Transfer this local tarball to the VM, then install (use if VM cannot reach go.dev)",
    )
    args = parser.parse_args()

    tarball_on_vm = args.tarball
    if args.transfer_tarball:
        local_path = os.path.abspath(args.transfer_tarball)
        if not os.path.isfile(local_path):
            print(f"Error: not a file: {local_path}")
            return 1
        tarball_on_vm = f"/tmp/{GO_TARBALL_NAME}"
        print("Transferring Go tarball to VM...")
        if not transfer_file_to_vm(local_path, tarball_on_vm, "go tarball"):
            return 1

    # Step 1: Get tarball on VM (download or use provided path)
    if not tarball_on_vm:
        print("Downloading Go on the VM...")
        ok, out, err = run_vm(
            f"curl -sL -o /tmp/{GO_TARBALL_NAME} {GO_DOWNLOAD_URL} && "
            f"wc -c /tmp/{GO_TARBALL_NAME}",
            timeout_sec=120,
        )
        print(out)
        if not ok:
            print("Download failed. Use --transfer-tarball to copy a local tarball, or --tarball if it is already on the VM.")
            return 1
        tarball_on_vm = f"/tmp/{GO_TARBALL_NAME}"

    # Step 2: Extract to /usr/local (sudo)
    print("Extracting Go to /usr/local...")
    ok, out, err = run_vm(
        f"{sudo_cmd('rm -rf /usr/local/go')} && "
        f"{sudo_cmd(f'tar -C /usr/local -xzf {tarball_on_vm}')} && "
        f"/usr/local/go/bin/go version",
        timeout_sec=60,
    )
    print(out or err)
    if not ok or "/usr/local/go/bin/go version" not in (out or ""):
        print("Extract or go version check failed.")
        return 1

    # Step 3: Build ollama.bin on VM
    print("Building ollama.bin on the VM...")
    ok, out, err = run_vm(
        f"cd {REMOTE_OLLAMA} && /usr/local/go/bin/go build -o ollama.bin . 2>&1; echo BUILD_EXIT=$?",
        timeout_sec=300,
    )
    print(out or err)
    if not ok or "BUILD_EXIT=0" not in (out or ""):
        print("Build did not succeed. Output above.")
        return 1

    # Step 4: Install and restart ollama
    print("Installing ollama.bin and restarting ollama service...")
    ok, out, err = run_vm(
        f"{sudo_cmd('systemctl stop ollama')} && "
        f"{sudo_cmd(f'cp {REMOTE_OLLAMA}/ollama.bin /usr/local/bin/ollama.bin')} && "
        f"{sudo_cmd('systemctl start ollama')} && "
        "echo DONE",
    )
    print(out or err)
    if not ok or "DONE" not in (out or ""):
        print("Install or restart failed.")
        return 1

    print("Go installed, ollama.bin built and installed. Ollama service restarted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
