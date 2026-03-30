#!/usr/bin/env python3
"""
Install dom0-matched libcublas / libcublasLt into the VM Ollama cuda_v12 tree.

STEP 1 — On dom0 (GPU host) as root, copy libs to the VM (one command):

  mkdir -p /tmp/cublas_from_dom0
  scp -o StrictHostKeyChecking=no \\
    /usr/local/cuda/lib64/libcublas.so.12 \\
    /usr/local/cuda/lib64/libcublas.so.12.3.2.9 \\
    /usr/local/cuda/lib64/libcublasLt.so.12 \\
    /usr/local/cuda/lib64/libcublasLt.so.12.3.2.9 \\
    test-4@10.25.33.12:/tmp/cublas_from_dom0/

STEP 2 — From your repo:

  cd phase3 && python3 install_cublas_align_from_dom0_on_vm.py

Backup on VM already done: /root/cublas_backup_pre_align/
"""
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

INSTALL_CMD = r"""
set -e
REMOTE_DIR=/tmp/cublas_from_dom0
OLLAMA_CUDA=/usr/local/lib/ollama/cuda_v12
if [ ! -f "$REMOTE_DIR/libcublas.so.12.3.2.9" ] || [ ! -f "$REMOTE_DIR/libcublasLt.so.12.3.2.9" ]; then
  echo "MISSING: run STEP 1 on dom0 first. Contents:"
  ls -la "$REMOTE_DIR/" 2>/dev/null || true
  exit 1
fi
sudo cp -a "$REMOTE_DIR"/libcublas.so.12* "$REMOTE_DIR"/libcublasLt.so.12* "$OLLAMA_CUDA"/
ls -la "$OLLAMA_CUDA"/libcublas.so.12* "$OLLAMA_CUDA"/libcublasLt.so.12* | head -12
sudo systemctl restart ollama
sleep 2
systemctl is-active ollama
echo INSTALL_OK
"""


def main():
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), INSTALL_CMD.strip()],
        cwd=SCRIPT_DIR,
    )
    return r.returncode


if __name__ == "__main__":
    sys.exit(main())
