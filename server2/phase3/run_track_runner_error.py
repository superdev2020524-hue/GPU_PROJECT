#!/usr/bin/env python3
"""
Run track_runner_error.sh on the VM via connect_vm to capture the runner's
actual stderr and exit code (strace on server + children).
"""
import base64
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

def main():
    sh_path = os.path.join(SCRIPT_DIR, "track_runner_error.sh")
    with open(sh_path, "rb") as f:
        raw = f.read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    b64 = base64.b64encode(raw).decode()
    # Run on VM: decode script and execute (long timeout for 7min wait inside script)
    cmd = f"echo '{b64}' | base64 -d | bash -e"
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        cwd=SCRIPT_DIR,
        timeout=540,  # 9 min (script has 7 min sleep + strace/curl)
        capture_output=True,
        text=True,
    )
    out = (r.stdout or "") + (r.stderr or "")
    print(out)
    return 0 if r.returncode == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
