#!/usr/bin/env python3
"""Run short (2 min) track_runner_error_short.sh on VM to find where runner blocks."""
import base64
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

def main():
    sh_path = os.path.join(SCRIPT_DIR, "track_runner_error_short.sh")
    with open(sh_path, "rb") as f:
        raw = f.read().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    b64 = base64.b64encode(raw).decode()
    cmd = f"echo '{b64}' | base64 -d | bash -e"
    r = subprocess.run(
        [sys.executable, os.path.join(SCRIPT_DIR, "connect_vm.py"), cmd],
        cwd=SCRIPT_DIR,
        timeout=260,
        capture_output=True,
        text=True,
    )
    out = (r.stdout or "") + (r.stderr or "")
    print(out)
    return 0 if r.returncode == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
