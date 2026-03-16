#!/usr/bin/env python3
"""
Capture runner stderr by running inference in foreground (ollama run) so the
runner's stderr is visible. Use when "exit status 2" has no detailed message in journalctl.

Usage: python3 capture_runner_stderr.py

On the VM this will:
  1. Run: OLLAMA_DEBUG=1 timeout 30 ollama run llama3.2:1b Hi 2>&1
  2. Print the full output (including any CUDA/GGML error from the runner)
"""
import subprocess
import sys

SCRIPT_DIR = __file__.rpartition("/")[0] or "."
sys.path.insert(0, SCRIPT_DIR)

def main():
    cmd = (
        "OLLAMA_DEBUG=1 timeout 30 ollama run llama3.2:1b Hi 2>&1 "
        "|| true"
    )
    r = subprocess.run(
        [sys.executable, f"{SCRIPT_DIR}/connect_vm.py", cmd],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=SCRIPT_DIR,
    )
    out = (r.stdout or "") + (r.stderr or "")
    print(out)
    return 0

if __name__ == "__main__":
    sys.exit(main())
