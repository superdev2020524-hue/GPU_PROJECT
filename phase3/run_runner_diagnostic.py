#!/usr/bin/env python3
"""
Capture llama runner error that causes "exit status 2".
Run from a machine that can reach the VM (connect_vm.py must work).

Usage: python3 run_runner_diagnostic.py

1. Triggers a generate request
2. Waits for it to fail
3. Fetches journalctl from VM to extract runner error (CUDA error: ..., etc.)
"""
import subprocess
import sys
import time

SCRIPT_DIR = __file__.rpartition("/")[0] or "."
sys.path.insert(0, SCRIPT_DIR)

def run_vm(cmd):
    """Run command on VM via connect_vm."""
    r = subprocess.run(
        [sys.executable, f"{SCRIPT_DIR}/connect_vm.py", cmd],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=SCRIPT_DIR,
    )
    combined = (r.stdout or "") + (r.stderr or "")
    if "Unexpected response" in combined or not combined.strip():
        return combined, 1  # connection failed
    return combined, r.returncode

def main():
    print("1. Triggering generate on VM...")
    run_vm('curl -s -X POST http://127.0.0.1:11434/api/generate '
           '-d \'{"model":"llama3.2:1b","prompt":"Hi","stream":false}\' >/dev/null 2>&1')

    print("2. Waiting for failure...")
    time.sleep(12)

    print("3. Fetching ollama journal (runner errors)...")
    out, rc = run_vm(
        "sudo journalctl -u ollama -n 180 --no-pager 2>/dev/null | "
        "grep -iE 'CUDA error|error:|Load failed|exit status|runner|ggml|LastErr|ERR' || echo '(no matches)'"
    )
    print(out)

    print("4. Last 40 journal lines...")
    out2, _ = run_vm("sudo journalctl -u ollama -n 40 --no-pager 2>/dev/null")
    print(out2)

    # Look for CUDA error in output
    combined = out + out2
    if "Unexpected response" in combined or (not out.strip() and not out2.strip()):
        print("\n*** VM unreachable. Run from a machine that can SSH to the VM.")
        print("    Or copy trigger_and_capture.sh to the VM and run: sudo ./trigger_and_capture.sh")
        return 1
    if "CUDA error" in combined:
        print("\n*** FOUND: CUDA error above - this is likely the failing call ***")
    elif "exit status" in combined:
        print("\n*** Runner exited but no 'CUDA error' in captured output. Try capture_runner_error.sh for full stderr. ***")

    return 0

if __name__ == "__main__":
    sys.exit(main())
