#!/usr/bin/env python3
"""
Capture the actual Ollama (Oyu) error by running the server in foreground with
full logging, then triggering a generate. Server + runner output go to a log file.

Steps:
1. On VM: stop ollama service, start "OLLAMA_DEBUG=1 ollama serve" with output to /tmp/ollama_actual_error.log
2. Wait for server to be ready
3. Trigger a generate request
4. Wait for request to finish (success or fail)
5. Read the log and extract any error lines (CUDA, ggml, error:, failed, exit, etc.)
"""
import subprocess
import sys
import time

SCRIPT_DIR = __file__.rpartition("/")[0] or "."
sys.path.insert(0, SCRIPT_DIR)

def run_vm(cmd, timeout=90):
    r = subprocess.run(
        [sys.executable, f"{SCRIPT_DIR}/connect_vm.py", cmd],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=SCRIPT_DIR,
    )
    return (r.stdout or "") + (r.stderr or ""), r.returncode

def main():
    print("1. Stopping ollama service and starting server with logging...")
    # Stop service, kill any remaining ollama, ensure port 11434 is free, then start.
    out, rc = run_vm(
        "sudo systemctl stop ollama 2>/dev/null; "
        "sleep 2; "
        "sudo pkill -f 'ollama serve' 2>/dev/null; sudo pkill -f 'ollama.real' 2>/dev/null; "
        "sleep 2; "
        "rm -f /tmp/ollama_actual_error.log; "
        "OLLAMA_DEBUG=1 nohup ollama serve >> /tmp/ollama_actual_error.log 2>&1 & "
        "sleep 5; "
        "echo SERVER_STARTED"
    )
    if "SERVER_STARTED" not in out:
        print("Failed to start server:", out)
        return 1
    print("   Server started in background.")

    print("2. Triggering generate request...")
    run_vm('curl -s -m 45 -X POST http://127.0.0.1:11434/api/generate '
           '-H "Content-Type: application/json" '
           '-d \'{"model":"llama3.2:1b","prompt":"Hi","stream":false}\' >/dev/null 2>&1 || true',
           timeout=60)
    time.sleep(3)

    print("3. Reading log for actual error...")
    out2, _ = run_vm(
        "sleep 2; cat /tmp/ollama_actual_error.log 2>/dev/null | tail -500"
    )
    print("--- Last 500 lines of server log ---")
    print(out2)

    print("\n4. Grepping for error/CUDA/ggml/fail/exit...")
    out3, _ = run_vm(
        "grep -iE 'error|cuda|ggml|fail|exit|panic|fatal|assert|status|runner|terminated' "
        "/tmp/ollama_actual_error.log 2>/dev/null | tail -80"
    )
    print("--- Matching lines ---")
    print(out3 if out3.strip() else "(no matches)")

    print("\n5. Restarting ollama service...")
    run_vm("sudo systemctl start ollama 2>/dev/null; echo DONE")

    return 0

if __name__ == "__main__":
    sys.exit(main())
