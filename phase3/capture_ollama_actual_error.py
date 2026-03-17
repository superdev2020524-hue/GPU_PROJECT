#!/usr/bin/env python3
"""
Capture the actual Ollama (Oyu) error by running the server in foreground with
full logging, then triggering a generate. Server + runner output go to a log file.

Uses the PATCHED binary (ollama.bin) so discovery fix (NeedsInitValidation skip) is tested.
The default 'ollama' script on the VM may exec ollama.real (unpatched); we explicitly
start /usr/local/bin/ollama.bin serve.

Steps:
1. On VM: stop ollama service, start "OLLAMA_DEBUG=1 /usr/local/bin/ollama.bin serve" with output to /tmp/ollama_actual_error.log
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
        "sudo pkill -f 'ollama serve' 2>/dev/null; sudo pkill -f 'ollama.real' 2>/dev/null; sudo pkill -f 'ollama.bin' 2>/dev/null; "
        "sleep 2; "
        "rm -f /tmp/ollama_actual_error.log; "
        "LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama OLLAMA_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama OLLAMA_LLM_LIBRARY=cuda_v12 OLLAMA_NUM_GPU=1 OLLAMA_DEBUG=1 nohup /usr/local/bin/ollama.bin serve >> /tmp/ollama_actual_error.log 2>&1 & "
        "sleep 5; "
        "echo SERVER_STARTED"
    )
    if "SERVER_STARTED" not in out:
        print("Failed to start server:", out)
        return 1
    print("   Server started in background.")

    print("2. Triggering generate request (10min timeout so load completes and runner crash is visible)...")
    run_vm('curl -s -m 600 -X POST http://127.0.0.1:11434/api/generate '
           '-H "Content-Type: application/json" '
           '-d \'{"model":"llama3.2:1b","prompt":"Hi","stream":false,"options":{"num_predict":2}}\' 2>&1 || true',
           timeout=620)
    time.sleep(5)

    print("3. Reading log for actual error...")
    out2, _ = run_vm(
        "sleep 2; cat /tmp/ollama_actual_error.log 2>/dev/null | tail -500"
    )
    print("--- Last 500 lines of server log ---")
    print(out2)

    print("\n4. Grepping for error/CUDA/ggml/fail/exit/assert/signal...")
    out3, _ = run_vm(
        "grep -iE 'error|cuda|ggml|fail|exit|panic|fatal|assert|status|runner|terminated|signal|SIGABRT|killed|abort' "
        "/tmp/ollama_actual_error.log 2>/dev/null | tail -100"
    )
    print("--- Matching lines ---")
    print(out3 if out3.strip() else "(no matches)")
    print("\n5a. Full log line count and any line containing 'error' or 'Error' (actual error string):")
    out4, _ = run_vm(
        "wc -l /tmp/ollama_actual_error.log 2>/dev/null; echo '---'; grep -n 'error\\|Error\\|failed\\|Failed\\|assert\\|panic\\|CUDA\\|exit status' /tmp/ollama_actual_error.log 2>/dev/null | tail -50"
    )
    print(out4)

    print("\n5. Restarting ollama service...")
    run_vm("sudo systemctl start ollama 2>/dev/null; echo DONE")

    return 0

if __name__ == "__main__":
    sys.exit(main())
