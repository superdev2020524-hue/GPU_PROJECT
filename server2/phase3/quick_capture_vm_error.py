#!/usr/bin/env python3
"""Quick capture: run server with tee, trigger tinyllama generate, extract error lines. Run from phase3: python3 quick_capture_vm_error.py"""
import subprocess
import sys

SCRIPT_DIR = __file__.rpartition("/")[0] or "."
sys.path.insert(0, SCRIPT_DIR)

def run_vm(cmd, timeout=120):
    r = subprocess.run(
        [sys.executable, f"{SCRIPT_DIR}/connect_vm.py", cmd],
        capture_output=True, text=True, timeout=timeout, cwd=SCRIPT_DIR,
    )
    return (r.stdout or "") + (r.stderr or ""), r.returncode

def main():
    env = "LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama OLLAMA_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama OLLAMA_LLM_LIBRARY=cuda_v12 OLLAMA_NUM_GPU=1 OLLAMA_DEBUG=1"
    log = "/tmp/quick_capture.log"
    print("1. Stop ollama, start server with tee to capture all output...")
    out, _ = run_vm(
        "sudo systemctl stop ollama 2>/dev/null; sleep 2; "
        "pkill -f 'ollama.bin serve' 2>/dev/null; pkill -f 'ollama.bin runner' 2>/dev/null; sleep 1; "
        "rm -f " + log + "; "
        "(" + env + " timeout 75 /usr/local/bin/ollama.bin serve 2>&1 | tee " + log + ") & "
        "sleep 6; echo READY"
    )
    if "READY" not in out:
        print("Start failed:", out[-500:])
        run_vm("sudo systemctl start ollama 2>/dev/null")
        return 1

    print("2. Trigger generate (tinyllama, 60s timeout)...")
    run_vm(
        "curl -s -m 60 -X POST http://127.0.0.1:11434/api/generate "
        "-d '{\"model\":\"tinyllama\",\"prompt\":\"Hi\",\"stream\":false,\"options\":{\"num_predict\":2}}' -o /tmp/gen.json -w '%{http_code}' 2>/dev/null || true",
        timeout=70
    )
    run_vm("sleep 4; pkill -f 'ollama.bin' 2>/dev/null; sleep 1")

    print("3. Extract error lines and last 120 lines...")
    out2, _ = run_vm(
        "grep -n -iE 'error|CUDA|ggml|invalid|fail|exit|status|set_tensor|backend|panic|abort|assert|terminated|Load failed|exit status 2' " + log + " 2>/dev/null | tail -80"
    )
    print("--- ERROR / KEY LINES ---")
    print(out2 if out2.strip() else "(none matched)")
    out3, _ = run_vm("tail -120 " + log + " 2>/dev/null")
    print("\n--- LAST 120 LINES OF LOG ---")
    print(out3)
    out4, _ = run_vm("echo '=== /tmp/ollama_errors_full.log ==='; cat /tmp/ollama_errors_full.log 2>/dev/null || echo '(empty)'; echo '=== /tmp/ollama_errors_filtered.log ==='; cat /tmp/ollama_errors_filtered.log 2>/dev/null || echo '(empty)'")
    print("\n--- SHIM ERROR CAPTURE FILES ---")
    print(out4)

    print("\n4. Restart ollama service...")
    run_vm("sudo systemctl start ollama 2>/dev/null; sleep 2; systemctl is-active ollama")
    return 0

if __name__ == "__main__":
    sys.exit(main())
