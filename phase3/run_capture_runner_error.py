#!/usr/bin/env python3
"""
Capture the llama runner's CUDA error when generate fails.

Run this from a machine that can reach the VM (e.g. your workstation).
Uses connect_vm.py to run diagnostic commands on the VM.

  python3 run_capture_runner_error.py
"""
import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from connect_vm import connect_and_run_command

def main():
    # Trigger generate and capture ollama journal
    cmd = """
curl -s -X POST http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.2:1b","prompt":"Hi","stream":false}' \
  > /tmp/gen_out.json 2>&1 &
sleep 10
echo "=== Journal grep (CUDA/error/runner) ==="
sudo journalctl -u ollama -n 150 --no-pager 2>/dev/null \
  | grep -iE "CUDA error|error:|exit status|Load failed|runner|ggml" | tail -25
echo ""
echo "=== Generate response ==="
head -3 /tmp/gen_out.json 2>/dev/null
echo ""
echo "=== Last 20 ollama lines ==="
sudo journalctl -u ollama -n 20 --no-pager 2>/dev/null
"""
    result = connect_and_run_command(cmd)
    if result:
        print(result)
        return 0
    return 1

if __name__ == "__main__":
    sys.exit(main())
