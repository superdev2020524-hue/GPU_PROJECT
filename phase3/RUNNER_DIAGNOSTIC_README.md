# Capturing llama runner "exit status 2" error

The host cublasGemmEx calls succeed (rc=0). The failure occurs in the llama runner subprocess on the VM. These scripts capture the runner's stderr to identify the failing CUDA call.

## When VM is reachable (SSH works)

### Option A: Automated (recommended)

```bash
cd /home/david/Downloads/gpu/phase3
python3 run_runner_diagnostic.py
```

Triggers generate, waits, then fetches journalctl from VM. Look for "CUDA error:" in the output.

### Option B: Manual journal check

From a machine that can SSH to the VM:

```bash
# 1. Trigger generate
curl -s -X POST http://127.0.0.1:11434/api/generate \
  -d '{"model":"llama3.2:1b","prompt":"Hi","stream":false}' &

# 2. Wait ~15 seconds
sleep 15

# 3. On the VM, run:
sudo journalctl -u ollama -n 200 --no-pager | grep -iE "CUDA error|Load failed|exit status|runner"
```

### Option C: Foreground Ollama (full stderr)

Copy `capture_runner_error.sh` to the VM and run:

```bash
sudo ./capture_runner_error.sh
```

Then from another terminal (on the VM), trigger generate. Runner stderr will appear in the script output and in the log file.

## Deploy scripts to VM

If you have another path to the VM (e.g. SCP from a different machine):

```bash
scp phase3/trigger_and_capture.sh phase3/capture_runner_error.sh test-3@10.25.33.11:~/
ssh test-3@10.25.33.11 "chmod +x trigger_and_capture.sh capture_runner_error.sh"
```

Then on the VM: `sudo ./trigger_and_capture.sh`

## Current status

- **Host:** cublasGemmEx remoting works (rc=0). Use `python3 connect_host.py "grep cublasGemmEx /tmp/mediator.log"` to verify.
- **VM:** Runner exits with status 2; need "CUDA error: ..." from runner stderr to identify failing call.
- **Network:** VM (10.25.33.11) may not be reachable from all environments (e.g. Cursor cloud workspace). Run diagnostics from a machine that can SSH to the VM.
