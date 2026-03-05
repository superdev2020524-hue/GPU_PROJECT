# GPU mode fix: Symlink Ollama libs to vGPU shims

## Why this is needed

- **Main Ollama process** gets our shims via `LD_PRELOAD` (in `vgpu.conf`) and sees the vGPU (`cuDeviceGetCount` returns 1).
- **Runner subprocess** (used for discovery and inference) may **not** inherit `LD_PRELOAD` and uses `LD_LIBRARY_PATH` with `/usr/local/lib/ollama` and `/usr/local/lib/ollama/cuda_v12`. If it loads **real** libcuda/libnvidia-ml from those dirs, it gets **0 devices** → Ollama logs `id=cpu`, `total_vram="0 B"`.
- **Fix:** Symlink **both** the parent dir and `cuda_v12` so that whichever path the runner uses, it loads our shims.

## Option A: Run the script on the VM

1. Copy the script to the VM (e.g. `scp phase3/VM_SYMLINK_CUDA_V12_FOR_GPU_MODE.sh test-11@10.25.33.111:~/`)
2. SSH to the VM and run:

   ```bash
   chmod +x ~/VM_SYMLINK_CUDA_V12_FOR_GPU_MODE.sh
   sudo ~/VM_SYMLINK_CUDA_V12_FOR_GPU_MODE.sh
   ```

3. If your shims are in a different path, set `VGPU_LIB` before running:

   ```bash
   sudo VGPU_LIB=/usr/lib64 ./VM_SYMLINK_CUDA_V12_FOR_GPU_MODE.sh
   ```

4. Restart Ollama and verify:

   ```bash
   sudo systemctl restart ollama
   sleep 5
   sudo journalctl -u ollama -n 80 --no-pager | grep -iE 'inference compute|total_vram|library=|discovering'
   ```

   Look for **id=gpu** (or similar), **library=cuda** (or cuda_v12), and **total_vram** non-zero.

## Option B: Commands to run manually on the VM

**Important:** Symlink in **both** `/usr/local/lib/ollama/cuda_v12/` and `/usr/local/lib/ollama/` (parent). The runner often resolves libs from the parent dir first.

```bash
VGPU_LIB=/opt/vgpu/lib
OLLAMA=/usr/local/lib/ollama
OLLAMA_CUDA_V12=/usr/local/lib/ollama/cuda_v12

# --- cuda_v12 ---
sudo mv -n "$OLLAMA_CUDA_V12/libcuda.so.1" "$OLLAMA_CUDA_V12/libcuda.so.1.backup" 2>/dev/null || true
sudo ln -sf "$VGPU_LIB/libcuda.so.1" "$OLLAMA_CUDA_V12/libcuda.so.1"
sudo mv -n "$OLLAMA_CUDA_V12/libcudart.so.12.8.90" "$OLLAMA_CUDA_V12/libcudart.so.12.8.90.backup" 2>/dev/null || true
sudo rm -f "$OLLAMA_CUDA_V12/libcudart.so.12"
sudo ln -sf "$VGPU_LIB/libcudart.so.12" "$OLLAMA_CUDA_V12/libcudart.so.12"
sudo ln -sf "$VGPU_LIB/libcudart.so.12" "$OLLAMA_CUDA_V12/libcudart.so.12.8.90"
sudo mv -n "$OLLAMA_CUDA_V12/libnvidia-ml.so.1" "$OLLAMA_CUDA_V12/libnvidia-ml.so.1.backup" 2>/dev/null || true
sudo ln -sf "$VGPU_LIB/libnvidia-ml.so.1" "$OLLAMA_CUDA_V12/libnvidia-ml.so.1"

# --- parent ollama dir (runner often loads from here first) ---
sudo mv -n "$OLLAMA/libcuda.so.1" "$OLLAMA/libcuda.so.1.backup" 2>/dev/null || true
sudo ln -sf "$VGPU_LIB/libcuda.so.1" "$OLLAMA/libcuda.so.1"
sudo mv -n "$OLLAMA/libnvidia-ml.so.1" "$OLLAMA/libnvidia-ml.so.1.backup" 2>/dev/null || true
sudo ln -sf "$VGPU_LIB/libnvidia-ml.so.1" "$OLLAMA/libnvidia-ml.so.1"

# Optional: force GPU count in discovery (add to vgpu.conf)
# Environment="OLLAMA_NUM_GPU=999"

sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Then check logs as above.

## Restoring originals

If you need to revert (e.g. run without vGPU):

```bash
OLLAMA_CUDA_V12=/usr/local/lib/ollama/cuda_v12
sudo mv -n "$OLLAMA_CUDA_V12/libcuda.so.1.backup" "$OLLAMA_CUDA_V12/libcuda.so.1" 2>/dev/null || true
# Repeat for libcudart.so.12*.backup and libnvidia-ml.so.1.backup if you created them
```

## After the fix

- **On the VM:** `journalctl -u ollama` should show inference compute **id** with a GPU ID and **total_vram** > 0.
- **On the host:** While running a model on the VM, `nvidia-smi` and `./vgpu-admin show-metrics` should show new jobs and GPU utilization.

## Option C: Ollama wrapper (so runner gets vGPU env)

If the runner subprocess does not inherit `LD_PRELOAD` from the service, replace `/usr/local/bin/ollama` with a wrapper that sets env and execs the real binary. Then every `ollama` invocation (including `ollama runner` spawned by the server) gets the vGPU libs.

**On the VM (use Unix line endings; avoid CRLF):**

```bash
# 0. Stop Ollama first (otherwise "Text file busy" when overwriting)
sudo systemctl stop ollama

# 1. Backup real binary
sudo cp /usr/local/bin/ollama /usr/local/bin/ollama.real

# 2. Create wrapper (paste the block below; heredoc ends at WRAP)
sudo tee /usr/local/bin/ollama << 'WRAP'
#!/bin/bash
export LD_PRELOAD="/opt/vgpu/lib/libnvidia-ml.so.1:/opt/vgpu/lib/libcuda.so.1:/opt/vgpu/lib/libcudart.so.12"
export LD_LIBRARY_PATH="/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:${LD_LIBRARY_PATH:-/usr/lib64}"
export OLLAMA_LLM_LIBRARY="cuda_v12"
export OLLAMA_NUM_GPU="999"
exec /usr/local/bin/ollama.real "$@"
WRAP
sudo chmod +x /usr/local/bin/ollama

# 3. Start Ollama and check
sudo systemctl start ollama
sleep 5
sudo journalctl -u ollama -n 50 --no-pager | grep -E 'inference compute|total_vram'
```

To **restore** the original binary: `sudo cp /usr/local/bin/ollama.real /usr/local/bin/ollama` then `sudo systemctl restart ollama`.

## If logs still show id=cpu

Applied on the VM (Mar 2): symlinks in both `cuda_v12` and parent `/usr/local/lib/ollama`, plus `OLLAMA_NUM_GPU=999` in vgpu.conf. Discovery still reports `id=cpu`; the runner may be started with an environment that does not use these paths, or discovery may use a different code path.

**Next steps:**

1. **Confirm whether inference uses GPU despite the log:** Run a model on the VM (e.g. `ollama run llama3.2:1b 'hello'`) and on the host run `watch -n 2 nvidia-smi` and `./vgpu-admin show-metrics`. If GPU memory or job count increases during the run, the remoting path is in use even if the discovery log says CPU.
2. **Inspect the runner:** When a request is in flight, find the runner PID (`pgrep -af "ollama runner"`) and check `cat /proc/<pid>/environ | tr '\\0' '\\n' | grep -E 'LD_|OLLAMA'` and `cat /proc/<pid>/maps | grep -E 'cuda|nvml|vgpu'` to see which libs the runner loaded.
3. **Trace discovery:** Run `strace -f -e openat,open -o /tmp/ollama_open.log ollama run llama3.2:1b "hi"` and inspect which libraries are opened during discovery.
