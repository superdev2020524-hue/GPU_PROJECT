# Restore GPU logic — exact checklist

When GPU discovery worked, the fix was:

1. **Service: no LD_PRELOAD, LD_LIBRARY_PATH with /opt/vgpu/lib first**  
   So the whole process tree (and the runner) only uses `LD_LIBRARY_PATH`; `dlopen` can load `libggml-cuda` and the shim.

2. **Runner gets LD_LIBRARY_PATH and OLLAMA_LIBRARY_PATH with /opt/vgpu/lib and cuda_v12 first**  
   So the backend loader sees cuda_v12 before the parent dir and loads the CUDA backend; libcuda resolves to the shim.

3. **Hopper lib in cuda_v12**  
   `libggml-cuda.so` built with sm_90 in `/usr/local/lib/ollama/cuda_v12/`, plus symlink `libggml-cuda-v12.so` → `libggml-cuda.so`.

4. **No CUBLAS or CUBLASLt shim in /opt/vgpu/lib**  
   So GGML gets real CUBLAS/CUBLASLt from cuda_v12 and init succeeds. Remove both:  
   `sudo rm -f /opt/vgpu/lib/libcublas.so.12 /opt/vgpu/lib/libcublasLt.so.12`

5. **NeedsInitValidation**  
   In `ml/device.go`, `NeedsInitValidation()` returns false for CUDA so devices are not filtered when verification runners crash.

6. **cuda_v12 symlinks**  
   `libcuda.so.1`, `libcudart.so.12` in cuda_v12 point to `/opt/vgpu/lib` shims.

---

## Apply on VM (in order)

```bash
# 1. Service override: no LD_PRELOAD, paths with /opt/vgpu/lib and cuda_v12 first
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/vgpu.conf << 'EOF'
[Service]
ExecStart=
ExecStart=/usr/local/bin/ollama.bin serve
Environment=LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama
Environment=OLLAMA_NUM_GPU=1
Environment=OLLAMA_LLM_LIBRARY=cuda_v12
Environment=OLLAMA_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama
EOF
sudo systemctl daemon-reload
```

```bash
# 2. No CUBLAS or CUBLASLt shim in /opt/vgpu/lib (either breaks discovery)
sudo rm -f /opt/vgpu/lib/libcublas.so.12 /opt/vgpu/lib/libcublasLt.so.12
```

```bash
# 3. Patched ollama.bin (device.go, server.go, discover/runner.go)
#    Use transfer_ollama_go_patches.py from host, then on VM:
cd /home/test-3/ollama
/usr/local/go/bin/go build -o ollama.bin .
sudo systemctl stop ollama
sudo cp ollama.bin /usr/local/bin/ollama.bin
sudo systemctl start ollama
```

```bash
# 4. Verify
sudo journalctl -u ollama -n 20 --no-pager | grep -E "inference compute|starting runner"
# Expect: inference compute ... library=CUDA ... (not id=cpu library=cpu)
```

---

## Patches that must be in the binary

- **ml/device.go:** `NeedsInitValidation()` → `return d.Library == "ROCm"` only.
- **llm/server.go:** Prepend `/opt/vgpu/lib` to runner `libraryPaths` and `OLLAMA_LIBRARY_PATH`; strip `LD_PRELOAD` from `cmd.Env`.
- **discover/runner.go:** `dirs = []string{dir, ml.LibOllamaPath}` (GPU lib dir first so backend loader sees cuda_v12 before parent).

---

## If discovery still shows CPU

- Confirm service has **no** `Environment=LD_PRELOAD` in vgpu.conf.
- Confirm `OLLAMA_LIBRARY_PATH` in the override has **cuda_v12 before** `/usr/local/lib/ollama`.
- Run runner by hand and check `/info`:
  `LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama OLLAMA_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama /usr/local/bin/ollama.bin runner --ollama-engine --port 39999`
  Then `curl -s http://127.0.0.1:39999/info` — should show a CUDA device, not `[]`.
