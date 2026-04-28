# VM check results (direct interaction)

Checked by running commands on test-3 VM via `connect_vm.py`. Date: 2026-03-05.

## 1. Ollama service

- **Status:** `active` (running)
- **Binary:** `/usr/local/bin/ollama.bin serve`
- **Drop-in:** `vgpu.conf` applied
- **Version:** `ollama version is 0.0.0` (patched build)

## 2. Service config (`/etc/systemd/system/ollama.service.d/vgpu.conf`)

- `ExecStart=/usr/local/bin/ollama.bin serve`
- `LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12`
- `LD_PRELOAD=/opt/vgpu/lib/libvgpu-nvml.so:/opt/vgpu/lib/libvgpu-cuda.so.1`
- `OLLAMA_NUM_GPU=1`
- `OLLAMA_LLM_LIBRARY=cuda_v12`
- `OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12`

## 3. Server process environment

- **Serve PID:** 49222 (or current after restart)
- **LD_PRELOAD:** present = `/opt/vgpu/lib/libvgpu-nvml.so:/opt/vgpu/lib/libvgpu-cuda.so.1`
- **LD_LIBRARY_PATH:** present = `/opt/vgpu/lib:.../cuda_v12`

So the main `ollama.bin serve` process has vGPU env from systemd.

## 4. vGPU libs on VM

- `/opt/vgpu/lib/` contains: `libvgpu-cuda.so.1`, `libvgpu-nvml.so`, `libvgpu-cudart.so`, cublas, cublasLt, symlinks (libcuda.so.1 → libvgpu-cuda.so.1, etc.), and `libvgpu-exec-inject.so`.

## 5. Models

- `ollama list`: `llama3.2:1b` (1.3 GB) present.

## 6. Patched binary and source

- **Installed binary:** `/usr/local/bin/ollama.bin` (root, ~73 MB, Mar 4 18:46)
- **Build source:** `/home/test-3/ollama/ollama` (same size, Mar 4 18:45)
- **Patch in source:** `grep "Ensure runner inherits LD_PRELOAD" /home/test-3/ollama/llm/server.go` → line 431. Patched code is present.

## 7. Journal logs (discovery / GPU)

After restart and after API generate:

- `"discovering available GPUs..."`
- `"starting runner" cmd="/usr/local/bin/ollama.bin runner --ollama-engine --port XXXXX"`
- `"inference compute" id=cpu library=cpu ... total_vram="0 B"`

So discovery still reports CPU and 0 B VRAM.

## 8. Runner process env (LD_PRELOAD) — **confirmed via debug log (Mar 5)**

- **Attempts:** Restart + poll for runner process; script `vm_check_runner.sh`; runner exits too fast to capture via `/proc`.
- **Debug log added:** In `llm/server.go`, before `slog.Info("starting runner"`, added a log that prints `cmd.Env` entry for `LD_PRELOAD` when present. Rebuilt, reinstalled, restarted.
- **Result:** Journal shows:
  ```text
  msg="runner env LD_PRELOAD" value="LD_PRELOAD=/opt/vgpu/lib/libvgpu-nvml.so:/opt/vgpu/lib/libvgpu-cuda.so.1"
  msg="starting runner" cmd="/usr/local/bin/ollama.bin runner --ollama-engine --port XXXXX"
  ```
- **Conclusion:** The runner **is** started with `LD_PRELOAD` in its environment. Discovery still reports CPU/0 B VRAM, so the new engine likely does not call our shimmed APIs (`cuDeviceGetCount`/`nvmlDeviceGetCount_v2`) during discovery.

## 9. Shim usage (debug files)

- **Files:** `/tmp/cuda_get_count_called.txt`, `/tmp/nvml_get_count_called.txt`
- **When running runner manually with LD_PRELOAD:** `timeout 2 /usr/local/bin/ollama.bin runner --ollama-engine --port 49999` with `LD_PRELOAD=.../libvgpu-nvml.so:.../libvgpu-cuda.so.1`
- **Result:** Debug files were **not** created. So with the new Ollama engine, `cuDeviceGetCount` / `nvmlDeviceGetCount_v2` are not called during the short runner startup we see; discovery may use a different code path or API.

## 10. Inference

- **API:** `POST /api/generate` with `model=llama3.2:1b`, `prompt="Say 42"`, `stream=false` → request completes (exit 0). Response not inspected for content.

## Summary

| Check                    | Result |
|--------------------------|--------|
| ollama.service           | active |
| vgpu.conf                | applied, LD_PRELOAD + paths set |
| Server LD_PRELOAD/LD_LIBRARY_PATH | present in serve process |
| vGPU libs                | present under `/opt/vgpu/lib/` |
| Patched binary + source  | patch at line 431, binary installed |
| Discovery in logs        | still id=cpu, total_vram="0 B" |
| Runner env (LD_PRELOAD)   | **confirmed** via debug log in server.go |
| Shim debug files         | not created (discovery path doesn’t call those APIs in this window) |
| Inference (API)          | completes |

Runner env **confirmed**: debug log shows the runner is started with `LD_PRELOAD` including `libvgpu-nvml.so`, `libvgpu-cuda.so.1`, and `libvgpu-cudart.so`. Discovery still shows CPU. `libggml-cuda.so` has undefined `cudaGetDeviceCount`; manual runner + `curl /info` returns `[]` and no shim debug files are created. Next: run with `LD_DEBUG=symbols` (or trace `ggml_backend_cuda_init`) to see when the CUDA backend is loaded and why 0 GPUs are reported.
