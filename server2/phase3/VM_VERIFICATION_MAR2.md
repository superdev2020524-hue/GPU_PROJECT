# VM verification (Mar 2) — direct interaction summary

**VM:** test-11@10.25.33.111 (via connect_vm.py)

---

## 1. Shim and symbol check

- **`/home/test-11/phase3/guest-shim/libvgpu-cuda.so.1`** (Mar 2 08:03): built on VM; `nm -D` shows **`T cuCtxGetFlags`** and **`T cuCtxSetLimit`** (and 140 exported symbols).
- **Installed in two places:**
  - **`/usr/lib64/libvgpu-cuda.so`** — updated from guest-shim build (121736 bytes).
  - **`/opt/vgpu/lib/libcuda.so.1`** — updated from guest-shim build; this is what Ollama uses via `LD_PRELOAD` in `vgpu.conf`.

---

## 2. Ollama service

- **Before:** Service was failing with `undefined symbol: cuCtxGetFlags` (then, after fixing that, with `undefined symbol: cuInit` when only `/usr/lib64` was updated).
- **Fix:** Copied the new shim to **`/opt/vgpu/lib/libcuda.so.1`** so the loader used by Ollama gets the fixed library.
- **After:** With only `libcuda.so.1` and `libcudart.so.12` in `LD_PRELOAD`, Ollama started but hit **SEGV** during startup when the three ggml interceptors were also in `LD_PRELOAD`.
- **Change:** Removed the ggml interceptors from `LD_PRELOAD` in `/etc/systemd/system/ollama.service.d/vgpu.conf`:
  - **Before:** `LD_PRELOAD=.../libggml-alloc-intercept.so:.../libggml-mmap-intercept.so:.../libggml-assert-intercept.so:.../libcuda.so.1:.../libcudart.so.12`
  - **After:** `LD_PRELOAD=/opt/vgpu/lib/libcuda.so.1:/opt/vgpu/lib/libcudart.so.12`
- **Backup:** Original config saved as `vgpu.conf.bak` in the same directory.
- **Result:** Ollama is **active (running)** and the API responds (`/api/tags`, `ollama list` work).

---

## 3. Current vgpu.conf (active)

```ini
[Service]
Environment="LD_PRELOAD=/opt/vgpu/lib/libcuda.so.1:/opt/vgpu/lib/libcudart.so.12"
Environment="LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:/usr/lib64"
Environment="OLLAMA_LLM_LIBRARY=cuda_v12"
```

---

## 4. Model load / GPU discovery

- **Model load:** `ollama run llama3.2:1b '...'` returned **500 Internal Server Error** with “unable to load model” and journal showed **“gguf_init_from_file_impl: failed to read magic”**. The blob file exists (~1.3 GB); the failure looks like a loader/blob-format or path issue, not the shim.
- **GPU discovery:** At startup the log shows **“inference compute id=cpu”** (total_vram="0 B"). So the main process is not seeing the vGPU as CUDA; the runner subprocess or discovery path may need the same env or additional setup to use the shim.

---

## 5. Summary

| Item | Status |
|------|--------|
| cuCtxGetFlags (and stubs) in built shim | Verified (exported) |
| New shim installed in `/opt/vgpu/lib` and `/usr/lib64` | Done |
| Ollama starts without SEGV | Yes (with ggml interceptors removed from LD_PRELOAD) |
| Ollama API (tags, list) | Working |
| Model run (llama3.2:1b) | Fails with “failed to read magic” on blob |
| GPU discovery in main process | Shows CPU only |

---

## 6. Model load fix (Mar 2, same session)

- **Cause:** With the shim, the model blob was opened via `syscall(open)` + `fdopen()`; the GGUF loader then failed with "failed to read magic". Without the shim (no `LD_PRELOAD`), the model loaded and ran ("Hi.").
- **Change in `libvgpu_cuda.c`:** For paths excluded from interception (e.g. `/.ollama/models/`, `/models/blobs/`), added lazy resolution of **libc’s real `fopen`**:
  - Try `dlopen("/lib/x86_64-linux-gnu/libc.so.6", RTLD_NOLOAD)` or load by path, then `dlsym(libc, "fopen")`.
  - Fallback: `dlsym(RTLD_NEXT, "fopen")` and use it only if the symbol is not our own `fopen` (so we use libc’s).
- **Result:** With the updated shim deployed to the VM, `ollama run llama3.2:1b 'Say one word.'` **succeeds** (e.g. output "Cloud"). Model load and inference work with the vGPU shim enabled.

**Next steps (optional):**  
- Re-add ggml interceptors to `LD_PRELOAD` one at a time if needed for alignment; see which (if any) triggers the SEGV.  
- Confirm on the host that mediator/CUDA executor and GPU utilization show activity when running inference.
