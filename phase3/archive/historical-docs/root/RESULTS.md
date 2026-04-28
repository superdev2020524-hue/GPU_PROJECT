# Results

## What was done

1. **Restored backup shim** (`libvgpu-cuda.so.backup_working`) as `/opt/vgpu/lib/libcuda.so.1` and re-enabled `vgpu.conf`. Service stays **active**; no SEGV.
2. **Checked model load without shim**: Disabled vGPU (renamed `vgpu.conf`), restarted Ollama, ran generate. **Model load still failed** with the same blob path error. So the failure is **not** caused only by the shim.
3. **Checked blob file**: Blob exists, ~1.3 GB, starts with GGUF magic. Permissions and path are fine.
4. **Shim → VGPU-STUB → mediator**: With current/backup shim, main process and runner (when started from CLI with env) see VGPU at `0000:00:05.0`, `cuInit` succeeds, "CUDA ready", "device found".
5. **Code change**: In `libvgpu_cuda.c`, for blob (excluded) paths when `g_real_fopen_global` is NULL, added use of **RTLD_NEXT open + fdopen** so the blob is opened with libc’s `open`/`fdopen` and the `FILE*` is from libc. Deployed and tested. **Model load still fails** (same “unexpectedly reached end of file” / “failed to load model”).
6. **Re-enabled vGPU**: `vgpu.conf` restored, service using shim again.

## Current state

| Item | Status |
|------|--------|
| Ollama service | **active** |
| List models (API) | **works** |
| Shim loads, finds VGPU (0000:00:05.0), cuInit OK | **yes** (main and runner when env has LD_PRELOAD) |
| Discovery in service (id=cpu, total_vram=0 B) | Runner started by service likely does not get LD_PRELOAD |
| Model load / inference | **fails** (“unable to load model” / “unexpectedly reached end of file”) |

## Conclusion

- **Pipeline (shim → VGPU-STUB → mediator)** works when the process has the shim: VGPU is found, cuInit succeeds.
- **Model loading fails** with and without the shim on this VM, so something else (Ollama, env, or how the runner loads the blob) is also wrong.
- **Change to use RTLD_NEXT open+fdopen for blobs** did not fix model load; either that path is not used for the failing open, or the failure is elsewhere (e.g. runner without shim, or different open/read path).

## Installed shim

- **Currently in use**: Built from current `libvgpu_cuda.c` (with RTLD_NEXT open+fdopen for blob path), copied to `/opt/vgpu/lib/libcuda.so.1` after `make guest` on the VM.
- **Backup**: `~/phase3/guest-shim/libvgpu-cuda.so.backup_working` is still on the VM if you want to revert.
