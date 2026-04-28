# Quick error capture on the VM

Use this to quickly see why a generate fails (runner exit status 2).

## One command (from host, in `phase3/`)

```bash
./quick_capture_errors.sh
```

Or:

```bash
python3 quick_capture_vm_error.py   # full capture: stop server, run with tee, generate, then show logs + shim files
```

`quick_capture_errors.sh` only triggers one generate and then prints:

- **`/tmp/ollama_errors_full.log`** – Written by shims when a call fails:
  - `[libvgpu-cudart] cudaMemcpy FAILED: ...` (runtime copy failure)
  - `[libvgpu-cuda] cuMemcpyHtoD FAILED: ...` / `cuMemcpyDtoH FAILED: ...` (driver copy failure)
  - Any stderr lines from the runner if `write()` interception is used (e.g. with LD_PRELOAD).
- **`/tmp/ollama_errors_filtered.log`** – Filtered stderr (errors/CUDA/ggml) when write interception is active.
- **`/tmp/vgpu_next_call.log`** – Last CUDA/cudart calls (htod, stream_sync, gemm_ex, etc.).
- **`/tmp/gen.json`** – API response (e.g. `{"error":"llama runner process has terminated: exit status 2"}`).

## If the error log is empty

Then the runner is likely:

1. **Hanging** in a call (e.g. after the last GEMM, before the next logged call), or  
2. **Exiting** from GGML (assert/abort) before our shim returns.

In that case the last lines of `/tmp/vgpu_next_call.log` show the last call that completed (e.g. `gemm_ex_return`). The next call (e.g. `cuLaunchKernel`, `cuMemcpyDtoH`) is where to look.

## Deploy updated shims after code changes

- CUDA shim (with HtoD/DtoH error logging, write interception):  
  `python3 transfer_libvgpu_cuda.py`
- Cudart shim (with cudaMemcpy failure logging):  
  `python3 transfer_libvgpu_cudart.py`

## VM config note

`LD_PRELOAD=/opt/vgpu/lib/libvgpu-cuda.so.1` was added to the VM’s `vgpu.conf` so the runner uses our `write()` interceptor; if runner stderr is still not in the log files, the runner may not inherit LD_PRELOAD (e.g. Go clears it).

---

## Error tracking results (runner exit status 2)

**What we know:** The last line in `/tmp/vgpu_next_call.log` is always **`gemm_ex_return`** (third successful `cublasGemmEx`). We never see any of these *after* that:

- `launch_kernel` / `launch_kernel_return`
- `stream_sync` / `stream_sync_return`
- `memcpy_dtoh` / `memcpy_dtoh_return`
- `ctx_sync` / `ctx_sync_return`
- `cublas_destroy`
- `gemm_batched` / `gemm_strided_batched`

So the runner exits **after** the last GEMM returns and **before** entering any of the above. That points to **application code** (GGML/llama.cpp) that runs between “GEMM returns” and the next CUDA/CUBLAS call—e.g. an error check that calls `exit(2)`.

**Next step to get the real error:** Capture the runner’s **stderr** (where GGML usually prints “CUDA error: …” or similar):

1. Confirm the runner process gets `LD_PRELOAD` (e.g. `cat /proc/$(pgrep -f 'ollama runner' | head -1)/environ | tr '\0' '\n' | grep LD_PRELOAD` on the VM while a generate is running).
2. If the runner does **not** get `LD_PRELOAD`, ensure the Ollama server passes it when spawning the runner (may require a patch to the Go code that `exec`s the runner).
3. Alternatively, run the server so the runner’s stderr is redirected to a file (if the server supports it) or run the runner binary directly under `strace -f` and inspect the last syscalls before `exit_group(2)`.
