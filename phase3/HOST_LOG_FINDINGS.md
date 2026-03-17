# Host mediator log findings

*From reading `/tmp/mediator.log` on the host (read-only).*

## Why ~448 MiB at mediator start vs expected 1.6–1.7 GB after model transfer?

- **448 MiB** when you first run `./mediator_phase3` is the **mediator’s initial GPU use** (CUDA context, driver structures). The **model** (~1.2–1.3 GB) is allocated only when a VM connects and loads a model (HtoD + `cuMemAlloc` for the weight buffer).
- **Expected:** After model transfer, nvidia-smi could show **mediator base + model ≈ 448 MiB + 1.2 GB ≈ 1.6–1.7 GB** for the process.
- **If you see only ~1.2 GB (and not 1.6–1.7 GB):**  
  - The driver can report **process memory as the total of current device allocations**. The initial 448 MiB is often **one-time context/scratch**; after the 1.2 GB (and other) allocs, that initial use may be **freed or reclassified**, so the **reported total can drop to roughly the model + runtime allocs** (e.g. ~1.2 GB if only the model buffer is live at that moment). So **1.2 GB does not mean the model failed midway** — it can mean the base 448 MiB is no longer counted in the same way once larger allocs exist.
  - If the **model file** is ~1.3 GB and the **alloc** we do is ~1.2 GB (1,313,251,456 bytes), the small difference is normal (sizes in GiB vs GB, or metadata not in the single buffer). Full HtoD in the host log confirms the transfer completed.
- **Summary:** Seeing ~1.2 GB instead of 1.6–1.7 GB is consistent with the driver reporting mainly the large allocation(s); the model did not fail midway.

## Cause of current failure (vm=9)

The host log shows:

1. **Model load (HtoD)** for vm=9 completes (e.g. ~17GB transferred).
2. **Several `cuMemAlloc` SUCCESS** for vm=9: 4GB, ~419MB, 8.8MB, 2×2.2MB.
3. **`cuMemAlloc FAILED: rc=2`** for **1,313,251,456 bytes** (~1.2 GB).

In CUDA, **rc=2** is **`CUDA_ERROR_OUT_OF_MEMORY`**. The physical GPU on the host runs out of memory when allocating this buffer. The guest then sees the failure and reports *"unable to allocate CUDA0 buffer"* / *"error loading model"* and the runner exits.

So the **root cause** is **host GPU out of memory**, not GEMM pointer mapping or transport.

## What you can do on the host

1. **Check GPU memory**
   ```bash
   nvidia-smi
   ```
   See how much is used and by which processes.

2. **Free GPU memory**
   - Stop other GPU-using processes (other VMs, containers, or jobs).
   - Restart the mediator so it starts with a clean primary context and no leftover allocations from previous runs (if the mediator does not already clean up on exit).

3. **Use a smaller model**
   - e.g. `llama3.2:1b` instead of a 3B/7B model so the 1.2GB allocation (and total footprint) fits.

4. **Optional: clearer host log**
   In `cuda_executor.c`, in the `CUDA_CALL_MEM_ALLOC` case, when `rc != CUDA_SUCCESS` you can log the name:
   ```c
   } else {
       fprintf(stderr, "[cuda-executor] cuMemAlloc FAILED: rc=%d (vm=%u) %s\n",
               rc, call->vm_id, rc == 2 ? "(OUT_OF_MEMORY)" : "");
   }
   ```
   (CUDA error code 2 = OUT_OF_MEMORY.)

## GEMM path (after allocation succeeds)

After a mediator restart (GPU free), generate was retried. **Result:**

- **Allocations:** All `cuMemAlloc` for vm=9 **succeeded** (no OOM).
- **Guest log:** `cublas_create`, `set_stream`, `gemm_ex`, `gemm_ex_before_send`, **`gemm_ex tc_rc=0 num_results=1 status=0`** (multiple times) — GEMM RPC returns success.
- **Runner:** Still exits with **exit status 2** (no `launch_kernel` or `ctx_sync` in the guest log).

So the failure is **after** successful GEMM returns: either in GGML before the next CUDA call, or in a subsequent call (e.g. `cuLaunchKernel` / `cuCtxSynchronize`) that crashes before our entry log.

**Latest check (Mar 16):** Guest log shows **gemm_ex_return** but **no** stream_sync, memcpy_dtoh/dtod, launch_kernel, or ctx_sync (all 0 after adding cuStreamSynchronize logging). So the runner crashes **after** returning from the last GEMM and **before** calling `cuLaunchKernel` or `cuCtxSynchronize` — i.e. inside GGML or in another CUDA API we don’t yet log. Use **capture_runner_stderr.py** (or run `ollama run llama3.2:1b Hi` on the VM with OLLAMA_DEBUG=1) to try to get a concrete error; the server currently only reports "exit status 2".

**Event/stream logging (Mar 16):** Entry-point logs were added for `cuEventRecord`, `cuEventSynchronize`, and `cuStreamWaitEvent` in `libvgpu_cuda.c` and deployed. After a generate, `/tmp/vgpu_next_call.log` still ends with **gemm_ex_return** — no event_record, event_sync, or stream_wait_event. So the crash is after the last GEMM and before any of those calls (and before launch_kernel, ctx_sync, stream_sync, memcpy_dtoh/dtod). Next: get the actual error (e.g. capture_runner_stderr with ollama.bin, or run runner under strace/gdb); or add logging to more CUDA APIs to narrow further.

**Runner stderr via strace (Mar 16):** Ran server under `strace -f -ff` and triggered generate; see **ACTUAL_ERROR_VERIFICATION.md** and **track_runner_error.sh** / **run_track_runner_error.py**. Result: the llama runner (PID 40370) never wrote a CUDA/GGML error to stderr; its last output was normal load messages (`load_all_data: using async uploads for device CUDA0...`). The server then reported "timed out waiting for llama runner to start - progress 0.00". So the runner either hangs during async upload (before sending progress) or exits without writing an error.
