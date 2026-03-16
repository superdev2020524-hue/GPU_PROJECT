# Ollama (Oyu) vGPU — current revisions and status

*Updated: Mar 15, 2026*

## Host-side revisions (cuda_executor.c)

1. **Primary context for module ops**  
   Module load, unload, get-function, and get-global use `cuCtxSetCurrent(exec->primary_ctx)` instead of `ensure_vm_context(exec, vm)`. Aligns with allocations and CUBLAS.

2. **Fat binary passing**  
   For payload magic `0xBA55ED50` (raw fat binary from guest), `load_host_module()` now tries **passing the raw pointer** to `cuModuleLoadFatBinary()` first. If that fails, it falls back to the wrapper (0x466243b1). On the current host driver, **raw** works and resolves INVALID_IMAGE.

3. **Memory and CUBLAS**  
   MEM_ALLOC, MEM_FREE, MEMCPY_*, MEMSET_*, MEM_GET_INFO and all CUBLAS calls already use primary context in this tree.

## Current outcome

- **Module load:** Host log shows `module-load done ... rc=0 name=CUDA_SUCCESS` for vm=9. INVALID_IMAGE is resolved.
- **After module load:** Host reports several more `cuMemAlloc` SUCCESS (1024, 131072, 8M, 8.8M, 2.2M, 2.2M bytes).
- **Guest:** Runner still exits with “CUDA error” (exit status 2); no detailed message in journal. Failure is **after** module load and post-load allocs.

## Next step

- Host log: after module-load rc=0, six cuMemAlloc SUCCESS; no CUBLAS_CREATE or Launch. Runner exits with "CUDA error" (exit 2); journal has no detailed string.
- Run analysis: host sees no CUBLAS_CREATE or Launch after post-module allocs; runner uses real libcublas. Next: capture runner stderr or OLLAMA_DEBUG=1 to get exact CUDA error; then fix path or consider CUBLAS shim for runner only.

## Diagnostics (Mar 15)

- **OLLAMA_DEBUG=1:** No extra CUDA error string in CLI or journal; only "CUDA error" from server.
- **run_runner_diagnostic.py:** Journal shows "llama runner terminated" / "Load failed ... CUDA error"; no runner stderr with exact message.
- **/tmp/vgpu_last_error:** Not created on failure → crash may be in real libcublas (not a remoted call returning error to shim) or runner exits before shim writes it.
- **Shim log (runner pid):** Only `cuInit() OK`; then many `ensure_init` calls. Host sees module-load rc=0 and six allocs, then no CUBLAS_CREATE (0x00AC) or Launch.
- **Conclusion:** Runner uses **real** libcublas (CUBLAS shim not in `/opt/vgpu/lib`). Real `cublasCreate` runs in-VM and may fail (e.g. "library was not initialized") and GGML exits with "CUDA error". Remoting CUBLAS to the host may fix it.

## CUBLAS symlink test (Mar 15)

- **Done:** Added symlinks `libcublas.so.12` → `libvgpu-cublas.so.12` and `libcublasLt.so.12` → `libvgpu-cublasLt.so.12` in `/opt/vgpu/lib` on the VM (`LD_LIBRARY_PATH` has `/opt/vgpu/lib` first).
- **Result:** Generate still fails (runner exit status 2). Host log still shows **no** `CUBLAS_CREATE` (0x00AC), **no** `MODULE_GET_FUNCTION` (0x0044), **no** Launch — only module-load rc=0 and the six post-module allocs, then more INIT/alloc from a later run.
- **Implication:** Runner likely crashes **before** sending CUBLAS_CREATE or the next CUDA call (e.g. in guest after the 6 allocs, or before our CUBLAS shim is invoked). CUBLAS symlinks left in place for now. If discovery starts reporting CPU again, remove them per **GPU_MODE_DO_NOT_BREAK.md**.

## Call-sequence trace (Mar 15)

- **Added:** In `cuda_transport.c`, before ringing the doorbell, append `call_id` and name to `/tmp/vgpu_call_sequence.log`. Deployed via `transfer_cuda_transport.py` (libvgpu-cuda.so.1 on VM).
- **Result:** After a failing generate, the log contained only `0x0030 cuMemAlloc_v2` (6 or 12 lines). So the **last call sent** before the crash is **cuMemAlloc**. The **next** call (never sent) is therefore where the failure occurs — either in guest code before the send (e.g. in `cuModuleGetFunction` or `cublasCreate` wrapper) or in GGML/runner between the 6th alloc return and the next CUDA/CUBLAS call.
- **Conclusion:** Crash is **after** the 6 post-module allocs return and **before** the guest sends the following request (likely `MODULE_GET_FUNCTION` 0x0044 or `CUBLAS_CREATE` 0x00AC). Next: add a small debug write at the **entry** of `cuModuleGetFunction` and `cublasCreate` in the guest shims to see which is reached first and whether we crash inside the shim before sending.

## Next-call entry debug (Mar 15)

- **Added:** At the very start of `cuModuleGetFunction` (libvgpu_cuda.c) and `cublasCreate_v2` (libvgpu_cublas.c), append `get_function` / `cublas_create` to `/tmp/vgpu_next_call.log`. Deployed via transfer_libvgpu_cuda.py and transfer_libvgpu_cublas.py.
- **Result:** After a failing generate, `/tmp/vgpu_next_call.log` was **not created** — so **neither** `cuModuleGetFunction` nor `cublasCreate_v2` was entered. The runner crashes **before** calling either.
- **Conclusion:** The failure is in guest code that runs after the last remoted call (e.g. after the 6 allocs or after cuMemcpyHtoD) and before the first `cuModuleGetFunction` or `cublasCreate`. Possibilities: another CUDA call (e.g. `cuLaunchKernel`, `cuCtxSynchronize`), or GGML/runner logic that crashes (e.g. invalid pointer, assert) before it reaches get-function or cublas create.

## cublasCreate RPC result (Mar 15)

- **Entry:** `cublas_create` in log; **RPC:** `cublas_rpc_rc=0 num_results=2` — remote cublasCreate **succeeds**. Crash is **after** cublasCreate returns (likely next CUBLAS call or GGML).

## set_stream / gemm_ex progression (Mar 15)

- **set_stream:** Reached and completes (`set_stream_done` in log). **gemm_ex:** Reached (multiple times). No `gemm_ex tc_rc=...` line after the first `gemm_ex` — crash or hang **inside** `cuda_transport_call(..., CUDA_CALL_CUBLAS_GEMM_EX, ...)` (guest never logs the return).
- **Conclusion:** Failure is either (1) host GEMM_EX handler: `vm_find_mem` returns NULL for one of A/B/C (host logs `MAPPING FAILED`), host returns error and guest might crash handling it; (2) host hangs or crashes in `cublasGemmEx`; (3) transport deadlock or timeout on the GEMM_EX request.

## Host log analysis (Mar 15)

- **Read from host:** `/tmp/mediator.log` (via `connect_host.py`). **Finding:** For vm=9, after HtoD and several successful `cuMemAlloc`, **cuMemAlloc FAILED: rc=2** for ~1.2 GB. **rc=2 = CUDA_ERROR_OUT_OF_MEMORY.** Current failure is **host GPU out of memory**. See **HOST_LOG_FINDINGS.md** for what to do on the host and GEMM checks after OOM is fixed.

## Next step (recommended)

- **On host (you):** Free GPU memory or use a smaller model so the ~1.2 GB allocation succeeds; then retry. If it then fails at GEMM, check the log for `cublasGemmEx` and `MAPPING FAILED` per HOST_LOG_FINDINGS.md.
- **On host:** When running a generate, check `/tmp/mediator.log` for `cublasGemmEx` and `MAPPING FAILED` for vm_id=9. If MAPPING FAILED appears, guest pointers for A/B/C are not in the host’s `vm_state` mem list — ensure allocations from the same VM are recorded so GEMM_EX can resolve them.
- **On guest:** Optional: increase transport timeout for GEMM_EX or add a “gemm_ex_after_send” log after the transport call returns to confirm whether the failure is timeout vs. crash.

- **Widen entry-point debug (done):** Add the same style of write to `/tmp/vgpu_next_call.log` at the start of `cuLaunchKernel` and `cuCtxSynchronize` in the guest shim; trigger generate and see which line appears. Alternatively, capture the runner process stderr (e.g. run ollama in foreground with the runner’s stderr visible) to get the exact CUDA/GGML error string.

## References

- **HOST_FIX_MODULE_LOAD_PRIMARY_CTX.md** — Host fix steps and rationale.
- **CURRENT_STATE_AND_DIRECTION.md** — Pipeline, permissions, timing.
- **PHASE3_PURPOSE_AND_GOALS.md** — Ultimate goal and first-stage (Ollama) milestone.
- **RUNNER_DIAGNOSTIC_README.md** — How to capture runner CUDA error.
