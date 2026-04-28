# Phase 3 SGEMM Debug Progress

Last updated: 2026-03-13

## Current state

- The guest-side startup path has advanced well past the earlier crashes in `cuGetExportTable()`, `cuModuleLoadData`, and missing symbol export issues.
- `cublasCreate_v2()` now succeeds inside the VM with the active vGPU `libcuda.so.1` shim.
- The current blocker is no longer generic initialization. It is the first real CUBLAS compute path:
  - `cublasSgemm_v2(N,N) -> CUBLAS_STATUS_NOT_SUPPORTED`
  - `cublasSgemm_v2(T,N) -> CUBLAS_STATUS_NOT_SUPPORTED`
- This was reproduced with a tiny VM-local test program, so the failure is not specific to GGML or Ollama's higher-level call pattern.
- New isolated temp-shim experiments show that the old SGEMM `NOT_SUPPORTED` result is no longer the deepest blocker once the experimental UUID path is enabled correctly.

## What was fixed before reaching SGEMM

- `cuModuleLoadData` payload corruption was fixed by flushing BAR0 writes before the doorbell and correcting small BAR0-inline bulk transfer handling in `guest-shim/cuda_transport.c`.
- Missing `cuMemcpy2DAsync_v2` export was fixed by exporting it as an alias to the existing `cuMemcpy2DAsync` path in `guest-shim/libvgpu_cuda.c`.
- The dark API compatibility path in `cuGetExportTable()` was improved:
  - `TOOLS_TLS` table size/layout was corrected to a 4-slot table.
  - `CONTEXT_WRAPPER` / context-check table was updated to match the newer ABI shape used by recent CUDA user-space.
  - Null dark-API function pointers that previously caused `SIGSEGV` were replaced with callable stubs.
- These changes moved the failure from early startup crashes to the first real matrix operation.

## New debugging work completed in this session

### 1. Added a VM-local SGEMM reproducer

- `phase3/guest-shim/test_cublas_vm.c` was expanded from a simple `cublasCreate_v2()` probe into a real SGEMM reproducer.
- The test now:
  - loads `libcuda.so.1` and `libcublas.so.12`
  - creates/sets a CUDA primary context
  - allocates small device buffers
  - copies tiny host matrices to device memory
  - calls `cublasSgemm_v2()` in both `N,N` and `T,N`
  - attempts to copy the result back
- Result:
  - `cublasCreate_v2()` succeeds
  - both SGEMM calls return `15 (NOT_SUPPORTED)`

### 2. Improved dark context-local storage behavior

- In `guest-shim/libvgpu_cuda.c`, dark context-local storage callbacks were updated so `ctx == NULL` resolves to the active context instead of acting like a separate fake namespace.
- Missing lookup now returns `CUDA_ERROR_INVALID_HANDLE` instead of a generic not-found status.
- This matched runtime behavior better, but it did not change the SGEMM result.

### 3. Instrumented `TOOLS_TLS`

- The generic `TOOLS_TLS` callbacks were replaced with slot-specific logging handlers.
- Observation:
  - SGEMM repeatedly drives `TOOLS_TLS` slot 2.
  - This is now a concrete active path, not just a placeholder compatibility path.

### 4. Identified a new SGEMM-path dark API UUID

- Immediately before SGEMM returns `NOT_SUPPORTED`, CUBLAS requests:
  - `21318c60-9714-3248-8ca6-41ff7324c8f2`
- That UUID is currently unknown in our shim, so `cuGetExportTable()` logs it as unknown.
- This is the strongest current lead.

### 5. Ran one guarded experiment and rolled it back

- An experimental table was briefly returned for UUID `21318c60-9714-3248-8ca6-41ff7324c8f2`.
- Outcome:
  - SGEMM no longer returned `NOT_SUPPORTED`
  - the process crashed immediately after the UUID was handed back
- Interpretation:
  - the UUID is definitely active in the SGEMM path
  - its callback ABI/table shape matters
  - a guessed table is not safe
- That experiment was rolled back, and the VM was restored to the last non-crashing state.

### 6. Mapped the new UUID inside `libcublas.so.12`

- VM-side binary inspection of `/usr/local/lib/ollama/cuda_v12/libcublas.so.12` showed that UUID `21318c60-9714-3248-8ca6-41ff7324c8f2` is stored immediately next to `TOOLS_TLS`, not in an unrelated area.
- Disassembly showed `libcublas.so.12` initializes and caches this UUID in the same general dark-API family as `TOOLS_TLS`.
- This strongly suggests the UUID is a companion helper interface for the SGEMM / cuBLASLt path rather than an unrelated probe.

### 7. Captured the experimental crash under `gdb`

- A temporary VM-only shim swap was used to return an experimental table for the new UUID, run the tiny SGEMM reproducer, and immediately restore the stable `/opt/vgpu/lib/libvgpu-cuda.so.1`.
- `gdb` showed:
  - the crash happens in `libcublasLt.so.12`
  - the path goes through `cublasLtSSSMatmulAlgoGetHeuristic()`
  - the caller executes `call *0x20(%rax)` on the object returned for UUID `21318c60-...`
- Important ABI result:
  - this interface needs at least a callable entry at offset `0x20`
  - the earlier 4-entry guess was definitely too short

### 8. Confirmed slot 4 is live and moved the failure further

- A second temporary VM-only experiment returned a 5-entry table with a logging callback at offset `0x20` (slot 4).
- Result:
  - the slot-4 callback was actually called
  - its observed call pattern was:
    - `a1 = NULL`
    - `a2 = <stack output pointer>`
- After that, the SGEMM path advanced beyond the earlier immediate crash and began hitting repeated:
  - `cuLibraryGetKernel() NOT_SUPPORTED STUB CALLED`
- This is strong evidence that:
  - the new UUID ABI is partially understood now
  - satisfying slot 4 unlocks a deeper cuBLASLt kernel-loading path
  - the next blocker after that path is kernel/library lookup support

### 9. Replaced `cuLibraryGetKernel()` / `cuKernelGetFunction()` stubs

- In the stable working tree, `guest-shim/libvgpu_cuda.c` was updated so:
  - `cuLibraryGetKernel()` resolves the underlying library module and then calls `cuModuleGetFunction()`
  - `cuKernelGetFunction()` unwraps the resulting handle back to a `CUfunction`
- This stable change was deployed to the VM.
- Verification:
  - the normal non-experimental reproducer still behaves like the stable baseline
  - `cublasCreate_v2()` succeeds
  - SGEMM still returns `NOT_SUPPORTED` while the new UUID remains unresolved
  - no regression was introduced in the stable path

### 10. Found and fixed a shim self-load bug in `cuGetProcAddress()`

- While re-running the experimental UUID test through a temp `libcuda.so.1` placed in `/tmp/expvgpu`, the process behaved inconsistently:
  - a direct `cuGetExportTable(21318c60-...)` probe succeeded under the temp shim
  - but `test_cublas_vm` still fell back to `UNKNOWN UUID`
- `LD_DEBUG=libs` showed why:
  - `test_cublas_vm` correctly loaded `/tmp/expvgpu/libcuda.so.1`
  - but `cuGetProcAddress()` then loaded `/opt/vgpu/lib/libvgpu-cuda.so.1` by hard-coded path
  - that created a second shim image in the same process and routed symbol resolution back to the installed stable copy
- `guest-shim/libvgpu_cuda.c` was updated so `cuGetProcAddress()`:
  - uses `dladdr()` on `cuGetProcAddress` itself
  - tries to open the currently loaded shim image first
  - only falls back to hard-coded install paths if self-resolution fails
- This fixes a real guest-side correctness bug, not just an experimental workflow issue.

### 11. Experimental UUID path now advances past `NOT_SUPPORTED`

- After rebuilding the temp experimental shim with the `cuGetProcAddress()` self-load fix:
  - the SGEMM process stayed on `/tmp/expvgpu/libcuda.so.1`
  - `cuGetExportTable(21318c60-...)` succeeded inside the actual `test_cublas_vm` process
  - `dark_experimental_sgemm_probe_slot4()` was called with:
    - `a1 = NULL`
    - `a2 = <stack pointer>`
    - `before = <returned table pointer>`
- Most importantly:
  - the run no longer immediately returns `CUBLAS_STATUS_NOT_SUPPORTED`
  - instead it proceeds into a large `MODULE_LOAD` stream over the guest transport

### 12. Newly exposed blocker: very large module load over BAR0

- Once the experimental UUID path is satisfied enough to continue, the guest starts sending a large ELF/module payload through repeated `MODULE_LOAD` chunks.
- In a 60-second bounded temp-shim run:
  - the process progressed far beyond the old SGEMM failure point
  - but did not finish before timeout
- In a 180-second bounded temp-shim run:
  - the process was still actively streaming `MODULE_LOAD` chunks
  - chunk sequence numbers had reached the high 6000s
  - the run again ended by timeout, not by a CUDA or cuBLAS error
- This means the next exposed problem is no longer the original SGEMM dark-API gate. It is now the cost/path used to move the larger cuBLAS/cuBLASLt module payload.

### 13. Found and removed a forced 1024-byte BAR0 module-upload path

- In `guest-shim/cuda_transport.c`, `cuda_transport_call_module_load_chunked()` was still hard-coded to:
  - split module images into `1024`-byte chunks
  - force BAR0-inline transport for every chunk
- That explained why the large post-UUID path looked like thousands of tiny `MODULE_LOAD` requests even after the earlier transport fixes.
- The chunking logic was updated to use `max_single_payload(tp)` instead, so module images can use the active large-payload transport path.

### 14. New transport split after the module-upload fix

- After rebuilding the temp experimental shim with the transport chunking fix:
  - non-root VM runs no longer use tiny BAR0 chunks
  - they immediately use a single larger `bar1` module transfer
- Observed result for the first post-UUID module load:
  - `len=9360`
  - `path=bar1`
  - `MODULE_LOAD written first8=0000000000000000`
  - transport returns an error
- This confirms the old BAR1 corruption bug is still real for this path: BAR1 is not a safe fallback for these larger module loads.

### 15. Root-only probe proved shmem is the correct path

- Running the same temp experimental shim as `root` inside the VM changes the picture:
  - shmem GPA resolution succeeds
  - the first larger module upload uses `path=shmem`
  - its payload is preserved correctly
- Concrete observation:
  - `MODULE_LOAD source seq=15 len=9360 ... path=shmem`
  - `MODULE_LOAD written seq=15 len=9360 first8=50ed55ba01001000`
- That is the first proof in this session that the larger post-UUID module path can move correctly when shmem is active.

### 16. Shmem registration and later shmem module load failure are different issues

- Two separate shmem-related facts were established:

- As non-root:
  - shmem setup fails in the guest at GPA resolution
  - exact message:
    - `Cannot resolve GPA for shmem (need CAP_SYS_ADMIN or /proc/self/pagemap access) — using BAR1`

- As root:
  - GPA resolution succeeds, but shmem registration can still be rejected by the stub in some runs
  - guest logging was improved so this path now reports both status and error code

- In the latest short root probe:
  - shmem became active and the first `9360`-byte module load succeeded
  - a later `28120`-byte shmem module load failed with transport error `0x5`
  - `cublasCreate_v2()` then returned `3 (ALLOC_FAILED)`

- This means:
  - guest privilege is one blocker for reaching shmem at all
  - but once shmem is active, there is still a second host-side CUDA/module-load failure later in the cuBLAS initialization path

### 17. The failing 28120-byte image is not obviously invalid by itself

- Historical VM-side module-load diagnostics show both:
  - `wrapper-fatbin size=9360`
  - `wrapper-fatbin size=28120`
  successfully loaded in earlier runs
- So the current `28120` failure is not simply “that image size or wrapper-fatbin kind is unsupported.”
- The remaining problem is now narrower:
  - either the later execution context/path is different
  - or the host CUDA side is failing on that second load only under the now-correct post-UUID path

## Current safe deployed state

- The VM is back on the non-crashing version of `libvgpu-cuda.so.1`.
- The current behavior is:
  - `cublasCreate_v2()` succeeds
  - SGEMM returns `NOT_SUPPORTED`
  - no experimental crashing UUID response is currently deployed
- Useful new logging remains deployed:
  - `TOOLS_TLS` slot-specific logging
  - improved dark context-local storage handling
  - real `cuLibraryGetKernel()` / `cuKernelGetFunction()` implementations
- The newly found `cuGetProcAddress()` self-load fix exists in the working tree, but has not been deployed to the stable VM system copy yet.
- Experimental UUID handling was tested only via a temp shim under `/tmp/expvgpu`, not left installed system-wide.
- The module-upload chunking fix in `guest-shim/cuda_transport.c` also exists only in the working tree / temp experimental builds so far.

## Current root-cause hypothesis

The original SGEMM blocker was a combination of:

- a missing/ABI-incomplete dark API helper for `21318c60-9714-3248-8ca6-41ff7324c8f2`
- and a separate shim correctness bug where `cuGetProcAddress()` could silently load a second copy of the shim from `/opt/vgpu/lib`

The current remaining blocker exposed after those fixes is now:

- guest privilege / configuration needed to enable shmem in normal runs
- plus a later host-side CUDA/module-load failure that still occurs on a subsequent `28120`-byte shmem module load even after the first larger shmem upload succeeds

In other words:

- startup compatibility is mostly solved
- small/normal module loading is working
- context creation is working
- CUBLAS handle creation is working
- the first real GEMM path required the undocumented helper object/table behind UUID `21318c60-...`
- once that helper is partially satisfied, execution advances into a much larger module-load transfer path
- BAR0 chunking was an avoidable transport bottleneck and has been fixed locally
- BAR1 is still corrupt for this larger module-load path
- shmem is the right direction, but normal user-mode runs cannot currently activate it because GPA resolution fails in the guest
- even with shmem active under root, a later `28120`-byte module load still triggers a host-side CUDA/module-load failure

## Recommended next direction

1. Keep the stable deployed VM on the non-experimental path unless doing a short isolated test.
2. Deploy or temporarily test the `cuGetProcAddress()` self-image fix whenever further experimental UUID work is needed, otherwise results may be masked by the installed stable shim.
3. Keep using the module-upload chunking fix in temp builds; it removes the misleading BAR0 bottleneck and exposes the real next failure much faster.
4. Solve normal-user shmem enablement in the guest so larger module uploads do not fall back to corrupt BAR1.
5. Once shmem is available in the normal path, focus on the later `28120`-byte shmem module-load failure, which now looks like a concrete host CUDA/module-load error rather than raw transport corruption.

## Files most relevant now

- `phase3/guest-shim/libvgpu_cuda.c`
- `phase3/guest-shim/test_cublas_vm.c`
- `phase3/guest-shim/cuda_transport.c`
- `phase3/deploy_and_run_cublas_test.py`
- `phase3/transfer_libvgpu_cuda.py`

## Important caution

- Avoid leaving guessed dark-API tables deployed on the VM unless the exact ABI is understood.
- The last experiment proved that returning a non-null table for the new UUID changes behavior immediately and can turn a controlled `NOT_SUPPORTED` failure into a crash.
