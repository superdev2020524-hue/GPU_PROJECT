# PHASE3 Debug Handoff (2026-03-10)

## Scope
- Goal: make `ollama` GPU mode in VM complete model load/inference without runner crash.
- Constraint repeatedly enforced: distinguish `HOST (xcp-ng/dom0)`, `VM (test-3 guest)`, and `LOCAL PC`.
- Validation style used: find -> analyze logs -> patch -> deploy -> verify.

## Current Outcome
- Host mediator path is alive and sustained:
  - `HtoD` progress reached beyond earlier failure points (up to ~5 GB in latest observed run).
  - Multiple large `cuMemAlloc` calls succeeded repeatedly (4 GB and ~1.38 GB allocations).
- Final failure still occurs in VM runner startup path:
  - `llama runner terminated: exit status 2`
  - `CUDA error: CUBLAS_STATUS_NOT_INITIALIZED`
  - callsite in Ollama logs: `cublasCreate_v2(&cublas_handles[device])`.

## Files Changed During This Debug Cycle
- `phase3/guest-shim/libvgpu_cudart.c`
  - Hardened device property sanitization to ensure non-zero values for additional fields used by upper layers.
  - Added handling for `cudaDevAttrMaxBlocksPerMultiprocessor` (`106`) returning safe non-zero value.
  - Updated occupancy path to avoid mutating caller input (`blockSize`), using local sanitized value instead.
- `phase3/guest-shim/libvgpu_cuda.c`
  - Added enum value and handling for `CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR` (`106`).
  - Added post-read clamping for that attribute.
- `phase3/guest-shim/libvgpu_cublas.c`
  - Replaced no-op success stubs with forwarding implementation that tries to load real `libcublas.so.12` and forwards by `dlsym`.
  - Added candidate real-library paths and fallback behavior.
  - Added status-string improvement so non-success codes are not always mislabeled as success.
  - Added candidate path `/usr/local/lib/ollama/cuda_v12/libcublas.so.12` to match VM reality.
- Transfer scripts (timeouts/robustness):
  - `phase3/transfer_libvgpu_cuda.py`
  - `phase3/transfer_guest_shim_transport_cudart.py`
  - `phase3/transfer_libvgpu_cublas.py`
  - Changes: increased VM command timeout windows and changed remote hash verification to `sha256sum | awk` for reliability.

## Deploy/Verify Work Performed
- Repeatedly transferred and rebuilt guest shims in VM using transfer scripts.
- Restarted `ollama` after shim deployments.
- Ran long, no-timeout `api/generate` probes from VM side.
- Correlated VM `journalctl -u ollama` failure timing with HOST `/tmp/mediator.log` transfer/allocation progress.

## Key Findings (Important)
- A major earlier defect was confirmed and fixed:
  - `libvgpu_cublas.c` previously returned success without executing real math for important CUBLAS APIs.
  - This was replaced with real-call forwarding.
- Current unresolved issue narrowed:
  - Failure mode changed from misleading `CUBLAS_STATUS_SUCCESS` message to explicit `CUBLAS_STATUS_NOT_INITIALIZED`.
  - Indicates progress in error reporting, but initialization failure remains.
- VM library location fact established:
  - Real CUBLAS exists at `/usr/local/lib/ollama/cuda_v12/libcublas.so.12`.
  - Simple direct load of this file in VM succeeded via Python `ctypes.CDLL(...)`.
  - However, runtime inside full Ollama/runner still ends with CUBLAS init failure.

## Most Recent Confirmed State
- Host side (latest slice): continued `HtoD` growth and successful large allocations.
- VM side (same run end): HTTP 500 after ~40 minutes, runner exit status 2, `CUBLAS_STATUS_NOT_INITIALIZED` on `cublasCreate_v2`.

## What This Means
- Not a host-transfer stall at the observed failure point.
- Remaining blocker is in VM/runtime CUBLAS initialization context (library load/dependency/runtime context mismatch), not just raw data movement.

## Suggested Next Owner Actions
- Focus VM-only first (fast iteration):
  - ✅ **DONE (Mar 12):** Add explicit logging in `libvgpu_cublas.c` for:
    - chosen `dlopen` path,
    - `dlerror()` text on each failed candidate,
    - `dlsym("cublasCreate_v2")` resolution result,
    - return code from real `cublasCreate_v2`.
    - **Output:** `/tmp/vgpu_cublas_init_diag.txt` (written on every `cublasCreate_v2` call).
    - **Gated stderr:** Set `VGPU_DEBUG` or `CUBLAS_DEBUG` for init_real_cublas summary on stderr.
  - Add temporary `LD_DEBUG=libs` for runner process launch path (if feasible) to capture real loader decisions.
  - Verify dependency chain of `/usr/local/lib/ollama/cuda_v12/libcublas.so.12` in VM runtime context.
- Keep host passive monitor only while iterating VM shim logs to avoid long blind cycles.

## Notes About Process
- Multiple long runs were consumed due to late-failing model load path (~40 minutes each run).
- The debugging direction converged from broad runtime hardening to a narrowed CUBLAS init blocker in VM context.
