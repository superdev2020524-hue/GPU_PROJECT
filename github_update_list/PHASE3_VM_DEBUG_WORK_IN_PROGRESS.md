# Phase 3 VM-Side Debugging Work in Progress

**Last Updated:** 2026-03-06  
**Scope:** VM-only fixes for Ollama GPU inference under vGPU shim. No host-transfer experiments.

**Status (2026-03-06):** Transport and mediator are **working**. Model load is slow (~0.5 MB/s over remoting); use `curl -m 3600` for long loads. **Gated all hot-path logging** in `libvgpu_cudart.c` behind `cudart_debug_logging()` (VGPU_DEBUG / CUDART_DEBUG); without env set, model load avoids ~2000+ syscalls and should be faster.

---

## 1. Context and Goal

**Environment:**
- **VM:** test-3@10.25.33.11, running Ollama with vGPU shims (`libvgpu-cudart.so`, `libvgpu-cuda.so`, `libvgpu-cublas.so`, `cuda_transport.c`)
- **Host:** Physical machine with real GPU; mediator forwards CUDA calls via BAR0/BAR1
- **Ollama:** Uses `libggml-cuda.so`, `libcublas.so.12`, `libcublasLt.so.12` from `/usr/local/lib/ollama/cuda_v12/`

**Goal:** Fix all VM-side errors until `ollama run llama3.2:1b` with GPU succeeds and returns generated text.

---

## 2. Errors Identified and Fixes Applied

### 2.1 CUBLAS_STATUS_NOT_INITIALIZED

**Symptom:** `cublasCreate_v2` returns `CUBLAS_STATUS_NOT_INITIALIZED` (rc=1) when loading the real `libcublas.so.12` through our shim.

**Root cause:** `libcublas.so.12` calls `cuGetExportTable()` with two UUIDs to obtain internal CUDA driver interfaces. Our shim returned `CUDA_ERROR_NOT_SUPPORTED` or a dummy table, causing CUBLAS init to fail.

**Fix (libvgpu_cuda.c):**
- Implemented `cuGetExportTable()` handling for:
  - `6bd5fb6c-5bf4-e74a-8987-d93912fd9df9` (CUDART_INTERFACE)
  - `a094798c-2e74-2e74-93f2-0800200c0a66` (TOOLS_RUNTIME_CALLBACK_HOOKS)
- Provide minimal vtable arrays with non-NULL function pointers for required slots:
  - Slot 0: table size (bytes)
  - Slot 2: `dark_get_primary_context` → calls `cuDevicePrimaryCtxRetain` then `cuCtxSetCurrent`
  - Slots 1, 6, 7, 8, 9: stubs for get_module_from_cubin, launch_kernel, etc.
- `dark_tools_runtime_fn2` / `dark_tools_runtime_fn6`: return buffers (1 MiB each) for tooling callbacks.

**Code location:** `libvgpu_cuda.c` around line 4051; search for `cuGetExportTable`.

---

### 2.2 Compute Capability 0.0 (GGML Device Discovery)

**Symptom:** Log line `Device 0: NVIDIA H100 80GB HBM3, compute capability 0.0` despite correct `computeCapabilityMajor=9` and `0x148=9 0x14C=0`.

**Root cause:** GGML (or a dependent path) reads legacy `major`/`minor` fields. Our `cudaDeviceProp` struct and patch helper used wrong offsets:
- We patched `0x158`/`0x15C` as legacy major/minor, but `0x158` is `totalConstMem` in CUDA 12 layout.
- Our struct defines `major` at `0x15C`, `minor` at `0x160`.
- The patch helper was overwriting `0x150`/`0x154` (memoryClockRate, memoryBusWidth) and `0x158`/`0x15C`, corrupting layout.

**Fix (libvgpu_cudart.c):**
- In `cudaGetDeviceProperties_v2`:
  - Patch only safe offsets: `0x148`/`0x14C` (computeCapabilityMajor/Minor) and `0x15C`/`0x160` (legacy major/minor in our struct).
- In `patch_ggml_cuda_device_prop()`:
  - `offsets_major[] = {0x148, 0x15C}`; `offsets_minor[] = {0x14C, 0x160}`.
  - Removed `0x150`/`0x154` and `0x158`/`0x15C` from the patch loop to avoid corrupting other fields.

**Verification:** After fix, logs show `legacy_major=9 legacy_minor=0` and `0x15C=9 0x160=0`.

**Code location:** `libvgpu_cudart.c`:
- `patch_ggml_cuda_device_prop()` around line 677
- `cudaGetDeviceProperties_v2()` around line 739
- Legacy offset patching around lines 815–820

---

### 2.3 Transport Timeouts During Model Load

**Symptom:**
- `[cuda-transport] Timeout on call 0x0001 (seq=1)` (CUDA_CALL_INIT)
- `[cuda-transport] Timeout on call 0x00f0 (seq=2)` (GPU info fetch)
- `CUDA error: invalid value` at `ggml_backend_cuda_set_tensor_async` / `cudaMemcpyAsyncReserve`

**Root cause:** Under heavy model load, individual memcpy RPCs can time out. A single failure propagates as `cudaErrorInvalidValue` and aborts the run.

**Fix (libvgpu_cuda.c):**
- In `cuMemcpyHtoD_v2` and `cuMemcpyDtoH_v2`:
  - On transport failure (`rc != CUDA_SUCCESS`), destroy transport, call `ensure_connected()` to reconnect, then retry the memcpy once.

**Code location:** `libvgpu_cuda.c` around lines 5630 (cuMemcpyHtoD_v2) and 5697 (cuMemcpyDtoH_v2).

---

### 2.4 cuDevicePrimaryCtxRetain Dummy Context Fallback

**Symptom:** When transport is not ready, `cuDevicePrimaryCtxRetain` previously returned a dummy context `0xDEADBEEF`. CUBLAS init paths that use the export-table `get_primary_context` callback now call `cuDevicePrimaryCtxRetain` and `cuCtxSetCurrent`; dummy context can cause downstream `CUBLAS_STATUS_NOT_INITIALIZED`.

**Status:** Dummy context still used when RPC fails (e.g. timeouts). The dark-API `dark_get_primary_context` calls our real `cuDevicePrimaryCtxRetain`, which may return dummy context if transport times out. This is a known limitation; memcpy retry (above) aims to reduce timeouts.

---

## 3. libcublasLt.so.12 Dependency

**Finding:** `libcublas.so.12` has many undefined symbols (`cublasLt*`) that require `libcublasLt.so.12`. Strace shows loader probing paths like:
- `/usr/local/lib/ollama/cuda_v12/glibc-hwcaps/x86-64-v3/libcublasLt.so.12` (ENOENT)
- `/usr/local/lib/ollama/cuda_v12/x86_64/libcublasLt.so.12` (ENOENT)
- Eventually resolves via standard search (e.g. same dir as libcublas).

**Status:** If `libcublasLt.so.12` exists in Ollama’s CUDA dir and is loadable, this may be OK. If not, missing `libcublasLt` could contribute to `CUBLAS_STATUS_NOT_INITIALIZED`.

---

## 4. Host Mediator Requirement

**Critical:** The VM shims talk to the **host mediator** via BAR0/BAR1. If the host mediator is not running or not connected to this VM, all RPCs (INIT, GPU info, memcpy, etc.) will time out.

**User directive:** Focus on VM-side fixes only. The mediator must be running on the host for end-to-end tests; that is outside the “VM-only” scope.

---

## 5. Current Verification Approach

**Long generate run:**
```bash
curl -sS -m 2700 http://127.0.0.1:11434/api/generate \
  -d '{"model":"llama3.2:1b","prompt":"Say hello in one short sentence.","stream":false}'
```
- Timeout: 2700 s (45 min) to avoid 2-minute client timeout during model load.
- Success = JSON response with `response` field containing generated text.
- Failure = `error` in JSON (e.g. `llama runner process has terminated`, `CUDA error: ...`).

**VM-only checks (no host transfer):**
- `journalctl -u ollama -n 200`
- `python3 -c "import ctypes; lib=ctypes.CDLL('.../libcublas.so.12'); ... cublasCreate_v2(...)"` to probe CUBLAS init directly
- `strace -f` on Python cublasCreate probe to see syscall failures (e.g. libcublasLt open failures)

---

### 2.5 Hot-Path Logging Overhead (Model Load Slow)

**Symptom:** Model load (~545 MB) takes ~45+ min; excessive `syscall(__NR_write, ...)` during memcpy, cudaMalloc, cudaGetDeviceProperties, stream/event stubs.

**Fix (libvgpu_cudart.c):**
- All verbose logging gated behind `cudart_debug_logging()` (reads `VGPU_DEBUG` or `CUDART_DEBUG`).
- Removed unconditional logs from: `cudaGetDeviceProperties`, `cudaGetDeviceProperties_v2`, `patch_ggml_cuda_device_prop`, `cudaDriverGetVersion`, `cudaGetLastError`, `cudaMalloc`, `cudaMallocHost`, `cudaSetDevice`, `cudaMemGetInfo`, `cudaDeviceGetAttribute`, all stream/event/kernel stubs, `__cudaRegisterFatBinary*`, `cublasCreate_v2`, `__cudaPushCallConfiguration`, etc.
- Removed `/tmp/cudart_get_count_called.txt` file write from `cudaGetDeviceCount` (was open/write/close on every call).
- Without `VGPU_DEBUG`, hot path avoids ~2000+ syscalls during model load.

---

## 6. File Change Summary

| File | Changes |
|------|---------|
| `phase3/guest-shim/libvgpu_cuda.c` | `cuGetExportTable` impl, dark-API vtables, memcpy reconnect+retry, cuMemAlloc retry with ensure_connected, cuDeviceComputeCapability fallback; verbose logging already gated by vgpu_debug_logging() |
| `phase3/guest-shim/libvgpu_cudart.c` | `patch_ggml_cuda_device_prop` offsets, legacy major/minor at 0x15C/0x160; **all hot-path logging gated by cudart_debug_logging()**; removed cudaGetDeviceCount file write |
| `phase3/guest-shim/cuda_transport.c` | (prior) VGPU_ERR decoding; `poll_timeout_sec()` env override `CUDA_TRANSPORT_TIMEOUT_SEC`; verbose logging already gated by vgpu_debug_logging() |

---

## 7. CUBLAS Init Diagnostic (Mar 12)

On each `cublasCreate_v2` call, a consolidated diagnostic is written to `/tmp/vgpu_cublas_init_diag.txt`:
- `dlopen chosen path` — which candidate succeeded (or none)
- `dlopen success` — 1=ok, 0=fail
- `dlerror() (last failed candidate)` — reason for last dlopen failure
- `dlsym(cublasCreate_v2) result` — "ok", "skipped(stub)", or dlerror text
- `real cublasCreate_v2 return` — return code from real library (or -1 if not called)

Set `VGPU_DEBUG` or `CUBLAS_DEBUG` for init_real_cublas summary on stderr during model load.

---

## 8. Outstanding / Next Steps

1. **Compute capability 0.0 still in one log path:** Even with `legacy_major=9`, the line `Device 0: ... compute capability 0.0` can still appear if GGML reads from another offset or a different code path (e.g. `cuDeviceComputeCapability`). The latter uses `g_gpu_info`, which may be uninitialized when transport times out.
2. **Mediator must be running:** For full generate to succeed, host mediator must be up and connected. Timeouts on 0x0001, 0x00f0, 0x0090, 0x0030 indicate VM RPCs are not being served.
3. **Long load time:** Model load (tensor copies) can take tens of minutes; 45-minute client timeout is used to avoid premature 499.
4. **VM-side mitigations applied:**
   - `cuDeviceComputeCapability` defensive fallback (returns 9.0 when `g_gpu_info` has zeros).
   - Transport timeout override: `CUDA_TRANSPORT_TIMEOUT_SEC=120` when mediator is slow.
   - cuMemAlloc retry: reconnect via `ensure_connected()` then retry once on failure.

---

## 9. Call ID Reference (for transport timeouts)

When logs show `Timeout on call 0xXXXX`, use this mapping:

| Call ID | Name | Purpose |
|---------|------|---------|
| 0x0001 | CUDA_CALL_INIT | Initialize CUDA on host |
| 0x00f0 | CUDA_CALL_GET_GPU_INFO | Fetch GPU properties, fill `g_gpu_info` |
| 0x0030 | CUDA_CALL_MEM_ALLOC | `cudaMalloc` / `cuMemAlloc` |
| 0x0031 | CUDA_CALL_MEM_FREE | `cudaFree` |
| 0x0032 | CUDA_CALL_MEMCPY_HTOD | Host-to-device memcpy |
| 0x0033 | CUDA_CALL_MEMCPY_DTOH | Device-to-host memcpy |
| 0x0090 | CUDA_CALL_DEVICE_PRIMARY_CTX_RETAIN | Primary context retain |

Protocol header: `phase3/include/cuda_protocol.h` (or `phase3/GOAL/INCLUDE/cuda_protocol.h`).

---

## 10. Relevant Code Snippets

### cuGetExportTable UUID matching (libvgpu_cuda.c)

```c
static const unsigned char k_uuid_cudart_interface[16] = {
    0x6b, 0xd5, 0xfb, 0x6c, 0x5b, 0xf4, 0xe7, 0x4a,
    0x89, 0x87, 0xd9, 0x39, 0x12, 0xfd, 0x9d, 0xf9
};
static const unsigned char k_uuid_tools_runtime_hooks[16] = {
    0xa0, 0x94, 0x79, 0x8c, 0x2e, 0x74, 0x2e, 0x74,
    0x93, 0xf2, 0x08, 0x00, 0x20, 0x0c, 0x0a, 0x66
};
// ...
if (memcmp(pExportTableId, k_uuid_cudart_interface, 16) == 0)
    *ppExportTable = (const void *)g_cudart_interface;
else if (memcmp(pExportTableId, k_uuid_tools_runtime_hooks, 16) == 0)
    *ppExportTable = (const void *)g_tools_runtime_hooks;
```

### patch_ggml_cuda_device_prop offsets (libvgpu_cudart.c)

```c
// Patch only known-safe offsets:
// CUDA 12: 0x148/0x14C (computeCapabilityMajor/Minor)
// Legacy in our struct: 0x15C/0x160 (major/minor)
size_t offsets_major[] = {0x148, 0x15C};
size_t offsets_minor[] = {0x14C, 0x160};
for (size_t i = 0; i < 2; i++) {
    *(int32_t *)(ptr + offsets_major[i]) = major;
    *(int32_t *)(ptr + offsets_minor[i]) = minor;
}
```

### cuMemcpyHtoD retry (libvgpu_cuda.c)

```c
rc = cuda_transport_call(g_transport, CUDA_CALL_MEMCPY_HTOD, ...);
if (rc != CUDA_SUCCESS) {
    ensure_mutex_init();
    pthread_mutex_lock(&g_mutex);
    if (g_transport) { cuda_transport_destroy(g_transport); g_transport = NULL; }
    pthread_mutex_unlock(&g_mutex);
    reconnect_rc = ensure_connected();
    if (reconnect_rc == CUDA_SUCCESS)
        rc = cuda_transport_call(g_transport, CUDA_CALL_MEMCPY_HTOD, ...);
}
```

---

*This document is the authoritative record of the current VM-side debugging work and should be updated when new findings or fixes are applied.*
