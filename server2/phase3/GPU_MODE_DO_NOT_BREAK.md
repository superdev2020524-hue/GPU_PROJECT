# PHASE3: GPU mode — how it works and how not to break it

This document describes the **exact** conditions under which Ollama runs in GPU mode on the vGPU guest (VM) and what must not be changed so it keeps working.

---

## What “GPU mode” means here

- **Discovery:** At startup, Ollama discovers one or more CUDA devices (e.g. `inference compute ... library=CUDA ... "NVIDIA H100 80GB HBM3"`, `initial_count=2`).
- **Backend:** The server selects the CUDA backend for inference (not CPU).
- **Inference:** Model load and generate use the CUDA backend; allocation and compute go through the shim → VGPU-STUB → mediator → host GPU.

---

## Why it breaks: what we fixed (Mar 5, 2026)

### 1. CUBLAS / CUBLASLt shim in `/opt/vgpu/lib`

- **Symptom:** Discovery reports `id=cpu library=cpu`, `total_vram="0 B"`, `initial_count=0`.
- **Cause:** With `LD_LIBRARY_PATH=/opt/vgpu/lib:...`, the runner loads **our** CUBLAS or CUBLASLt shim instead of the real libraries from `cuda_v12`. The shim returns dummy handles; GGML/CUDA init then sees no valid GPU.
- **Fix (do not revert):** Do **not** install the CUBLAS or CUBLASLt shim in `/opt/vgpu/lib`:
  ```bash
  sudo rm -f /opt/vgpu/lib/libcublas.so.12 /opt/vgpu/lib/libcublasLt.so.12
  ```
- **Keep:** Only these in `/opt/vgpu/lib`: `libcuda.so.1` → shim, `libcudart.so.12` → shim, `libnvidia-ml.so.1` → shim. CUBLAS/CUBLASLt must resolve to the **real** libraries in `cuda_v12`.

### 2. Library path order (service + runner)

- **Symptom:** Runner loads CPU backend first or does not see CUDA; discovery shows CPU.
- **Cause:** If `cuda_v12` is **after** `/usr/local/lib/ollama` in `LD_LIBRARY_PATH` or `OLLAMA_LIBRARY_PATH`, the loader can pick the CPU backend or wrong libs first.
- **Fix (do not revert):** In the systemd override, paths must have **cuda_v12 before** the parent ollama path:
  - `LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama`
  - `OLLAMA_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama`
- **No** `LD_PRELOAD` in the service override (runner must use normal `dlopen` so it can load libggml-cuda and the shim correctly).

### 3. Hopper (sm_90) libggml-cuda

- **Symptom:** Discovery or model load fails / times out; GPU not used.
- **Cause:** Bundled `libggml-cuda.so` is not built with Hopper (sm_90); on an H100 the backend does not recognize the device.
- **Fix (do not revert):** Use a Hopper-built `libggml-cuda.so` in `/usr/local/lib/ollama/cuda_v12/` (see `BUILD_LIBGGML_CUDA_HOPPER.md`, `deploy_libggml_cuda_hopper.py`).

### 4. Ollama binary patches (runner env and discovery)

- **Symptom:** Runner does not get the right lib paths or CUDA devices are filtered out.
- **Fix (do not revert):** Use a patched `ollama.bin`:
  - **ml/device.go:** `NeedsInitValidation()` returns false for CUDA (`return d.Library == "ROCm"` only).
  - **llm/server.go:** Prepend `/opt/vgpu/lib` to runner `LD_LIBRARY_PATH` and `OLLAMA_LIBRARY_PATH`; strip `LD_PRELOAD` from runner `cmd.Env`.
  - **discover/runner.go:** `dirs = []string{dir, ml.LibOllamaPath}` so the GPU lib dir (e.g. cuda_v12) is first when the backend loader runs.

Apply via `transfer_ollama_go_patches.py` (or equivalent) and reinstall the binary after any change.

---

## Single checklist: “Do not break GPU mode”

Before changing the VM or guest stack, ensure:

| # | Check | Command / action |
|---|--------|-------------------|
| 1 | No CUBLAS/CUBLASLt shim in `/opt/vgpu/lib` | `ls /opt/vgpu/lib/libcublas* 2>/dev/null` → should be empty |
| 2 | Service path order: cuda_v12 before parent | `grep -E 'LD_LIBRARY_PATH|OLLAMA_LIBRARY_PATH' /etc/systemd/system/ollama.service.d/vgpu.conf` → must contain `cuda_v12` before `ollama` (no trailing path with only `ollama` before `cuda_v12`) |
| 3 | No LD_PRELOAD in service | `grep LD_PRELOAD /etc/systemd/system/ollama.service.d/vgpu.conf` → should have no match |
| 4 | Hopper lib in cuda_v12 | `ls -la /usr/local/lib/ollama/cuda_v12/libggml-cuda.so` → exists; symlink `libggml-cuda-v12.so` → `libggml-cuda.so` |
| 5 | Patched ollama.bin in use | Binary built from source with device.go, server.go, discover/runner.go patches above |
| 6 | cuda_v12 symlinks for shim | `libcuda.so.1`, `libcudart.so.12` in cuda_v12 point to `/opt/vgpu/lib` shims |

After any change to the above, restart Ollama and re-check discovery:

```bash
sudo systemctl restart ollama
sudo journalctl -u ollama -n 25 --no-pager | grep -E "inference compute|initial_count"
# Expect: inference compute ... library=CUDA ... (not id=cpu library=cpu)
```

---

## Restore procedure (if GPU mode is broken)

Follow **RESTORE_GPU_LOGIC_CHECKLIST.md** in order. In short:

1. Apply the systemd override (`vgpu.conf`) with path order and no LD_PRELOAD.
2. Remove CUBLAS/CUBLASLt shims from `/opt/vgpu/lib`.
3. Deploy Hopper `libggml-cuda.so` to `cuda_v12` if needed.
4. Apply Ollama Go patches, rebuild `ollama.bin`, install (stop service → copy binary → start service).
5. Verify with the journalctl grep above.

---

---

## Inference: “unable to allocate CUDA0 buffer”

When GPU mode is working (discovery shows CUDA) but a generate fails with **“unable to allocate CUDA0 buffer”**, the failure is in the **allocation path**: guest shim → VGPU-STUB (BAR0) → **host mediator** → host CUDA executor → physical GPU.

- **Guest side:** The runner calls `cudaMalloc` (CUDART) or `cuMemAlloc_v2` (Driver API). The shim sends `CUDA_CALL_MEM_ALLOC` (0x0030) to the stub. The guest sees `RECEIVED from VGPU-STUB: status=ERROR` when the host returns an error.
- **Host side:** The mediator must be running and must call `cuda_executor_call()`. The executor then:
  1. Finds or creates VM state for `call->vm_id`.
  2. Calls `ensure_vm_context(exec, vm)` (creates CUDA context for that VM if needed).
  3. Calls `cuMemAlloc(&dptr, bytesize)` on the physical GPU.

If any of these fail, the executor sets `result->status` to the CUDA error and the stub reports ERROR to the guest.

**What to check on the host:**

1. **Mediator running:** The Phase 3 mediator (e.g. `mediator_phase3`) must be running and bound to the same socket the VGPU-STUB (QEMU) uses. Otherwise the stub logs “Cannot connect to mediator” and the guest sees ERROR.
2. **Mediator stderr:** Look for:
   - `[cuda-executor] cuMemAlloc FAILED: rc=<code> (vm=<id>)` — host `cuMemAlloc` failed (e.g. out of memory, or context/device issue).
   - `[cuda-executor] cuInit failed` / `No CUDA devices found` — host GPU or driver issue.
3. **Context creation:** `ensure_vm_context` calls `cuCtxCreate` per VM. If that fails (e.g. device in use, driver error), allocation will fail. **Fix (do not revert):** In `cuda_executor.c`, use the **primary context** for MEM_ALLOC and all memory ops (MEM_FREE, MEMCPY_HTOD/DTOH/DTOD, MEMSET_*, MEM_GET_INFO): call `cuCtxSetCurrent(exec->primary_ctx)` instead of `ensure_vm_context(exec, vm)`. Then rebuild the mediator on the host.
4. **VM id:** Guest and host must agree on `vm_id` (e.g. 13). The executor uses `find_or_create_vm(exec, vm_id)` so the VM is created on first use; no pre-registration needed.

**Quick guest-side check (with VGPU_DEBUG=1):** Restart ollama with `VGPU_DEBUG=1`, run a short generate, then inspect journalctl for “SENDING to VGPU-STUB”, “RECEIVED … status=ERROR”, and “cudaMalloc() ERROR: transport call failed”. That confirms the request reaches the stub and the host is returning an error.

---

## Related files

- **RESTORE_GPU_LOGIC_CHECKLIST.md** — Step-by-step restore and verify.
- **PHASE3_GPU_AND_TRANSPORT_STATUS.md** — Current status and inference/transport notes.
- **VM_TEST3_GPU_MODE_STATUS.md** — Historical context (CUBLAS regression, Hopper, patches).
- **BUILD_LIBGGML_CUDA_HOPPER.md** — Building and deploying the Hopper lib.
- **apply_vgpu_service_no_ldpreload.py** — Applies vgpu.conf (path order: cuda_v12 before parent).
