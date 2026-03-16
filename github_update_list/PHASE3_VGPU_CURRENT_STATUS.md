# Phase 3 vGPU Ollama — current status

*Last updated: Mar 15, 2026*

## Summary

- **Transport path:** Guest → shim → VGPU-STUB → mediator → physical GPU is **working**. Allocations and HtoD copies succeed; model load over this path is **slow** (15–40+ min for ~1.3 GB) because every byte goes over the remoting pipe.
- **cublasGemmEx remoting (Mar 15):** Host-side `cublasGemmEx` now succeeds (rc=0). Added `CUDA_CALL_CUBLAS_GEMM_EX` handler in `cuda_executor.c`; guest shim in `libvgpu_cublas.c` remotes the call. Mediator log shows all GEMM calls returning success; no MAPPING FAILED, no Unsupported calls.
- **VM ID mapping:** **test-4 = vm_id=9** (vm_id=8 is another VM, e.g. test-3). When checking mediator logs for test-4, grep for `vm=9` or `vm_id=9`.
- **GPU discovery fixed (test-4, Mar 15):** (1) No CUBLAS/CUBLASLt shim in `/opt/vgpu/lib`. (2) `libggml-cuda-v12.so` symlinks in cuda_v12 and parent. (3) Patched **ollama.bin** (device.go, server.go, discover/runner.go) and service `ExecStart=/usr/local/bin/ollama.bin serve`. Discovery now shows library=CUDA, "NVIDIA H100 80GB HBM3", total="80.0 GiB".
- **"unable to allocate CUDA0 buffer" (test-4) — FIXED (Mar 15):** The first `cuMemAlloc` on the host was using per-VM context (`ensure_vm_context` → `cuCtxCreate`), which could fail. **Fix applied:** Use **primary context** for allocation and memory ops in `cuda_executor.c`. Deployed via `deploy_cuda_executor_to_host.py` (chunked over connect_host), then host rebuild and mediator restart. Mediator log for vm=9 now shows `cuMemAlloc SUCCESS` (1.3 GB, 4 GB, 419 MB) and HtoD progress to ~1250 MB.
- **Current failure (test-4, vm_id=9):** The mediator log for vm=9 shows the GPU path **failing at module load**: `cuModuleLoadFatBinary` returns **CUDA_ERROR_INVALID_IMAGE** (rc=200, "device kernel image is invalid"). Sequence: vm=9 does CUDA_CALL_INIT, cuMemAlloc, HtoD progress to ~1250 MB, further allocs (4 GB, ~400 MB), then module-chunk + module-load; module-load fails with INVALID_IMAGE; then "Cleaned up VM 9". So test-4’s GPU path fails before any CUBLAS inference; Ollama then **falls back to CPU** and the generate completes with HTTP 200. The successful response is therefore **CPU mode**, not vGPU.
- **CUBLAS fix (primary context):** Using the **primary context** for all CUBLAS calls in `cuda_executor.c` fixes `cublasCreate_v2` ALLOC_FAILED for VMs that get past module load. That fix is deployed but does not help test-4 until the module-load failure is fixed.
- **Real blocker for test-4 GPU:** Fix **CUDA_ERROR_INVALID_IMAGE** on host `cuModuleLoadFatBinary`: the **bundled** `libggml-cuda.so` on the VM is **not** built with Hopper (sm_90), so embedded kernels have no image for H100. **Fix (see BUILD_LIBGGML_CUDA_HOPPER.md):** Build `libggml-cuda.so` with `CMAKE_CUDA_ARCHITECTURES=90` (on a host with CUDA toolkit or via Docker: `./build_libggml_cuda_hopper_docker.sh`), then deploy to the VM with `python3 deploy_libggml_cuda_hopper.py /path/to/libggml-cuda.so`. The script uses `vm_config.py` (test-4). After deploy, restart Ollama and re-test generate.
- **Previous failure (before Mar 6 unified-memory fix):** A full deploy using the patient client (~27m 58s on test-3) finished copying the model (mediator showed HtoD progress up to ~1250 MB), then the **llama runner process terminated** with `exit status 2` and Ollama returned **HTTP 500**.
- **Current state (after unified-memory fix):** `cuMemCreate` / `cuMemMap` / `cuMemRelease` in `libvgpu_cuda.c` now back unified memory with **real GPU allocations** via `cuMemAlloc_v2` / `cuMemFree_v2`, and the updated shim is deployed to test-3 with Ollama restarted. **Mediator log (Mar 6):** No `cuMemcpyHtoD FAILED` or `cuMemcpyDtoH FAILED` — all host-side copies and the two large post-load allocs (4 GB + ~1.3 GB) succeeded; crash is guest-side. **Guest fix applied:** Transport round-trips are now serialized with a mutex in `cuda_transport.c` so one thread cannot read another’s result from BAR0; `cudaGetErrorString` returns real error strings (e.g. "invalid value") instead of always "no error". **Deployed (Mar 7):** Used `transfer_guest_shim_transport_cudart.py` (same pattern as `transfer_libvgpu_cuda.py`: `connect_vm.py` + chunked base64 + SHA256). Only **cuda_transport.c** and **libvgpu_cudart.c** are transferred; build on VM, then install both .so to `/opt/vgpu/lib` and restart ollama. Do **not** copy unneeded files (e.g. full phase3 or libvgpu_cuda.c for this fix). Re-test with the patient client.
- **Mar 7 re-test:** After deploying transport mutex + cudaGetErrorString, long generate still failed (~32m 47s, same HTTP 500 / exit status 2). Mediator log again shows HtoD to 1250 MB, then three cuMemAlloc (4 GB, ~1.3 GB, 16 MB) all SUCCESS, then only heartbeats—no FAILED lines. So the mutex did not fix the crash; the failure is still guest-side. **Next:** On the VM, right after a crash run `sudo journalctl -u ollama -n 200 --no-pager` and search for the runner’s last stderr (e.g. "CUDA error: ..."). With the new cudaGetErrorString we may now see a real string like "invalid value" instead of "no error", which would confirm which error code the shim is returning.
- **Design:** Deploy once (first load), then use. After a successful load, the model should stay on the host GPU and inference should be normal until the runner/mediator disconnects.
- **Client requirement:** Use a **patient client** with **no time limit** and a **progress bar** so users don’t think it’s stuck. Standard curl/CLI still time out too early and will abort the load.

---

## Server (VM) configuration

- **Ollama** with vGPU: `LD_LIBRARY_PATH` includes `/opt/vgpu/lib` and CUDA libs; no LD_PRELOAD for the main process (runner loads shims via dlopen).
- **Load timeout:** `OLLAMA_LOAD_TIMEOUT=20m` in the Ollama vGPU service drop-in so the server does not abort the load early.
- **VM:** Target from `vm_config.py` (e.g. `test-4@10.25.33.12`). New VM setup: see `SETUP_TEST4_CHECKLIST.md`.

---

## Patient client script

| Item | Value |
|------|--------|
| **Script** | `phase3/ollama_vgpu_generate.py` |
| **On VM** | `/tmp/ollama_vgpu_generate.py` (copy via SCP when updated) |
| **Request timeout** | None (7-day timeout; effectively waits until done or Ctrl+C) |
| **Progress** | Bar + estimated % over **40 minutes** (not real server progress; time-based estimate) |
| **Cancel** | Only Ctrl+C |

**Usage on VM:**
```bash
python3 /tmp/ollama_vgpu_generate.py llama3.2:1b "Say hello."
# Or with custom prompt:
python3 /tmp/ollama_vgpu_generate.py llama3.2:1b "Your prompt"
```

**Copy updated script to VM (from host with phase3):**
```bash
# Use VM from vm_config.py (e.g. test-4@10.25.33.12)
scp -o StrictHostKeyChecking=no phase3/ollama_vgpu_generate.py test-4@10.25.33.12:/tmp/
```

---

## Quick reference — commands on the VM

1. **Run generate (patient client, no time limit):**
   ```bash
   python3 /tmp/ollama_vgpu_generate.py llama3.2:1b "Hi"
   ```

2. **Check Ollama and GPU path:**
   ```bash
   systemctl is-active ollama
   sudo journalctl -u ollama -n 30 --no-pager | grep -E "library=|cuda|GPU|listening"
   ```

3. **List models:**
   ```bash
   curl -s http://localhost:11434/api/tags
   ```

4. **Run generate in background and follow log:**
   ```bash
   nohup python3 /tmp/ollama_vgpu_generate.py llama3.2:1b "Hi" > /tmp/vgpu_gen.log 2>&1 &
   tail -f /tmp/vgpu_gen.log
   ```

---

## Host (mediator)

- **Mediator** logs HtoD progress every 10 MB (e.g. `HtoD progress: 650 MB total (vm=13)`). This is only on the host; the VM client does not see real copy percentage (Ollama does not expose load progress over HTTP).
- Rebuild mediator after changing `phase3/src/cuda_executor.c`: on host, `make mediator_phase3` (see `transfer_cuda_executor_to_host.py` for transfer).
- **Failure logging:** Any `cuMemcpyHtoD` or `cuMemcpyDtoH` failure on the host is logged with `FAILED: rc=... dst=0x... size=...` (HtoD) or `FAILED: rc=... src=0x... size=...` (DtoH). Capture mediator stderr when reproducing the crash to see the exact CUDA error code and address/size.

### Host debug steps (after runner crash with "CUDA error: no error")

1. **Transfer updated executor and rebuild (from machine with phase3):**
   ```bash
   export MEDIATOR_HOST=10.25.33.10   # or your mediator host
   export REMOTE_PHASE3_HOST=/root/phase3
   python3 phase3/transfer_cuda_executor_to_host.py
   ```
2. **On the mediator host:**
   ```bash
   cd /root/phase3   # or REMOTE_PHASE3_HOST
   make
   # Stop current mediator, then run with stderr captured:
   ./mediator_phase3 ... 2>&1 | tee /tmp/mediator.log
   ```
3. **Reproduce:** On the VM, run the patient generate (e.g. `python3 /tmp/ollama_vgpu_generate.py llama3.2:1b "Hi"`). When the runner crashes, check mediator log for lines containing `cuMemcpyHtoD FAILED` or `cuMemcpyDtoH FAILED` (with `rc=`, `dst=`/`src=`, `size=`). That gives the host-side CUDA error and the failing copy parameters.

---

## Related docs

- **Direction (goals):** `VGPU_CLIENT_DEPLOYMENT_DIRECTION.md`
- **What was verified:** `VM_INFERENCE_VERIFICATION.md`
- **Transport/GPU status:** `PHASE3_GPU_AND_TRANSPORT_STATUS.md`
