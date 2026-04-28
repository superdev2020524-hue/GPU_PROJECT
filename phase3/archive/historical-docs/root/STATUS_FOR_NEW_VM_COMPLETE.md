# Phase3 / Ollama vGPU — complete status for continuing on a new VM

**Date:** March 2026  
**Context:** VM TEST-4 (10.25.33.12) became unreachable after running `transfer_libvgpu_cuda.py`. This document preserves the full state so you can continue on a **new VM** (or after TEST-4 is recovered).

---

## 1. Goal

- **Ollama** in the VM runs in **GPU mode** using the **vGPU shim**: CUDA/CUBLAS calls are sent to the **host mediator**, which runs them on the physical **H100** and returns results.
- **First-stage success:** One inference (e.g. `ollama run llama3.2:1b Hi`) completes without "exit status 2" or "CUDA error".

---

## 2. Roles and permissions (as agreed)

| Where        | Who does it        | Permissions |
|-------------|--------------------|------------|
| **VM**      | Assistant / you    | Full: SSH, install, deploy shims, run Ollama, trigger generate, read logs. |
| **Host**    | You only           | Read-only for assistant (e.g. `/tmp/mediator.log`). No copy/build on host by assistant; you apply host fixes. |

---

## 3. Environment (last known)

- **VM (TEST-4):**
  - Address: **10.25.33.12**
  - User: **test-4** (hyphen, not underscore)
  - Password: **Calvin@123**
  - Config: `phase3/vm_config.py` — `VM_USER`, `VM_HOST`, `VM_PASSWORD`, `REMOTE_PHASE3` (e.g. `/home/test-4/phase3`).
- **Mediator host:**
  - Address: **10.25.33.10**
  - User: **root**
  - Password: (same as VM in vm_config)
  - Log: **/tmp/mediator.log** (read via `connect_host.py`).

---

## 4. What was working (before VM loss)

- **Ollama discovery:** VM reported GPU (e.g. NVIDIA H100 80GB, CUDA, 80 GiB) when symlinks and patches were in place (see section 7).
- **Host mediator:** Restarted with GPU free; allocations for vm=9 (TEST-4) succeeded (no OOM).
- **Model load:** HtoD and `cuMemAlloc` on host succeeded for the run we observed.
- **CUBLAS path:** Guest log showed `cublas_create`, `cublas_rpc_rc=0 num_results=2`, `set_stream`, `set_stream_done`, `gemm_ex`, `gemm_ex_before_send`, `gemm_ex tc_rc=0 num_results=1 status=0`, `gemm_ex_return` — i.e. remote CUBLAS create, set stream, and GEMM all succeeded.
- **Runner:** Still exited with **exit status 2**; no `launch_kernel` or `ctx_sync` in `/tmp/vgpu_next_call.log` — so crash is **after** last GEMM return, **before** `cuLaunchKernel` or `cuCtxSynchronize`.

---

## 5. What was failing (last known)

- **Symptom:** API returns `{"error":"llama runner process has terminated: exit status 2"}` (or "CUDA error").
- **Location:** After successful GEMM returns; before the first `cuLaunchKernel` or `cuCtxSynchronize` from the runner (or before another CUDA call we had not yet logged, e.g. `cuMemcpyDtoH` / `cuMemcpyDtoD`).
- **Host:** No OOM in the run after mediator restart; no `MAPPING FAILED` in the tail we checked.

---

## 6. Guest shims and layout (VM)

- **Libraries (installed under /opt/vgpu/lib):**
  - **libvgpu-cuda.so.1** — built from `libvgpu_cuda.c` + `cuda_transport.c` (CUDA Driver API shim).
  - **libvgpu-cudart.so** (if used) — CUDA Runtime shim.
  - **libvgpu-cublas.so.12** — CUBLAS shim (from `libvgpu_cublas.c` + `cuda_transport.c`).
  - **libvgpu-cublasLt.so.12** (if present).
  - Symlinks for discovery: **libcublas.so.12** → libvgpu-cublas.so.12, **libcublasLt.so.12** → libvgpu-cublasLt.so.12; **libggml-cuda-v12.so** (or similar) so Ollama loads GPU GGML.
- **Ollama:** Uses `LD_LIBRARY_PATH` including `/opt/vgpu/lib` so the runner loads the vGPU shims. Patches (device/server/discover) were applied so discovery sees the GPU and runner gets the right environment.
- **Debug log:** `/tmp/vgpu_next_call.log` on the VM — append-only log of which shim entry points were hit (e.g. `cublas_create`, `set_stream`, `gemm_ex`, `gemm_ex_return`, `launch_kernel`, `ctx_sync`, `memcpy_dtoh`, `memcpy_dtod`).

---

## 7. Key files and scripts (phase3 tree)

| File / dir              | Purpose |
|-------------------------|--------|
| **vm_config.py**        | VM_HOST, VM_USER, VM_PASSWORD, REMOTE_PHASE3, MEDIATOR_* . Update for new VM. |
| **connect_vm.py**       | SSH to VM and run a single command (used by transfer and diagnostic scripts). |
| **connect_host.py**     | SSH to mediator host, run command (e.g. `tail /tmp/mediator.log`). Read-only. |
| **transfer_libvgpu_cuda.py** | Transfer **libvgpu_cuda.c** in chunks, build on VM, install libvgpu-cuda.so.1, restart ollama. **Risk:** Build uses a lot of RAM; can trigger OOM and VM loss. See **TRANSFER_LIBVGPU_CUDA_SCRIPT_INVESTIGATION.md**. |
| **transfer_libvgpu_cublas.py** | Transfer libvgpu_cublas.c (and cuda_protocol.h), build, install libvgpu-cublas.so.12, symlink, restart ollama. |
| **transfer_cuda_transport.py** | Transfer cuda_transport.c/h, rebuild libvgpu-cuda.so.1, install, restart ollama. |
| **guest-shim/libvgpu_cuda.c** | CUDA Driver API shim (~9.6k lines). Includes entry-point logs for launch_kernel, ctx_sync, memcpy_dtoh, memcpy_dtod. |
| **guest-shim/libvgpu_cublas.c** | CUBLAS shim; logs cublas_create, set_stream, gemm_ex, gemm_ex_return, etc. |
| **guest-shim/cuda_transport.c** | Shared transport (BAR0, doorbell, chunked payloads). |
| **HOST_LOG_FINDINGS.md**       | Host log interpretation, OOM, GEMM path, 448 MiB vs 1.2 GB. |
| **OLLAMA_VGPU_REVISIONS_STATUS.md** | Host/guest revisions, diagnostics, next steps. |
| **capture_runner_stderr.py**  | Run `ollama run llama3.2:1b Hi` on VM to try to capture runner stderr. |

---

## 8. Host mediator (you)

- **Start:** `nohup ./mediator_phase3 >> /tmp/mediator.log 2>&1 &`
- **Stop (free GPU):** `kill $(pgrep -f mediator_phase3)`
- **Check GPU:** `nvidia-smi`
- **Log:** `/tmp/mediator.log` — grep for `vm=9`, `cuMemAlloc FAILED`, `cublasGemmEx`, `MAPPING FAILED`.

---

## 9. New VM setup (checklist)

When you have a new VM (or TEST-4 back):

1. **Update vm_config.py** with the new VM_HOST (and VM_USER / VM_PASSWORD if different).
2. **On the new VM:**
   - Install: gcc, make, curl, (optional) ollama, and the phase3 tree (e.g. copy phase3 or clone).
   - Create `/opt/vgpu/lib`, put shims there, set LD_LIBRARY_PATH for Ollama (and symlinks for libcublas, libggml-cuda as used on TEST-4).
   - Apply the same Ollama patches (device/server/discover) if you need GPU discovery and runner env.
3. **Avoid OOM on deploy:** Prefer **building libvgpu-cuda.so.1 on a machine with more RAM** and copying only the `.so` to the VM (see **TRANSFER_LIBVGPU_CUDA_SCRIPT_INVESTIGATION.md**). If you must run `transfer_libvgpu_cuda.py` on the VM, ensure the VM has at least 2–4 GB RAM and free `/tmp`.
4. **Mediator:** Start on host with free GPU before testing.

---

## 10. Next steps (after new VM is ready)

1. Deploy the **current** guest shims (including libvgpu_cuda with `memcpy_dtoh` / `memcpy_dtod` entry logs). Prefer copying a pre-built **libvgpu-cuda.so.1** to avoid heavy gcc on the VM.
2. Trigger one generate (e.g. `curl .../api/generate` with model llama3.2:1b, prompt "Hi").
3. On the VM, read **/tmp/vgpu_next_call.log**: see which of `memcpy_dtoh`, `memcpy_dtod`, `launch_kernel`, `ctx_sync` appear. The **last** line that appears is the last API reached before the crash.
4. If you get a successful response, run a short inference and confirm no exit status 2.
5. On the host, if failure continues, check `/tmp/mediator.log` for `cublasGemmEx`, `MAPPING FAILED`, and any new errors.

---

## 11. Incident: VM loss after transfer_libvgpu_cuda.py

- **Observed:** After running `transfer_libvgpu_cuda.py`, TEST-4 (10.25.33.12) stopped accepting SSH (from the assistant environment and from your PC).
- **Likely cause:** The script runs **gcc** on the VM to build **libvgpu_cuda.c** (very large file). That can use a lot of RAM; on a small VM, **OOM** can kill sshd or make the system unresponsive. See **TRANSFER_LIBVGPU_CUDA_SCRIPT_INVESTIGATION.md** for a full breakdown and safer options (e.g. build .so elsewhere, copy only the .so to the VM).

---

## 12. References

- **TRANSFER_LIBVGPU_CUDA_SCRIPT_INVESTIGATION.md** — What the script does and why it can cause VM loss.
- **HOST_LOG_FINDINGS.md** — Host log interpretation, OOM, GEMM, 448 MiB vs 1.2 GB.
- **OLLAMA_VGPU_REVISIONS_STATUS.md** — Revisions and diagnostic history.
- **HOST_FIX_MODULE_LOAD_PRIMARY_CTX.md** — Host-side module load fix (you apply on host).
- **GPU_MODE_DO_NOT_BREAK.md** — How to keep GPU mode (e.g. symlinks, what to remove if discovery breaks).
