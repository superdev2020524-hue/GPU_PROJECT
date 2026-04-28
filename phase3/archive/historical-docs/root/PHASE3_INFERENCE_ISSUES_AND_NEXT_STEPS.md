# PHASE3: Issues That Could Cause Inference to Fail — Analysis and Next Steps

This document is based on a content-level review of PHASE3 documents and code. It lists **only** issues that could actually cause the current symptom (Ollama discovers GPU but inference never completes / times out), then proposes ordered next steps to resolve them.

---

## 1. Classification of Issues (Exact Causes)

### A. Transport and VGPU-STUB / mediator (blocking or failing)

| # | Issue | Source | How it could cause inference to fail |
|---|--------|--------|--------------------------------------|
| **A1** | **VGPU-STUB PCI device missing or wrong in guest** | VGPU_STUB_DEVICE_MISSING.txt, cuda_transport.c, cuda_transport.h | Shim looks for PCI device vendor 0x10DE / device 0x2331 (or legacy class 0x030200). If the VM does not have this device (QEMU not started with `-device vgpu-cuda,...`), `find_vgpu_device()` fails → `cuda_transport_init()` fails → `ensure_connected()` fails. First compute call (e.g. `cuMemAlloc`) then returns an error; GGML/runner could abort or hang. |
| **A2** | **Transport init (BAR0) fails** | cuda_transport.c | Even if the PCI device exists, opening `resource0` and mmap of BAR0 can fail (permissions, sandbox, or device not exposing BAR0). Same effect as A1: `ensure_connected()` fails, first real CUDA call fails. |
| **A3** | **Mediator not running or not responding** | cuda_transport.h, cuda_transport.c, HOST_SIDE_REQUIREMENTS.txt | Transport is **blocking RPC**: guest writes to BAR0 and polls until status = DONE/ERROR. If the host mediator is not running or does not respond, `cuda_transport_call()` blocks indefinitely → **inference hangs and times out** (matches current symptom). |
| **A4** | **ensure_connected() never called** | CRITICAL_FINDING_NO_TRANSPORT.md, VGPU_STUB_COMMUNICATION_STATUS.md | Older finding: no compute calls → transport never initialized. **Current state**: With Hopper lib, CUDA backend is used, so the first `cuMemAlloc` / `cuMemcpyHtoD` / `cuLaunchKernel` should call `ensure_connected()`. If that call then fails (A1/A2) or blocks (A3), that explains failure or timeout. |

### B. SHIM coverage (wrong result or hang)

| # | Issue | Source | How it could cause inference to fail |
|---|--------|--------|--------------------------------------|
| **B1** | **CUBLAS stubbed, not forwarded** | VGPU_STUB_COMMUNICATION_STATUS.md, libvgpu_cublas.c | CUBLAS APIs (e.g. `cublasCreate_v2`, `cublasSgemm`) return success with **dummy handles / no real work**. If GGML uses CUBLAS for matrix ops during inference, those ops do nothing on the GPU; execution can hang waiting for results or produce wrong state. |
| **B2** | **Generic stubs for unknown Driver API** | libvgpu_cuda.c (cuGetProcAddress, generic_stub_* ) | Unknown symbols get generic stubs that return success with dummy values. If GGML calls a Driver API we only stub, later use of that value in a real transport call could be invalid and cause mediator error or hang. |
| **B3** | **VMM / cuMemCreate path returns dummy** | libvgpu_cuda.c (cuMemCreate, cuMemAddressFree, etc.) | Documented: "dummy handle", "no-op for dummy addresses". If the backend uses these for allocation and then passes the handle to a forwarded call, the host may receive an invalid handle and fail or hang. |

### C. Environment / process (runner not reaching transport)

| # | Issue | Source | How it could cause inference to fail |
|---|--------|--------|--------------------------------------|
| **C1** | **Runner has no access to PCI / sys** | VM_TEST3_GPU_MODE_STATUS.md, cuda_transport.c | If the runner runs under systemd with `PrivateDevices=yes` or restricted `/sys`/`/dev`, `find_vgpu_device()` or opening `resource0` could fail even when the device exists. |
| **C2** | **First compute path never reached** | CRITICAL_INSIGHT_SHIM_VS_ACTUAL_GPU.md, INVESTIGATION_FINDINGS.md | If something before the first `cuMemAlloc`/`cuMemcpy`/`cuLaunchKernel` fails (e.g. context creation, module load), we might never call `ensure_connected()`. cuCtxCreate_v2/cuDevicePrimaryCtxRetain can return a dummy context when transport is not ready; if GGML then uses that context in a call that does use transport, behavior depends on whether that call triggers `ensure_connected()` and whether init then succeeds. |

### D. Already addressed (for reference)

| # | Issue | Status |
|---|--------|--------|
| D1 | Hopper (sm_90) missing in libggml-cuda.so | Addressed: Hopper lib built and deployed. |
| D2 | Runner not loading shim (LD_PRELOAD / LD_LIBRARY_PATH) | Addressed: runner gets LD_LIBRARY_PATH, no LD_PRELOAD; discovery shows GPU. |
| D3 | cuGetErrorString undefined | Addressed in shim. |
| D4 | Model load blob read interception breaking load | Documented as fixed in later work; if still present, could cause load to hang. |

---

## 2. Which Issues Are Most Likely for “Inference Timeout”

- **Timeout with no response** is most consistent with:
  - **A3** — Mediator not running or not responding → `cuda_transport_call()` blocks.
  - **A1 / A2** — No device or no BAR0 → first compute call fails; if that failure is not handled and leads to a retry or wait loop, could also look like a timeout.
  - **B1** — CUBLAS stubbed: if GGML waits on a CUBLAS result that never comes (stub returns immediately with no work), could hang or spin.

- **Less likely for “timeout” specifically** (would usually be crash or error):
  - B2, B3 (wrong handle/result → more likely crash or mediator error).
  - C1 (would typically be init failure, not indefinite hang).

---

## 3. Next Steps to Resolve (in order)

1. **Confirm VGPU-STUB and mediator (A1, A2, A3)**  
   - On the **guest VM**:  
     - Run: `lspci -nn | grep -E '10de:2331|0302'` and check for vGPU device.  
     - Run: `ls -la /sys/bus/pci/devices/*/resource0` for that BDF; confirm the process can open it (e.g. run as same user as ollama).  
   - On the **host**:  
     - Confirm mediator is running: `pgrep -a mediator_phase3`.  
     - Confirm the VM has a vgpu-cuda device in its QEMU command line (see VGPU_STUB_DEVICE_MISSING.txt).  
   - **If device is missing or mediator is down**: fix VM configuration and/or start mediator; then retry inference.

2. **Confirm whether transport is used and where it blocks (A4, A3)**  
   - On the VM, during a single inference attempt (e.g. `curl .../api/generate` in one terminal):  
     - Check **stderr / journal** for `[libvgpu-cuda] ensure_connected()` and `[cuda-transport] SENDING to VGPU-STUB` / `RINGING DOORBELL` / `RECEIVED from VGPU-STUB`.  
   - If **ensure_connected() appears** but **no RECEIVED**: transport is blocking in `cuda_transport_call()` → mediator not responding (A3).  
   - If **ensure_connected() FAILED** or **device not found**: A1 or A2.  
   - If **ensure_connected() never appears**: first compute path not reached (C2) or different code path; then trace which CUDA API is called first during model load.

3. **If transport works but inference still hangs: CUBLAS (B1)**  
   - Search journal/stderr for `[libvgpu-cublas]` during a generate.  
   - If CUBLAS calls appear and transport calls complete but inference never finishes, consider that GGML may be relying on CUBLAS results that are currently stubbed. Next step: either forward CUBLAS to the mediator or confirm that the run path does not depend on CUBLAS for the model used.

4. **Optional: Strace runner to see first blocking point**  
   - Start a generate; attach to the runner process (or run a minimal test that loads the model and runs one step); run `strace -f -e openat,open,read,poll,ppoll -p <runner_pid>`.  
   - If you see `poll`/`ppoll` on a file descriptor that corresponds to the MMIO/BAR0 wait, that supports A3 (blocking in transport).

5. **C1 (runner PCI/sys access)**  
   - Only if A1/A2 are ruled out (device and BAR0 exist and are usable from a normal shell). Then run the same `find_vgpu_device`/open logic as the runner (e.g. same user, same cgroup) or under the service unit to see if something blocks.

---

## 4. Summary

| Category | Exact issues that could cause inference to fail | Next step |
|----------|--------------------------------------------------|----------|
| **Transport / device** | VGPU-STUB PCI missing (A1), BAR0 init fail (A2), mediator not responding (A3), transport never started (A4) | Verify device and mediator; check logs for ensure_connected and SENDING/RECEIVED. |
| **SHIM coverage** | CUBLAS stubbed (B1), generic/VMM stubs (B2, B3) | After transport is confirmed, look for CUBLAS usage and stub-only paths. |
| **Environment** | Runner cannot open PCI/sys (C1), first compute path not reached (C2) | Verify under runner environment; trace first CUDA call during load. |

**Recommended order:** Do step 1 (VGPU-STUB + mediator), then step 2 (logs for ensure_connected and transport). That will distinguish “no device / no init” from “init OK but blocking in RPC,” and from “transport never used.”

## 5. VM check (Mar 5) — scope narrowed

On VM: VGPU-STUB present (10de:2331). BAR0 was root-only; ollama user could not open it (A2/C1). Fixed with chmod 666 and udev rule 99-vgpu-stub-resource0.rules. Inference still times out; next suspect: host mediator (A3).

## 6. Host mediator confirmed (Mar 5)

Mediator is running: one socket at `/var/xen/qemu/root-213/tmp/vgpu-mediator.sock`, 1 QEMU VM with vgpu-cuda. **Concrete check:** With mediator in foreground, trigger a generate from the VM; watch for `[SOCKET] New connection` or `[CONNECTION] New connection`. If none → guest not writing to BAR0 or stub not connecting. If connection appears → debug mediator request handling (CUDA_CALL_INIT, etc.).

## 7. VM diagnostics + ensure_connected marker (Mar 5)

During generate: no process had resource0 open (lsof); no "ollama.bin runner" process seen (pgrep over 8 s). **Shim change:** `libvgpu_cuda.c` writes `pid=<pid>` to `/tmp/vgpu_ensure_connected_called` when the runner reaches `ensure_connected()` just before `cuda_transport_init()`. **Deploy:** Use `transfer_libvgpu_cuda.py` (transfers only `guest-shim/libvgpu_cuda.c` to VM, build on VM, install to `/opt/vgpu/lib`) — avoids full phase3 SCP. **Result:** After generate, marker file **absent** → runner **never reaches ensure_connected()**. **Prior fix checked:** cuda_v12 symlinks (ROOT_CAUSE_RUNNER_SUBPROCESSES.md) still point to /opt/vgpu/lib; runner env/symlinks were not regressed. **Regression:** Installing our CUBLAS shim at /opt/vgpu/lib/libcublas.so.12 caused discovery to report device_count=0 (our CUBLAS returns dummies; GGML init then sees no GPU). **Immediate fix:** `sudo rm -f /opt/vgpu/lib/libcublas.so.12 && sudo systemctl restart ollama` on the VM to restore discovery. **Update (Mar 5):** After removing CUBLAS shim, discovery still shows device_count=0. Added `libggml-cuda-v12.so` symlinks so GGML loader finds the CUDA backend; LD_DEBUG shows the **child** (runner subprocess) loads libcuda/libcudart but our shim’s device-count APIs are never called—likely the child does not inherit `LD_LIBRARY_PATH` with `/opt/vgpu/lib` first. **Next:** Confirm child’s env (e.g. `/proc/<child_pid>/environ` during discovery); if missing `/opt/vgpu/lib` first, ensure server/runner passes it to the child. With discovery restored, re-check ensure_connected marker and inference.

---

## 8. Quick path to get inference working (Mar 5)

State is confirmed: GPU discovery works when runner gets `LD_LIBRARY_PATH=/opt/vgpu/lib:...` and no LD_PRELOAD, Hopper lib is deployed, and no CUBLAS shim is in `/opt/vgpu/lib`. To get inference completing:

1. **Apply vGPU patches to Ollama** so the runner consistently gets the right env and CUDA devices are not filtered: run `python3 apply_ollama_vgpu_patches.py` from phase3 (see VM_TEST3_GPU_MODE_STATUS.md "Quick path to GPU inference"). This patches `ml/device.go` (NeedsInitValidation) and `llm/server.go` (prepend `/opt/vgpu/lib`, strip LD_PRELOAD for runner).
2. **Rebuild ollama.bin** and install on the VM; restart ollama.
3. **Verify VM**: no CUBLAS shim in `/opt/vgpu/lib`, BAR0 permissions, Hopper lib + symlinks.
4. **Run a short generate**; if it still times out, check `/tmp/vgpu_ensure_connected_called` and mediator logs (Section 6).

