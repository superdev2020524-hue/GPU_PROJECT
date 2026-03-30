# Current state (test-4) and direction

*Document created: Mar 15, 2026*

---

## 1. Current situation (test-4)

### Is Ollama already operating in GPU mode?

**Yes.** On test-4, Ollama is using the **CUDA/GPU path** end-to-end in the following sense:

- **Discovery:** Ollama reports the vGPU as a CUDA device (library=CUDA, "NVIDIA H100 80GB HBM3", total="80.0 GiB").
- **Runner:** The inference runner loads `libggml-cuda.so` and uses the CUDA backend. All CUDA API calls (Driver, Runtime, and the memory/kernel path) go through the **guest shims** (e.g. `libvgpu-cuda.so`, `libvgpu-cudart.so` in `/opt/vgpu/lib`), which intercept those calls.
- **Transport:** Intercepted calls and data are sent over the **VGPU-STUB** (MMIO/chunked transport) to the **host mediator**.
- **Host execution:** The mediator’s **cuda_executor** replays the calls on the **physical H100** using the host’s CUDA stack (Driver/Runtime). Allocations, host→device copies, and (once module load succeeds) kernel/CUBLAS execution happen on the real GPU.
- **Return path:** Results (return codes, device pointers, and later DtoH data) are sent back through the mediator to the stub and shims, and the guest process sees them as normal CUDA API results.

So the **pipeline is correct**: computations and data required for Ollama’s GPU operation are intercepted during CUDA calls/transmissions, processed via the VGPU path, passed through the host mediator, and executed on the actual H100 using the host’s CUDA, with results returned to the VM.

### Where it fails today

The pipeline **fails at one point**: loading the CUDA module (fat binary) on the host.

- The guest sends the fat binary (from the VM’s `libggml-cuda.so`) to the host via the existing module-chunk/module-load protocol.
- The host calls `cuModuleLoadFatBinary` on that binary.
- The host CUDA driver returns **CUDA_ERROR_INVALID_IMAGE** (rc=200, "device kernel image is invalid") because the binary does **not** contain a kernel image for **sm_90** (Hopper / H100). The default Ollama build (and the library currently on the VM) is built for older architectures (e.g. 50, 61, 70, 75, 80, 86, 89) and does not include 90.

Up to that point, mediator logs for **vm=9** show:

- `CUDA_CALL_INIT`, `cuMemAlloc` (1.3 GB, 4 GB, ~419 MB) all **SUCCESS**
- HtoD progress up to **~2.5 GB**
- Module chunks received; then **module-load done rc=200 INVALID_IMAGE**

So: **Ollama is already operating in GPU mode**; the only blocker is that the **CUDA kernel binary** loaded on the host must include Hopper (sm_90). The fix is to use a `libggml-cuda.so` built with `CMAKE_CUDA_ARCHITECTURES=90` and deploy that file to the VM (see BUILD_LIBGGML_CUDA_HOPPER.md).

### Timing: bulk model transfer and when errors occur

**test-3 (before it was destroyed):**  
The bulk model transfer to the host took **over 30 minutes** (e.g. ~27m 58s, ~32m 47s with the patient client). The mediator showed **HtoD progress up to ~1250 MB** (model data). When the host had finished that transfer and the follow-up allocations (4 GB, ~1.3 GB, 16 MB — all SUCCESS), the **llama runner then terminated with exit status 2** and Ollama returned **HTTP 500** immediately after. So: long transfer → host indicates transfer complete → **500 and exit 2 right after**.

**test-4 (current):**  
The **bulk model transfer is not quick**. HtoD still runs to **~2.5 GB** (model weights) and can take a long time, consistent with test-3. The error occurs **after** that transfer is complete. The **next** step is loading the **CUDA module** (fat binary of compiled kernels) on the host via `cuModuleLoadFatBinary`. That step fails with INVALID_IMAGE. So the failure is **not** “while the host is reading the model” (the model weight data is already transferred by then). It is **after** the transfer, when the host is **loading the GPU kernel module** (the fat binary). Summary:

- **Bulk model transfer (HtoD):** Still long (e.g. 30+ min to ~1.2–2.5 GB); on test-4 we see it complete (HtoD progress to ~2.5 GB).
- **After transfer:** Host then loads the CUDA **module** (kernel code). On test-4 that **module load** fails (INVALID_IMAGE). On test-3 the failure was runner exit 2 / 500 at the next step (e.g. CUBLAS init) after transfer and allocs.

So the report is **not** saying transfer is quick. It is saying: transfer completes (after a long time), and the error happens **immediately after**, at the **next** step — on test-4 that step is **module load** (kernel binary), not “reading” the model weights.

---

## 2. Correct direction from now on

### Permissions

**See ASSISTANT_PERMISSIONS.md for the authoritative statement.**

- **VM (test-4):** **Full authority** — run commands, configure, deploy, edit VM files, read logs, rebuild and install (e.g. ollama.bin, guest shims), restart services.
- **Host:** **Read only** — check host logs and read file contents for investigation; **no editing** of host files, no copy/build/make/restart on the host.

### Implications

- **Host-side fixes** (e.g. changes to `cuda_executor.c`, mediator rebuild, mediator restart) are **not** done by the assistant. They are documented and left for you to apply on the host (transfer updated sources, `make`, restart mediator).
- **VM-side work** is done by the assistant: checking Ollama, triggering generates, inspecting journalctl, deploying or updating guest-side components (e.g. Hopper-built `libggml-cuda.so` onto the VM), and reading host logs to confirm behavior (e.g. vm=9, SUCCESS/FAILED, module-load rc=…).

---

## 3. How similar issues were handled in PHASE 3 before, and why that differed from your direction

### “Unable to allocate CUDA0 buffer” (test-4)

- **Cause:** On the host, the first `cuMemAlloc` was done under a per-VM context (`ensure_vm_context` → `cuCtxCreate`), which could fail and led to "unable to allocate CUDA0 buffer" in the guest.
- **Fix:** Use the **primary context** for allocation and memory ops in `cuda_executor.c` (MEM_ALLOC, MEM_FREE, MEMCPY_*, MEMSET_*, MEM_GET_INFO) instead of `ensure_vm_context`.
- **How it was applied earlier:** The assistant **copied** the updated `cuda_executor.c` to the host (e.g. via `deploy_cuda_executor_to_host.py` or chunked transfer over `connect_host`) and **ran `make` on the host** to rebuild the mediator, then restarted the mediator. That required both **copy** and **build** on the host.

### Why that approach did not align with your direction

You specified that on the host you grant **only permission to read logs**, not to copy or build. So:

- The **previous approach** (push code to host, rebuild mediator on host) is **not** allowed under your current direction.
- The **aligned approach** is: the assistant **documents** what must be done on the host (e.g. “update `cuda_executor.c` with primary-context change, rebuild mediator, restart”) and **does not** perform copy or build on the host. The assistant can still **read** host mediator logs to verify behavior (e.g. cuMemAlloc SUCCESS for vm=9, or module-load rc=200).

### Going forward

- For any **host-only** fix: document the change and the steps (transfer, make, restart); you apply them on the host.
- For **VM-side** fixes (e.g. deploying a Hopper-built `libggml-cuda.so` to the VM): the assistant performs them via VM interaction.
- The assistant uses host log output only to **diagnose and confirm** (e.g. that test-4 is in GPU mode and where the pipeline fails: module-load INVALID_IMAGE).

---

## 4. Brief summary

| Item | Status |
|------|--------|
| **Ollama on test-4 in GPU mode?** | Yes. CUDA path is used; calls and data go guest shims → VGPU-STUB → mediator → H100. |
| **Current blocker** | Host `cuModuleLoadFatBinary` returns INVALID_IMAGE: VM’s `libggml-cuda.so` has no sm_90 (Hopper) kernel image. |
| **Fix (concept)** | Deploy a `libggml-cuda.so` built with `CMAKE_CUDA_ARCHITECTURES=90` to the VM (see BUILD_LIBGGML_CUDA_HOPPER.md). |
| **Assistant role** | VM: full interaction. Host: read logs only; no copy, no build. |
| **Previous host-side fix** | Allocation fix was applied by copying/rebuilding on the host; that method is out of scope under your current permissions. |
