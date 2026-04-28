# Current state (test-4) and direction

*Document created: Mar 15, 2026*

*Update: Mar 30, 2026 — the section below records the current state after restoring `SHMEM`. Older `INVALID_IMAGE`-only paragraphs later in this file are historical and do not fully describe the latest state.*

## Mar 30, 2026 — current state and best next step

### What just changed

- `SHMEM` was restored for `vm_id=9` by lowering the shared minimum from `8 MB` to `4 MB` in the shared protocol header (`include/vgpu_protocol.h`), then rebuilding and reinstalling the host QEMU `VGPU-STUB`.
- The actual recovery process was:
  1. update `VGPU_SHMEM_MIN_SIZE` to `4 MB`
  2. copy `src/vgpu-stub-enhanced.c`, `include/vgpu_protocol.h`, and `include/cuda_protocol.h` to dom0
  3. run `make qemu-prepare && make qemu-build` on dom0
  4. install the rebuilt QEMU RPM on dom0
  5. restart `mediator_phase3`
  6. reboot `Test-4`
- The first failed attempt during this recovery was not a code failure in `SHMEM`; it was an operational failure while reinstalling QEMU / restarting the VM. Re-running the dom0 steps cleanly recovered the stack.

### Evidence that `SHMEM` is back

- Host `daemon.log` now shows explicit `shmem registered` lines for `vm_id=9` on the rebuilt stub.
- The new build accepted a `4 MB` shared window:
  - `vm_id=9: shmem registered ... size=4 MB`
- After that recovery, a bounded `tinyllama` generate from inside the VM returned `HTTP=200` instead of stalling in the old long `BAR1` timeout pattern.
- The last bounded run completed in about `29.46s`, and the response JSON reported a load duration of about `3.27s`.
- `BAR1` status reads still appear in host logs. That is the status-mirror path and does **not** by itself mean bulk data is still using `BAR1`.

### What this means

- The transport bottleneck is no longer the main blocker for Step 1.
- GPU-mode execution is alive enough to complete a bounded request.
- The current blocker has moved forward from **transport/path selection** to **output correctness**.
- The latest successful bounded run returned `HTTP=200`, but the textual response was still garbage bytes rather than a correct answer.

### Best next step

**Do not spend the next cycle tuning `SHMEM` again.** The best next step is to isolate the **first corruption point in the successful GPU path**.

Concretely:

1. Run a **paired CPU vs GPU** `tinyllama` request with the same short deterministic prompt.
2. Keep the current restored `SHMEM` setup unchanged while capturing:
   - VM `journalctl -u ollama`
   - host `/var/log/daemon.log`
   - host `/tmp/mediator.log`
3. Add **focused output-path tracing** around the first small result return:
   - guest `cuMemcpyDtoH` / `cudaMemcpyDtoH`
   - host small `DtoH` payload logging in `cuda_executor.c`
4. Compare the first returned bytes on the GPU path against the CPU baseline.
5. If bytes are already wrong at `DtoH`, stay in transport / host replay.
6. If `DtoH` bytes are correct but Ollama text is wrong, move up into GGML / runner decode handling.

### Short recommendation

The best next step is: **freeze the restored transport, then debug correctness at the first returned output bytes.**  
The mistake to avoid is: **touching `SHMEM` / QEMU again before proving the corruption point.**

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

**See `ASSISTANT_PERMISSIONS.md` for the authoritative statement** (it overrides older paragraphs in this file).

- **VM (test-4):** **Full authority** — run commands, configure, deploy, edit VM files, read logs, rebuild and install (e.g. ollama.bin, guest shims), restart services.
- **Host (dom0 / mediator):** As of **2026-03-25**, the operator granted the assistant **autonomous** Phase 3 work on the host: **read** logs and files; **edit** sources/config under agreed paths (e.g. `/root/phase3`); **build** (`make`, etc.); **install** / **restart** mediator when documented; **deploy** from the workspace (e.g. `deploy_cuda_executor_to_host.py`, `connect_host.py`). **Binding condition:** **non-destruction** (no reckless wipes of system trees; risky changes → stop and ask). *Before that date, host was treated as read-only for the assistant; session notes may still say “human/dom0 only” for that period.*

### Implications

- **Host-side fixes** (e.g. `cuda_executor.c`, mediator rebuild/restart) **may** be performed by the assistant **when** they stay within **`ASSISTANT_PERMISSIONS.md`** and deployment docs.
- **VM-side work** remains with the assistant as before: Ollama, journal, guest shims, Hopper `libggml-cuda.so`, etc., plus **reading** host logs to confirm behavior (e.g. vm=9, module-load rc=…).

---

## 3. How similar issues were handled in PHASE 3 before, and why that differed from your direction

### “Unable to allocate CUDA0 buffer” (test-4)

- **Cause:** On the host, the first `cuMemAlloc` was done under a per-VM context (`ensure_vm_context` → `cuCtxCreate`), which could fail and led to "unable to allocate CUDA0 buffer" in the guest.
- **Fix:** Use the **primary context** for allocation and memory ops in `cuda_executor.c` (MEM_ALLOC, MEM_FREE, MEMCPY_*, MEMSET_*, MEM_GET_INFO) instead of `ensure_vm_context`.
- **How it was applied earlier:** The assistant **copied** the updated `cuda_executor.c` to the host (e.g. via `deploy_cuda_executor_to_host.py` or chunked transfer over `connect_host`) and **ran `make` on the host** to rebuild the mediator, then restarted the mediator. That required both **copy** and **build** on the host.

### Historical note (permissions evolution)

At one point the operator restricted the assistant to **host read-only**; some older work notes still describe host changes as **operator-only**. **Current** rules are in **`ASSISTANT_PERMISSIONS.md`** (2026-03-25 grant). When in doubt, follow that file—not dated “read-only host” sentences elsewhere in PHASE3.

### Going forward

- **Host:** implement or deploy per **`ASSISTANT_PERMISSIONS.md`** (non-destruction, documented paths).
- **VM:** unchanged — assistant performs guest-side work and uses host logs for verification.

---

## 4. Brief summary

| Item | Status |
|------|--------|
| **Ollama on test-4 in GPU mode?** | Yes. CUDA path is used; calls and data go guest shims → VGPU-STUB → mediator → H100. |
| **Current blocker** | Host `cuModuleLoadFatBinary` returns INVALID_IMAGE: VM’s `libggml-cuda.so` has no sm_90 (Hopper) kernel image. |
| **Fix (concept)** | Deploy a `libggml-cuda.so` built with `CMAKE_CUDA_ARCHITECTURES=90` to the VM (see BUILD_LIBGGML_CUDA_HOPPER.md). |
| **Assistant role** | VM: full. Host: per **`ASSISTANT_PERMISSIONS.md`** (dom0 edit/build/restart allowed under non-destruction; not “read-only only”). |
| **Previous host-side fix** | Allocation fix was applied by copying/rebuilding on the host; that method is out of scope under your current permissions. |
