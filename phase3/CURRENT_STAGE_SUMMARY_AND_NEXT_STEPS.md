# Current stage summary and next steps

*Mar 16, 2026 — summary of how we reached here and how to verify / continue to Stage 1.*

---

## Goal (reference)

- **Ultimate goal (Phase 3):** All GPU-utilizing software in the VM sees the VGPU-STUB as a GPU and uses the data flow to the host and back.
- **Stage 1 milestone:** Successfully complete GPU-mode inference in Ollama in the VM (proof of the path end-to-end).

See **PHASE3_PURPOSE_AND_GOALS.md**.

### Direction (do not deviate)

- **Stage 1 is proof of the path:** Ollama is the first target to prove the **vGPU data path** works (shims → transport → stub → mediator → host CUDA). The fix is in the **general path** (transport, BAR reply delivery, shims), not in Ollama-specific code.
- **Actual blocker:** Runner blocks during **model load** waiting for host response to **cuMemcpyHtoD_v2** (ACTUAL_ERROR_FOUND.md). Diagnosis (HtoD_DIAGNOSIS_RESULTS.md): host applies DONE and stub returns 0x2, but **guest MMIO read sees 0x01 (BUSY)** — bug is in the path between QEMU MMIO return and the guest (Xen/device model or BAR mapping). Fix that path so the guest sees DONE and load completes.
- **Do not:** Drift into Ollama-only work (runner env, discovery CPU fallback, runner stderr) except as the minimal step to confirm a **path** failure. Priority is HtoD reply visibility and transport, then any remaining path bug that causes exit 2.

---

## How we reached the current stage

### 1. Host and guest pipeline (already in place)

- **Host:** Mediator (`mediator_phase3`) receives requests from the VM via VGPU-STUB, replays CUDA/CUBLAS on the physical GPU (H100), returns results.
- **Guest:** Shims (`libvgpu-cuda`, `libvgpu-cudart`, `libvgpu-cublas`, etc.) in `/opt/vgpu/lib` intercept CUDA/CUBLAS; runner gets `LD_LIBRARY_PATH` with `/opt/vgpu/lib` first (patched Ollama `llm/server.go`).
- **Fixes already applied (host):** Primary context for module ops and memory/CUBLAS; raw fat binary for module load; OOM addressed by restarting mediator when GPU was full.
- **Fixes already applied (guest):** Module load and post-module allocs succeed; CUBLAS shim used; `cublasCreate`, `set_stream`, `cublasGemmEx` reach the host and return success; guest log shows `gemm_ex_return` — so we get **past** GEMM.

### 2. Two separate issues we addressed

**Issue A — “Filtering device which didn’t fully initialize” (discovery)**

- **What happened:** On a **fresh** Ollama server start, the second-pass “init validation” runs a bootstrap runner with only the vGPU visible. That runner reported **0 devices**, so Ollama filtered the vGPU and fell back to CPU (`total_vram="0 B"`, 404 on generate).
- **Cause:** `bootstrapDevices()` returns the device list from that runner; if the runner crashes or returns empty, the device is dropped with the log message above.
- **Fix:** Skip the second-pass validation for CUDA so the first-pass device list is kept: in `ml/device.go`, `NeedsInitValidation()` now returns `false` for CUDA (`return d.Library == "ROCm"` only). Requires a **patched Ollama binary**.
- **VM build:** Go 1.26.1 is at `/usr/local/go/bin/go` on the VM; always check with `go version` and `/usr/local/go/bin/go version` before concluding the VM cannot build. We:
  - Transferred the patched source (device.go, server.go, discover/runner.go) to the VM via `transfer_ollama_go_patches.py`.
  - Installed **Go 1.26.1** on the VM with **`install_go_and_build_ollama_on_vm.py`** (download on VM from go.dev, extract to `/usr/local/go`, build `ollama.bin`, install and restart ollama).
- **Result:** The VM now runs the **patched** `ollama.bin`; discovery should no longer filter the vGPU as “didn’t fully initialize.”

**Issue B — Runner exit status 2 after GEMM (inference path)**

- **What happened:** With discovery and model load working (and host OOM fixed), the runner proceeded through module load, allocs, CUBLAS create, set_stream, and multiple **successful** `cublasGemmEx` RPCs (`gemm_ex tc_rc=0`, `gemm_ex_return` in the guest log). Then the runner **exits with status 2** before any of: `cuLaunchKernel`, `cuCtxSynchronize`, `cuStreamSynchronize`, `cuMemcpyDtoH` (none of these appear in `/tmp/vgpu_next_call.log`).
- **Cause (current understanding):** The crash is **after** the last GEMM returns and **before** the next instrumented CUDA call — i.e. inside GGML or in another CUDA API we do not yet log. The server only reports “exit status 2”; no concrete CUDA error string was captured.
- **Status:** Not yet fixed. This is **where we stopped** before focusing on the discovery fix (Issue A).

### 3. Documents and scripts that got us here

| Item | Purpose |
|------|--------|
| **ACTUAL_OLLAMA_ERROR_CAPTURED.md** | Captured “actual Oyu”: “filtering device which didn’t fully initialize” from server log. |
| **BOOTSTRAP_FIX_SKIP_CUDA_INIT_VALIDATION.md** | Skip CUDA init validation; build/install options (local vs VM). |
| **VM_GO_UPGRADE_AND_BUILD.md** | Upgrade Go on VM; move/transfer tarball if needed. |
| **install_go_and_build_ollama_on_vm.py** | Install Go 1.26.1 on VM, build patched ollama.bin, install and restart. |
| **transfer_ollama_go_patches.py** | Apply patches in memory, transfer device/server/runner.go to VM (build was then done after Go upgrade). |
| **capture_ollama_actual_error.py** | Run server with logging, trigger generate, read log (to re-check discovery if needed). |
| **HOST_LOG_FINDINGS.md** | Host OOM, GEMM path, what to check on host. |
| **ACTUAL_ERROR_VERIFICATION.md** | Verified actual errors: client timeout → "context canceled"; server timeout → "timed out waiting for llama runner to start - progress 0.00". |
| **OLLAMA_VGPU_REVISIONS_STATUS.md** | Host/guest revisions, diagnostics, GEMM progression, exit 2. |

---

## Verify whether the discovery issue (Issue A) is fully resolved

Use this to confirm that **“filtering device which didn’t fully initialize”** no longer occurs and the vGPU is kept.

### 1. Restart Ollama so discovery runs with the new binary

On the VM (or via `connect_vm.py`):

```bash
sudo systemctl restart ollama
sleep 5
```

### 2. Check that the vGPU is in the device list (no CPU fallback)

Trigger discovery (e.g. list models or hit the API), then check logs:

```bash
sudo journalctl -u ollama -n 80 --no-pager | grep -E "inference compute|total_vram|filtering device|didn't fully initialize"
```

- **Success:** You see a line like `inference compute ... library=CUDA ... total="80.0 GiB"` (or similar non-zero VRAM) and **no** line containing `filtering device which didn't fully initialize`.
- **Failure:** You see `filtering device which didn't fully initialize` and/or `library=cpu`, `total_vram="0 B"`.

### 3. Thorough: run the capture script (fresh server + generate)

To simulate a cold start and confirm the bootstrap path does not filter the device:

```bash
cd /path/to/phase3
python3 capture_ollama_actual_error.py
```

The script now starts the **patched** binary (`/usr/local/bin/ollama.bin serve`) with the same env as the service, so it tests the discovery fix. (The default `ollama` command on the VM runs `ollama.real` (unpatched); using it would still show "filtering device which didn't fully initialize".)

In the printed log, check for **absence** of `filtering device which didn't fully initialize` and **presence** of:
- `inference compute ... library=CUDA ... total="80.0 GiB"` (or similar)
- `vram-based default context total_vram="80.0 GiB"`

**Verified (Mar 16):** With the script updated to use `ollama.bin`, the log shows CUDA device kept, `total="80.0 GiB"`, `total_vram="80.0 GiB"`, and **no** "filtering device which didn't fully initialize". Discovery issue (Issue A) is resolved when the patched binary is used. The script restarts the ollama service at the end.

### 4. Quick API check

```bash
curl -s http://127.0.0.1:11434/api/tags
# and/or
curl -s -X POST http://127.0.0.1:11434/api/generate -H "Content-Type: application/json" \
  -d '{"model":"llama3.2:1b","prompt":"Hi","stream":false,"options":{"num_predict":2}}'
```

- If discovery is fixed: tags may list models; generate may still fail with **exit status 2** (Issue B) but you should **not** get a 404 due to “no GPU” or “model not loaded.” A 500 with a message about the runner exiting is the known inference failure (Issue B), not discovery.

**Conclusion:** If the log shows CUDA with non-zero VRAM and no “filtering device which didn’t fully initialize,” then **Issue A (discovery) is resolved.** You can then treat the remaining problem as **Issue B** and continue from where we stopped.

---

### 5. Runner env (LD_PRELOAD) verified (Mar 17)

After the **OLLAMA_RUNNER_LD_PRELOAD_PATCH**, the discovery runner was confirmed to have the correct env on one run: restart ollama, poll every 0.05s for a child whose cmdline contains `runner`, dump `/proc/<pid>/environ`. **Result:** Runner had `LD_PRELOAD=/opt/vgpu/lib/libvgpu-cuda.so.1` and `LD_LIBRARY_PATH=/opt/vgpu/lib:...`. So the runner can get the shims. If discovery still reports `library=cpu` / `total_vram="0 B"`, the cause is GPU init/discovery falling back to CPU or the runner not actually loading our shims.

### 6. CPU mode cause and fix — resolved (Mar 17)

**Root cause (phase3 docs, VM_TEST3_GPU_MODE_STATUS.md, ROOT_CAUSE_FIXED.md):** GPU detection depends on the **GGML CUDA backend** (`libggml-cuda.so` / `libggml-cuda-v12.so`) being loaded during discovery. That .so is in the VM path (e.g. `/usr/local/lib/ollama/cuda_v12/` with top-level symlinks). The loader finds it only when the **runner does NOT have LD_PRELOAD**: with LD_PRELOAD, the runner’s `dlopen` resolves to libdl and the backend **never** loads `libggml-cuda.so`; without LD_PRELOAD, real `dlopen` loads `libggml-cuda.so`, which pulls in our shim via `LD_LIBRARY_PATH`, and the CUDA backend sees the GPU.

**Fix applied:** Rebuild Ollama with **runner env** that (1) passes `LD_LIBRARY_PATH`, `OLLAMA_LIBRARY_PATH`, `OLLAMA_LLM_LIBRARY`, `OLLAMA_NUM_GPU` to the runner, and (2) **strips `LD_PRELOAD`** from the runner’s `cmd.Env` so the backend loader can load `libggml-cuda.so`. Use `transfer_ollama_go_patches.py --from-vm` to apply the patch on the VM (server.go + discover/runner.go), build, and install.

**Verified (Mar 17):** After applying the patch and restarting, discovery shows `library=CUDA`, `description="NVIDIA H100 80GB HBM3"`, `total="80.0 GiB"`.

**Refresh during model load:** The scheduler calls `GPUDevices()` again when loading a model (to refresh free VRAM). That path can fall back to **bootstrapDevices** with a hardcoded `[]string{ml.LibOllamaPath, dir}` (parent first). If that order is used, the refresh bootstrap can fail to see CUDA and you get "unable to refresh free memory, using old values". **Fix:** Patch the refresh call to use `[]string{dir, ml.LibOllamaPath}` as well. See **DISCOVER_REFRESH_CUDA.md**. The same `transfer_ollama_go_patches.py --from-vm` (and `patch_discover_runner_go`) now apply both the initial and the refresh-path discover/runner.go fixes.

---

## Next step to reach Stage 1 (continue from where we stopped)

Once you have verified that discovery is resolved (vGPU kept, no “didn’t fully initialize”):

- **We continue from where we stopped:** the runner **exit status 2** that happens **after** successful GEMM returns and **before** `cuLaunchKernel` / `cuCtxSynchronize` / `cuStreamSynchronize` / `cuMemcpyDtoH` (see **HOST_LOG_FINDINGS.md**, **OLLAMA_VGPU_REVISIONS_STATUS.md**).

**Actual error identified (Mar 16):** Runner blocks during load waiting for host response to **cuMemcpyHtoD_v2**. See **ACTUAL_ERROR_FOUND.md** for details and host-side checks.

### Immediate next steps for Issue B (post-GEMM crash)

1. **Get a concrete error (if possible)**  
   - Run inference in a way that exposes the runner’s stderr (e.g. **capture_runner_stderr.py** or `OLLAMA_DEBUG=1 ollama run llama3.2:1b Hi` on the VM) and look for a CUDA/GGML error string.  
   - If the server or runner logs a specific error code or message, use that to target the failing call.

2. **Narrow the crash site**  
   - Add entry-point logging to more CUDA APIs that GGML might call after GEMM (e.g. any other sync/copy/launch variants), or inspect GGML’s CUDA path right after the GEMM that runs before the first sync/copy/launch.  
   - Confirm whether the process is exiting (crash) or hanging (e.g. transport timeout).

3. **Host log**  
   - After a failing generate, check `/tmp/mediator.log` on the host for any new errors (e.g. failed calls for vm=9 after GEMM, or timeouts). See **HOST_LOG_FINDINGS.md**.

4. **When the crash is fixed**  
   - A full generate should complete without exit status 2; the response should return generated text. That would complete **Stage 1**: Ollama GPU-mode inference working end-to-end on the vGPU path.

### Stage 1 checklist (for when both issues are resolved)

- [ ] Discovery: vGPU kept (no “filtering device which didn’t fully initialize”).
- [ ] Model load: Allocations and HtoD succeed (host has free GPU; no OOM).
- [ ] Inference: CUBLAS GEMM and subsequent CUDA calls succeed; runner does not exit with status 2.
- [ ] Result: Generate returns HTTP 200 with generated text (Ollama GPU inference over vGPU).

---

## Short reference

| Issue | Symptom | Fix / status |
|-------|--------|---------------|
| **A. Discovery** | “filtering device which didn’t fully initialize”; CPU fallback | Skip CUDA init validation (patched ollama.bin); Go upgraded on VM; patched binary built and installed. **Verify** with steps above. |
| **B. Post-GEMM crash** | Runner exit status 2 after `gemm_ex_return`; no launch_kernel/ctx_sync in log | Not yet fixed. **Next:** get concrete error; add logging or fix GGML/CUDA path; then re-test to achieve Stage 1. |

---

## Related docs

- **PHASE3_PURPOSE_AND_GOALS.md** — Ultimate goal and Stage 1 milestone.
- **ACTUAL_OLLAMA_ERROR_CAPTURED.md** — How we identified the discovery “actual Oyu.”
- **BOOTSTRAP_FIX_SKIP_CUDA_INIT_VALIDATION.md** — Skip init validation and build options.
- **VM_GO_UPGRADE_AND_BUILD.md** — Go upgrade and file transfer for VM build.
- **HOST_LOG_FINDINGS.md** — Host OOM, GEMM, and what to check after a generate.
- **OLLAMA_VGPU_REVISIONS_STATUS.md** — Full revision and diagnostic history.
