# Error tracking status (from where we left off)

*Updated: Mar 17, 2026*

---

## Goal and Phase 1 (do not lose focus)

- **Overall goal (Phase 3):** All GPU-utilizing software in the VM sees the VGPU-STUB as a GPU and uses the data flow to the host and back.
- **Phase 1 / Stage 1 milestone:** Successfully complete **GPU-mode inference in Ollama** in the VM (proof of the path end-to-end): discovery in GPU mode, model load over vGPU, inference, response back.
- Before diving into tracking/fixing any issue: **confirm Ollama is running and operating in GPU mode** (e.g. `systemctl is-active ollama`, API 200, `journalctl | grep "inference compute".*library=CUDA`). Then fix the blocker so the runner reaches alloc/HtoD and the guest sees completion (MMIO or response_len).

---

## 1. Review result: response_len workaround for HtoD

- **Done:** Ran generate (tinyllama, 120s and 180s timeout) with `/tmp/vgpu_host_response_verify.log` and `/tmp/vgpu_call_sequence.log` cleared.
- **Result:** No `SUBMIT call_id=0x0032` or `0x0030`; no `BREAK reason=RESPONSE_LEN`; no `BREAK reason=TIMEOUT`. Call sequence contained only init/context (cuInit, cuGetGpuInfo, cuDevicePrimaryCtxRetain, cuCtxSetCurrent); HtoD count 0.
- **Conclusion:** The runner **never reached** the first cuMemAlloc or HtoD in these runs. The response_len workaround was **not triggered** and could not be evaluated for HtoD. **Not worthwhile** to keep testing response_len for HtoD until the runner is confirmed to reach the alloc/HtoD path.

---

## 2. Current blocker: runner never sends alloc/HtoD

- **Observed:** On generate, the verify log shows only the same 6 RPCs (init + context), each completing with `BREAK reason=STATUS status=0x02`. No SUBMIT for 0x0030 (cuMemAlloc) or 0x0032 (cuMemcpyHtoD_v2).
- **Implication:** The process that performs those 6 RPCs either (a) exits or hands off before model load, or (b) the model-load path never issues the first alloc/HtoD (e.g. falls back to CPU after "unable to refresh free memory", or blocks before first alloc).

---

## 3. "unable to refresh free memory" and refresh patch

- **Journal:** When the generate runner starts, we see `inference compute library=CUDA` then `unable to refresh free memory, using old values`.
- **VM code:** On the VM, `discover/runner.go` line 340 already has the correct refresh order: `bootstrapDevices(ctx, []string{dir, ml.LibOllamaPath}, devFilter)`. So the refresh **patch is applied**.
- **Conclusion:** Refresh may still fail for another reason (e.g. timeout, or a CUDA call in the refresh path failing). The scheduler may still use "old values" and proceed with GPU load; the next step is to confirm whether the **load path** actually uses GPU and where it stops.

---

## 4. Root cause: runner LD_LIBRARY_PATH wrong (fixed)

- **Finding:** The generate **runner** process had **LD_LIBRARY_PATH=/usr/lib64** only (no `/opt/vgpu/lib`). The **serve** process had the correct env from systemd (`/usr/local/lib/ollama/cuda_v12:/opt/vgpu/lib:...`). So the runner was not loading the vGPU shims for model load (alloc/HtoD), which is why we never saw SUBMIT 0x0030/0x0032.
- **Fix applied:** In `llm/server.go`, when passing env to the runner we now **prepend `/opt/vgpu/lib`** to `LD_LIBRARY_PATH` if the value does not already contain it (`transfer_ollama_go_patches.py` and VM apply script). Rebuilt and installed on the VM (PATCHED_SERVER_UPGRADE, BUILD_EXIT=0). VM `server.go` now contains the defensive block.
- **Verification (Mar 17):** Added server-side debug log (`runner env LD_LIBRARY_PATH`). After rebuild and restart, journal shows: `value="LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/opt/vgpu/lib:/usr/local/lib/ollama"` when starting runner. So the runner **now receives** `/opt/vgpu/lib` first. Despite that, verify log and call_sequence still show **no SUBMIT 0x0030/0x0032** and only the same 6 init/context RPCs. So the blocker is **not** LD_LIBRARY_PATH anymore; the runner has the right env but model load (alloc/HtoD) is still not reaching the vGPU shim path.

## 5. Next steps (error tracking)

1. **Why does the runner not reach alloc/HtoD despite correct LD_LIBRARY_PATH?**
   - Run a generate with **OLLAMA_DEBUG=1** (e.g. in systemd override or `OLLAMA_DEBUG=1 ollama run tinyllama 'Hi'`) and capture logs for load path: "load_backend", "using device", "loading tensors", or CPU fallback.
   - Check whether **"unable to refresh free memory"** causes the scheduler to skip GPU for this load (e.g. empty GPU list after refresh), so the runner uses CPU and never calls cuMemAlloc/cuMemcpyHtoD.

2. **If the runner does use GPU for load but we still see no SUBMIT 0x0030/0x0032**
   - The first alloc/HtoD might be coming from a different process (e.g. a separate loader process that does not use the vGPU shim), or the runner might block before the first CUDA alloc (e.g. waiting on server, file I/O, or refresh).

3. **If the runner falls back to CPU after "unable to refresh free memory"**
   - Dig into why refresh fails despite the correct `dirs` order (e.g. bootstrap timeout, `cuMemGetInfo` failure in refresh context, or library path in that context).

4. **Re-test response_len once alloc/HtoD are observed**
   - After the runner is confirmed to reach the alloc/HtoD path (SUBMIT 0x0030/0x0032 in verify log), run again and check for `BREAK reason=RESPONSE_LEN` vs `BREAK reason=TIMEOUT` to see if the workaround helps.

---

## 6. References

- **Verification log format:** `HOST_RESPONSE_VERIFY_LOG.md`
- **Runner env / discovery:** `ROOT_CAUSE_RUNNER_SUBPROCESS.md`, `OLLAMA_RUNNER_LD_PRELOAD_PATCH.md`, `DISCOVER_REFRESH_CUDA.md`
- **HtoD / MMIO:** `HtoD_DIAGNOSIS_RESULTS.md`, `MMIO_WORKAROUND_RESPONSE_LEN.md`, `VERIFICATION_REPORT.md`
