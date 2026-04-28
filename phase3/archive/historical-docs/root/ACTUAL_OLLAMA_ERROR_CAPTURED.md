# Actual Ollama (Oyu) Error — Captured

*Captured: Mar 16, 2026 via `capture_ollama_actual_error.py`*

## What we did

1. Stopped the ollama service on the VM.
2. Started `OLLAMA_DEBUG=1 ollama serve` in the background with all output logged to `/tmp/ollama_actual_error.log`.
3. Triggered a generate request.
4. Read the log to get the real server/runner messages.

## Actual error (Oyu) observed

When the server starts **fresh** (not from a long-lived service), GPU discovery runs and the vGPU device is **filtered out**:

```
level=DEBUG source=runner.go:153 msg="filtering device which didn't fully initialize"
  id=GPU-00000000-1400-0000-00c0-000000000000
  libdir=/usr/local/lib/ollama/cuda_v12
  pci_id=0000:3d3bb990:7
  library=CUDA
```

So the **reported “error” from Ollama’s point of view** is: **“device didn’t fully initialize”** — i.e. the device is dropped during discovery and never used for inference.

## What this means technically

- In `discover/runner.go`, Ollama does a **second-pass** check for CUDA (and similar) devices: it starts a **bootstrap runner** with `CUDA_VISIBLE_DEVICES` set to this single vGPU and `GGML_CUDA_INIT=1`.
- It calls `bootstrapDevices(ctx, devices[i].LibraryPath, extraEnvs)`.  
  If that returns **zero devices**, the device is removed with the log line above.
- So in our run, the bootstrap runner (with only the vGPU visible) **did** run (bootstrap took ~820 ms) but **reported 0 devices**.
- Result: Ollama falls back to CPU (`inference compute id=cpu library=cpu`, `total_vram="0 B"`), and the later generate hit a 404 (no GPU path / model not loaded on GPU).

So the **actual Oyu** is: **during bootstrap validation, the runner that sees only the vGPU reports 0 devices**, so Ollama treats the device as “didn’t fully initialize” and filters it out.

## Two scenarios (summary)

| Scenario | What happens |
|----------|----------------|
| **Fresh server start** (e.g. `ollama serve` just started) | Discovery runs → bootstrap runner with vGPU only → reports 0 devices → device filtered as “didn’t fully initialize” → CPU fallback, no GPU. |
| **Long‑lived service** (e.g. after previous successful discovery or different code path) | Discovery may have passed earlier or state differs; we then see the **later** failure: allocs + GEMM succeed, then runner exits with “exit status 2” after the last GEMM, before any instrumented copy/sync/launch. |

## Likely causes for “didn’t fully initialize”

1. **Bootstrap runner reports 0 devices**  
   When the runner process starts with only the vGPU visible, something in the init chain (e.g. `cuDeviceGetCount`, GGML init, or our shim) causes it to return “no devices” to Ollama (e.g. `cuDeviceGetCount` returns 0, or init fails and the runner reports no GPUs).
2. **Timing**  
   Mediator or transport not ready yet during the short bootstrap window, so the shim can’t complete init and the runner concludes there are no devices.
3. **Compute / capability**  
   In other setups, `compute=0.0` led to the same “didn’t fully initialize” filter; here the log showed `compute=8.9` for the device, so filtering was due to **empty device list from bootstrap**, not compute.

## Next steps (to fix “actual Oyu”)

1. **Confirm why bootstrap returns 0 devices**  
   - Ensure the **mediator is running** on the host when the VM’s Ollama server (or service) starts, so the bootstrap runner can talk to the vGPU backend.  
   - Optionally add a small delay or retry in the shim so that a slow mediator doesn’t cause the first `cuDeviceGetCount` to see 0 devices.
2. **Inspect bootstrap runner init**  
   - Add logging (or use existing shim logs) for `cuDeviceGetCount` and `cuInit` during the **bootstrap** run (runner with `CUDA_VISIBLE_DEVices=GPU-...` and `GGML_CUDA_INIT=1`).  
   - Check whether the mediator is already up when this runner starts and whether the first remoted call succeeds.
3. **Service vs manual start**  
   - When ollama runs as a **service**, discovery may have been done when the mediator was up, or state may differ.  
   - Reproduce with: start mediator on host → then start (or restart) ollama on VM → trigger generate; and compare with “start ollama first, then mediator” to see if order fixes “didn’t fully initialize”.

## Script used

- **`capture_ollama_actual_error.py`** — stops the service, runs `ollama serve` with `OLLAMA_DEBUG=1` and log capture, triggers generate, then greps the log for errors. Re-run anytime to re-capture the current “actual Oyu” from the server log.

## Fix applied (skip CUDA init validation)

- **Patched source on VM:** `/home/test-4/ollama/ml/device.go` (and server.go, discover/runner.go) were updated via `transfer_ollama_go_patches.py`. `NeedsInitValidation()` now returns `false` for CUDA so the second-pass validation is skipped.
- **Build on VM failed:** VM has Go 1.18 only; this Ollama needs Go 1.23+. Build `ollama.bin` on a host with Go 1.23+ from the patched source and install on the VM. See **BOOTSTRAP_FIX_SKIP_CUDA_INIT_VALIDATION.md** for steps.

## References

- `phase3/ollama-src/discover/runner.go` (e.g. around 151–157): `bootstrapDevices` and “filtering device which didn’t fully initialize”.
- `OLLAMA_VGPU_REVISIONS_STATUS.md` — overall status and later failure (exit status 2 after GEMM).
- `HOST_LOG_FINDINGS.md` — host-side OOM and mediator notes.
