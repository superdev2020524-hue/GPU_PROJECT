# CUDA discovery refresh during model load

## What this fixes

When the scheduler loads a model, it calls **GPUDevices()** again to get a **refreshed GPU list** (and update free VRAM). That refresh can take two paths:

1. **Use existing runners** – `runner.GetDeviceInfos(ctx)` on already-loaded runners.
2. **Fallback: bootstrap again** – if not all devices were updated, it runs **bootstrapDevices(ctx, dirs, devFilter)** with `dirs = []string{ml.LibOllamaPath, dir}` (parent first, then e.g. `cuda_v12`).

The **initial** bootstrap was patched so `dirs = []string{dir, ml.LibOllamaPath}` (GPU lib dir first), so the backend loader finds `libggml-cuda.so` and sees the GPU. The **refresh** path used a **hardcoded** `[]string{ml.LibOllamaPath, dir}` and was **not** patched, so during model load the refresh could run bootstrap with the wrong order and fail to see CUDA → "unable to refresh free memory, using old values" and/or the scheduler could get an empty or wrong GPU list.

## Code locations (ollama discover/runner.go)

- **Initial bootstrap** (first discovery): `dirs = []string{ml.LibOllamaPath, dir}` around line 109 → patched to `dirs = []string{dir, ml.LibOllamaPath}`.
- **Refresh fallback** (during model load): `bootstrapDevices(ctx, []string{ml.LibOllamaPath, dir}, devFilter)` around line 340 → must be patched to `[]string{dir, ml.LibOllamaPath}`.

## Patches applied

- **transfer_ollama_go_patches.py** – `patch_discover_runner_go()` applies both the initial `dirs` order and the refresh-path `bootstrapDevices(..., []string{dir, ml.LibOllamaPath}, ...)`.
- **--from-vm** apply script – the same two discover/runner.go replacements are applied on the VM.

After applying, rebuild `ollama.bin` and restart the service. Then trigger a model load/generate and check logs for "refreshing free memory" and that you do **not** see "unable to refresh free memory" due to CUDA not being found in the refresh bootstrap.
