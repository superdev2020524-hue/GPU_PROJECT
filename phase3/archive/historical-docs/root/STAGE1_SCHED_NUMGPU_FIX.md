# Stage 1: Scheduler NumGPU fix (use GPU when API omits num_gpu)

*Applied: Mar 18, 2026 — to reach alloc/HtoD and complete Stage 1.*

---

## Blocker

- **Symptom:** Generate runner only performs 6 init/context RPCs (cuInit, cuGetGpuInfo, cuDevicePrimaryCtxRetain, cuCtxSetCurrent). No `cuMemAlloc` (0x0030) or `cuMemcpyHtoD_v2` (0x0032). Host mediator sees only those 6 results for vm=9.
- **Cause:** In `server/sched.go`, when **`pending.opts.NumGPU == 0`** the scheduler sets **`gpus = []ml.DeviceInfo{}`** and never calls `getGpuFn`. So the load gets **no GPUs** → CPU load → no CUDA alloc/HtoD. The API often sends no `num_gpu` (omitempty), which decodes as **0** in Go, so the default generate request forces CPU load even when GPUs exist.

---

## Fix (applied on VM)

**File:** `server/sched.go` (in the block that gets a refreshed GPU list for loading).

**Change:** Always call `getGpuFn` first. If `NumGPU == 0` but `getGpuFn` returns GPUs, set `pending.opts.NumGPU = -1` (dynamic) and use the returned list so the load uses GPU. Only use `gpus = []` when `NumGPU == 0` and there are no GPUs.

**Script:** `patch_sched_numgpu.py` (run on VM: `python3 patch_sched_numgpu.py /home/test-4/ollama/server/sched.go`).

**Result:** After patch, rebuild `ollama.bin`, install to `/usr/local/bin/ollama.bin.real`, restart ollama. Then trigger generate and check:
- `grep -E "0x0030|0x0032" /tmp/vgpu_call_sequence.log` should show lines (alloc/HtoD).
- Host mediator log should show `cuMemAlloc` / `cuMemcpyHtoD` for vm=9.

---

## Verification

- Patch applied on VM: `grep "Phase3/vGPU" /home/test-4/ollama/server/sched.go` → line 205.
- Rebuild: `go build -o ollama.bin .` → BUILD_EXIT=0.
- **Important:** Ensure the **running** process is the new binary: stop ollama, copy binary, start ollama, then run generate. If `systemctl stop` hangs, the old binary may still be in use and the fix will not take effect.

---

## If alloc/HtoD still don’t appear

**Retest (Mar 18):** Generate with default and with `options.num_gpu=1` still showed alloc/HtoD count=0; journal shows two runners, load runner after "unable to refresh free memory". Next: OLLAMA_DEBUG=1 or log at cuMemAlloc in shim.

1. Confirm the running exe is the new binary: `readlink -f /proc/$(pgrep -f "ollama.bin.real serve" | head -1)/exe`.
2. If `getGpuFn` returns empty (e.g. refresh path returns no devices), the scheduler will still pass empty `gpus` when `NumGPU == 0`. Then either fix the refresh path so it returns the device list, or ensure initial discovery populates the list that refresh reuses as “old values”.
