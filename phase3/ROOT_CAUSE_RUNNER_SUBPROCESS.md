# Root Cause: Runner Subprocess Not Getting Device Count

## Date: 2026-02-26

## Critical Finding

**`cuDeviceGetCount()` is being called in the MAIN process and returns 1, but discovery runs in the RUNNER subprocess which is not getting the device count!**

### Evidence from Logs

**Main Process (pid=148445):**
```
[libvgpu-cuda] cuDeviceGetCount() CALLED (pid=148445)
[libvgpu-cuda] cuDeviceGetCount() SUCCESS: returning count=1 (pid=148445)
[libvgpu-cudart] constructor: cuDeviceGetCount() called, rc=0, count=1
```
✅ **Main process gets count=1**

**Runner Subprocess (discovery):**
```
bootstrap discovery took 239ms
initial_count=0  ← PROBLEM!
library=cpu
```
❌ **Runner subprocess reports initial_count=0**

### The Problem

1. **Main process has shims loaded** ✓
   - `cuDeviceGetCount()` is called
   - Returns count=1

2. **Runner subprocess runs discovery** ✓
   - Bootstrap discovery completes
   - But reports `initial_count=0`

3. **Runner subprocess may not have shims loaded** ✗
   - Or shims are loaded but `cuDeviceGetCount()` is not being called
   - Or `cuDeviceGetCount()` is called but returns 0

### Why This Happens

Discovery runs in a **separate subprocess** (runner) that is spawned by Ollama. The runner subprocess needs to have the shim libraries loaded via `libvgpu-exec.so` (exec interception).

**The runner subprocess should:**
1. Inherit `LD_PRELOAD` via `libvgpu-exec.so` ✓ (should work)
2. Load shim libraries ✓ (should work)
3. Call `cuDeviceGetCount()` during discovery ✗ (NOT happening or returning 0)

### What Needs Investigation

1. **Does runner subprocess have shims loaded?**
   - Check if `libvgpu-cuda.so` is in runner process memory
   - Check if `libvgpu-exec.so` is intercepting exec() correctly

2. **Is `cuDeviceGetCount()` being called in runner?**
   - Check runner logs for `cuDeviceGetCount()` calls
   - Check if runner is calling the function but getting 0

3. **Why is runner not getting count=1?**
   - Maybe runner is calling real CUDA library instead of shim
   - Maybe runner's `cuDeviceGetCount()` is not intercepted
   - Maybe runner's initialization is different

## Next Steps

1. **Check runner subprocess memory maps** - Verify shims are loaded
2. **Check runner logs** - See if `cuDeviceGetCount()` is called
3. **Verify exec interception** - Ensure `libvgpu-exec.so` is working
4. **Compare with Feb 25** - What was different when it worked?

## Conclusion

**The main process correctly gets count=1, but the runner subprocess (where discovery runs) is reporting initial_count=0.**

This is why GPU mode is not active - discovery runs in the runner subprocess, and the runner is not getting the correct device count.
