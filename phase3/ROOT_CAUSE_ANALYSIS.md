# Root Cause Analysis - Complete Investigation Results

## Date: 2026-02-26

## Issues Identified and Fixed

### ✅ Issue 1: libvgpu-syscall.so Missing (FIXED)

**Problem:**
- `libvgpu-syscall.so` was listed in `LD_PRELOAD` but the file doesn't exist
- Error: `ERROR: ld.so: object '/usr/lib64/libvgpu-syscall.so' from LD_PRELOAD cannot be preloaded (cannot open shared object file): ignored.`

**Root Cause:**
- The library was never built or deployed
- It was listed in `LD_PRELOAD` configuration but doesn't exist on the system

**Fix Applied:**
- Removed `libvgpu-syscall.so` from `LD_PRELOAD` in `/etc/systemd/system/ollama.service.d/vgpu.conf`
- Updated `LD_PRELOAD` to: `/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so`

**Status:** ✅ Fixed

### ✅ Issue 2: libggml-cuda.so Symlink Name (FIXED)

**Problem:**
- A symlink named `libggml-cuda.so.symlink` existed instead of `libggml-cuda.so`
- Ollama's backend scanner looks for `libggml-cuda.so`, not `libggml-cuda.so.symlink`

**Root Cause:**
- Previous work created a symlink with wrong name
- However, a regular file `/usr/local/lib/ollama/libggml-cuda.so` also exists (the actual library)

**Fix Applied:**
- Removed the incorrectly named symlink `libggml-cuda.so.symlink`
- The regular file `libggml-cuda.so` is the actual library and is correctly in place

**Status:** ✅ Fixed

## Current Status After Fixes

### ✅ Working Components

1. **Symlinks in cuda_v12/ (for runner subprocess):**
   - ✅ `libcuda.so.1` → `/usr/lib64/libvgpu-cuda.so`
   - ✅ `libcudart.so.12` → `/usr/lib64/libvgpu-cudart.so`
   - ✅ `libcudart.so.12.8.90` → `/usr/lib64/libvgpu-cudart.so`
   - ✅ `libnvidia-ml.so.1` → `/usr/lib64/libvgpu-nvml.so`

2. **Top-level library:**
   - ✅ `/usr/local/lib/ollama/libggml-cuda.so` (regular file, actual library)

3. **Environment Variables:**
   - ✅ `OLLAMA_LLM_LIBRARY=cuda_v12`
   - ✅ `OLLAMA_NUM_GPU=999`

4. **LD_PRELOAD:**
   - ✅ Fixed (removed non-existent `libvgpu-syscall.so`)
   - ✅ Order: `libvgpu-exec.so:libvgpu-cuda.so:libvgpu-nvml.so:libvgpu-cudart.so`

5. **Function Calls:**
   - ✅ `cuDeviceGetCount()` is being called in main process
   - ✅ Returns `count=1` correctly
   - ✅ Constructor logs show successful initialization

### ⚠️ Remaining Issue

**Problem:**
- Ollama still reports `initial_count=0` and `library=cpu`
- `cuDeviceGetCount()` is called in main process (returns 1) but runner subprocess may not be calling it

**Observations:**
1. `cuDeviceGetCount()` is called in main Ollama process (PID 143435)
2. Function returns `count=1` correctly
3. But discovery still shows `initial_count=0`
4. This suggests the runner subprocess (which does bootstrap discovery) may not be calling our shim

**Possible Root Causes:**

1. **Runner subprocess not using shims:**
   - Runner subprocess may not have `LD_PRELOAD` inherited
   - Or runner subprocess loads libraries from `cuda_v12/` but our symlinks aren't being used

2. **OLLAMA_LLM_LIBRARY=cuda_v12 not taking effect:**
   - Environment variable may not be reaching runner subprocess
   - Or discovery happens before environment variable is read

3. **libggml-cuda.so not being loaded during discovery:**
   - Backend scanner may not be finding `libggml-cuda.so`
   - Or `ggml_cuda_init()` is failing before calling device count functions

## Next Steps for Investigation

1. **Verify runner subprocess has shims:**
   - Check if runner subprocess loads our shim libraries
   - Verify `LD_PRELOAD` is inherited by runner

2. **Check if libggml-cuda.so is loaded:**
   - Verify `libggml-cuda.so` is loaded during bootstrap discovery
   - Check if `ggml_cuda_init()` is being called

3. **Verify environment variables reach runner:**
   - Check if `OLLAMA_LLM_LIBRARY` and `OLLAMA_NUM_GPU` are in runner process environment

4. **Check discovery timing:**
   - Verify when discovery happens relative to library loading
   - Check if discovery happens before shims are initialized

## Summary

**Fixed Issues:**
- ✅ `libvgpu-syscall.so` missing → Removed from LD_PRELOAD
- ✅ `libggml-cuda.so.symlink` wrong name → Removed (regular file exists)

**Remaining Issue:**
- ⚠️ `initial_count=0` and `library=cpu` despite `cuDeviceGetCount()` returning 1 in main process

**Root Cause Hypothesis:**
The runner subprocess (which performs bootstrap discovery) may not be using our shims, or `libggml-cuda.so` is not being loaded/initialized correctly during discovery.
