# Current GPU Detection Status

## Date: 2026-02-27

## ✅ What's Working

1. **Shim Libraries Loaded**: ✅
   - `libvgpu-cuda.so` (Driver API shim) loaded via LD_PRELOAD
   - `libvgpu-cudart.so` (Runtime API shim) loaded via LD_PRELOAD
   - Both shims are active in the main Ollama process

2. **CUDA API Functions Working**: ✅
   - `cuInit()` → returns success (rc=0)
   - `cuDeviceGetCount()` → returns count=1
   - `cudaGetDeviceCount()` → returns count=1
   - VGPU-STUB device found at 0000:00:05.0

3. **Environment Configured**: ✅
   - `OLLAMA_LLM_LIBRARY=cuda_v12` set in systemd service
   - `LD_PRELOAD` configured correctly
   - `LD_LIBRARY_PATH` includes CUDA backend directory

## ❌ Current Blocker

**Ollama still reports `library=cpu` instead of `library=cuda`**

### The Problem

Ollama's bootstrap discovery process:
1. ✅ Loads our shims (via LD_PRELOAD)
2. ✅ Calls `cuInit()` and `cuDeviceGetCount()` → both succeed
3. ❌ **Does NOT load `libggml-cuda.so` (CUDA backend library)**
4. ❌ Reports `library=cpu` and `initial_count=0`

### Root Cause Hypothesis

**Ollama's backend loading logic checks for GPU presence BEFORE loading the CUDA backend.**

The discovery flow appears to be:
1. Check if GPU exists (calls `cuDeviceGetCount()`)
2. If GPU count = 0 → skip CUDA backend loading
3. If GPU count > 0 → load CUDA backend

**But our shim returns count=1, yet the backend still isn't loading.**

### Possible Reasons

1. **Backend init function fails silently**
   - `libggml-cuda.so` loads but `ggml_backend_cuda_init()` fails
   - No error logged

2. **Missing function calls during discovery**
   - GGML might call `cuDeviceGet()`, `cuDeviceGetAttribute()`, etc.
   - If these aren't called or fail, backend init fails

3. **Runner subprocess doesn't inherit environment**
   - Discovery happens in runner subprocess
   - Runner might not have LD_PRELOAD or OLLAMA_LLM_LIBRARY

4. **Backend library path issue**
   - Ollama can't find `libggml-cuda.so` in expected location
   - Or library exists but can't be loaded

## Next Steps to Fix

### 1. Verify Backend Library Loading
```bash
# Check if libggml-cuda.so is being loaded
LD_DEBUG=libs ollama list 2>&1 | grep ggml
```

### 2. Check Runner Subprocess Environment
```bash
# Find runner PID and check its environment
ps auxf | grep 'ollama runner'
cat /proc/<runner_pid>/environ | tr '\0' '\n' | grep -E 'LD_PRELOAD|OLLAMA'
```

### 3. Add More Logging to Shim
- Log when `cuDeviceGet()` is called
- Log when `cuDeviceGetAttribute()` is called
- Log when `cuCtxCreate()` is called
- These functions might be called during backend init

### 4. Force Backend Loading
- Try setting additional environment variables
- Check if there's a way to bypass bootstrap filtering
- Verify backend library can be loaded manually

## Current Status Summary

**✅ Architecture is correct (software-level virtualization)**
**✅ Shim libraries are working**
**✅ CUDA APIs return correct values**
**❌ Ollama's CUDA backend isn't loading during discovery**
**❌ Result: Ollama falls back to CPU**

The blocker is in Ollama's backend loading logic, not in our shim implementation.
