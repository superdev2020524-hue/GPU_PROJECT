# Why GPU is Detected But Not Used

## The Problem

**Ollama detects the GPU but uses CPU backend instead.**

### Evidence

From logs:
```
[libvgpu-cuda] cuInit() device found at 0000:00:05.0
[libvgpu-cuda] cuDeviceGetCount() SUCCESS: returning count=1
load_backend: loaded CPU backend from /usr/local/lib/ollama/libggml-cpu-haswell.so
msg="inference compute" id=cpu library=cpu
load_tensors:          CPU model buffer size =  1252.41 MiB
```

**What this means:**
- ✅ GPU is detected (cuInit, cuDeviceGetCount succeed)
- ❌ Ollama loads CPU backend instead of CUDA backend
- ❌ All computation runs on CPU

## Root Cause

**GGML's CUDA backend initialization (`ggml_backend_cuda_init`) is failing silently**, causing Ollama to fall back to CPU.

### What Should Happen

1. Ollama detects GPU (cuInit, cuDeviceGetCount) ✅
2. Ollama tries to load `libggml-cuda.so` (CUDA backend)
3. GGML calls `ggml_backend_cuda_init()` inside `libggml-cuda.so`
4. `ggml_backend_cuda_init()` should:
   - Call `cuInit()` ✅ (succeeds)
   - Call device query functions (cuDeviceGet, cuDeviceGetAttribute, etc.)
   - Create CUDA context
   - Initialize CUBLAS
   - Return success
5. Ollama uses CUDA backend for computation

### What Actually Happens

1. Ollama detects GPU ✅
2. Ollama tries to load `libggml-cuda.so` ✅
3. GGML calls `ggml_backend_cuda_init()` inside `libggml-cuda.so`
4. `ggml_backend_cuda_init()`:
   - Calls `cuInit()` ✅ (succeeds)
   - **FAILS IMMEDIATELY** ❌ (before calling device queries)
   - Returns error
5. Ollama falls back to CPU backend ❌

## Why `ggml_backend_cuda_init` Fails

**The function stops immediately after `cuInit()` succeeds, before calling any device query functions.**

### Possible Reasons

1. **Missing function call**: GGML might call a CUDA function we don't implement
2. **Error check fails**: GGML might check error state after `cuInit()` and fail
3. **Version check fails**: GGML might check driver/runtime version compatibility
4. **Context requirement**: GGML might require a CUDA context to exist
5. **Internal state check**: GGML might check internal state that isn't set correctly

### What We Know

- `cuInit()` is called and succeeds ✅
- `cuDeviceGetCount()` is called (but maybe not by GGML) ✅
- **No device query functions are called by GGML** ❌
- **No Runtime API functions are called by GGML** ❌
- **No error messages logged** ❌

## Solution

To fix this, we need to:

1. **Identify what GGML checks after `cuInit()`**:
   - Check if `cudaGetDevice()` is called
   - Check if `cudaDeviceGetAttribute()` is called
   - Check if `cudaGetDeviceProperties()` is called
   - Check if `cuDevicePrimaryCtxRetain()` is called

2. **Ensure all required functions return success**:
   - All Runtime API functions must work
   - All Driver API functions must work
   - Context functions must work

3. **Verify function implementations**:
   - Check if `cudaGetDevice()` is implemented correctly
   - Check if context functions work
   - Check if all error-checking functions return success

## Current Status

- ✅ **GPU Detection**: Working
- ❌ **CUDA Backend Initialization**: Failing silently
- ❌ **GPU Usage**: Not happening (CPU fallback)
- ⚠️ **Root Cause**: Unknown (GGML init fails after cuInit)

## Next Steps

1. Add logging to `cudaGetDevice()` to see if it's called
2. Check if `cuDevicePrimaryCtxRetain()` is called
3. Verify all Runtime API functions are implemented correctly
4. Check if there are any error messages we're missing
