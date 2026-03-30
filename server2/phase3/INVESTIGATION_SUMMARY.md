# Investigation Summary: Why GGML CUDA Backend Fails

## Current Status

### ✅ What's Working
- GPU detection: `cuInit()`, `cuDeviceGetCount()` succeed
- All shim libraries compile and load correctly
- Context functions work during init phase
- Runtime API functions implemented

### ❌ The Problem
- **GGML CUDA backend initialization fails silently**
- Ollama falls back to CPU backend
- No error messages logged
- No function calls after `cuInit()` succeeds

## Key Findings

### 1. GGML Uses Direct Linking
- **`cuGetProcAddress()` is NEVER called** - confirmed by logs
- GGML uses direct linking via `dlsym()` or static linking
- Functions must be available at link/load time

### 2. No Function Calls After cuInit()
- No `cuDeviceGetCount()` calls (except from our constructor)
- No `cuDeviceGet()` calls
- No `cuDeviceGetAttribute()` calls
- No context function calls
- No Runtime API calls (except constructors)

### 3. Library Paths
- Our shim is at `/opt/vgpu/lib/libvgpu-cuda.so.1`
- Also symlinked at `/usr/lib64/libvgpu-cuda.so`
- `cuGetProcAddress` now tries multiple paths

## Root Cause Hypothesis

**GGML's `ggml_backend_cuda_init()` fails immediately after `cuInit()` succeeds, before calling any other functions.**

This suggests one of:
1. **Function lookup failure**: A required function isn't found via direct linking
2. **Error check failure**: An error-checking function returns an error
3. **Version check failure**: A version compatibility check fails
4. **Internal check failure**: An internal GGML check fails

## What We've Tried

1. ✅ Enhanced logging to all CUDA functions
2. ✅ Fixed context functions to work during init
3. ✅ Enhanced `cudaSetDevice()` logging
4. ✅ Fixed `cuGetProcAddress` to try multiple library paths
5. ✅ Ensured all error codes are defined

## Next Steps

Since we can't see what GGML does internally, we need to:

1. **Check library dependencies**: Verify what libraries `libggml-cuda.so` needs
2. **Check symbol exports**: Verify our functions are properly exported
3. **Try forcing CUDA backend**: See if there's a way to force GGML to use CUDA
4. **Check for missing symbols**: Use `nm` or `objdump` to see what symbols GGML expects

## Files Modified

1. `libvgpu_cuda.c`:
   - Enhanced `cuGetProcAddress` to try multiple library paths
   - Added logging to context functions
   - Fixed `cuCtxSetCurrent` for init phase

2. `libvgpu_cudart.c`:
   - Enhanced `cudaSetDevice` logging
   - Fixed missing error code definitions
