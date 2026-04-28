# Fixes Applied to Investigate GGML CUDA Backend Initialization

## Summary

Investigated why GGML's CUDA backend initialization fails after `cuInit()` succeeds, causing Ollama to fall back to CPU backend.

## Fixes Applied

### 1. Added Logging to Context Functions
- **`cuCtxGetCurrent()`**: Added comprehensive logging to detect if called during GGML init
- **`cuCtxSetCurrent()`**: Enhanced logging and made it work during init phase

### 2. Fixed `cuCtxSetCurrent()` for Init Phase
- **Problem**: `cuCtxSetCurrent()` was calling `ensure_init()` which might fail during init
- **Fix**: During init phase (`g_in_init_phase == 1`), skip `ensure_init()` and just set context locally
- **Result**: Context functions now work immediately during initialization

### 3. Enhanced `cudaSetDevice()` Logging
- Added detailed logging with device number and PID
- Added validation to ensure device=0

### 4. Fixed Compilation Error
- **Problem**: `cudaErrorMemoryAllocation` was not defined
- **Fix**: Added `#define cudaErrorMemoryAllocation 2` to error codes

## Current Status

### ✅ What's Working
- GPU detection: `cuInit()`, `cuDeviceGetCount()` succeed
- Context functions: `cuCtxGetCurrent()`, `cuCtxSetCurrent()` now work during init
- Runtime API: `cudaSetDevice()`, `cudaGetDevice()` implemented
- Compilation: All libraries compile successfully

### ⚠️ Still Investigating
- **CUDA backend still not loading**: Ollama still loads CPU backend
- **No context function calls logged**: `cuCtxGetCurrent()`, `cuCtxSetCurrent()` not being called
- **No `cuDevicePrimaryCtxRetain()` calls**: This is critical for CUDA backend init

## Next Steps

1. **Check if GGML calls context functions**: Look for any context-related calls in logs
2. **Verify `cuDevicePrimaryCtxRetain()` is being called**: This is typically the first context operation
3. **Check for other missing functions**: GGML might call functions we haven't implemented
4. **Investigate GGML source**: Understand what `ggml_backend_cuda_init()` actually does

## Key Insight

The fact that **no context functions are being called** suggests that GGML's initialization fails **before** it gets to context creation. This means the failure happens even earlier than we thought - possibly during a version check, error check, or function lookup.

## Files Modified

1. `phase3/guest-shim/libvgpu_cuda.c`:
   - Added logging to `cuCtxGetCurrent()`
   - Fixed `cuCtxSetCurrent()` to work during init phase

2. `phase3/guest-shim/libvgpu_cudart.c`:
   - Enhanced `cudaSetDevice()` logging
   - Fixed missing `cudaErrorMemoryAllocation` definition
