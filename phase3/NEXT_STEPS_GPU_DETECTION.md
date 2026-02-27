# Next Steps: GPU Detection and Backend Selection

## Current Status

✅ **Working:**
- Model file loading (fixed `fopen()` interception)
- Model inference (produces correct output)
- CUDA backend library loads (`libggml-cuda.so`)
- `cuInit()` and `cuDeviceGetCount()` are called and succeed
- Driver API shim intercepts calls correctly

❌ **Not Working:**
- `cuDeviceGet()` is **never called** - this is the blocker
- `cuDeviceGetAttribute()` is **never called** - device validation doesn't happen
- `cuCtxCreate()`, `cuMemAlloc()`, etc. are **never called** - no GPU compute
- Ollama reports `library=cpu` even though CUDA backend is loaded

## Root Cause Analysis

The CUDA backend (`libggml-cuda.so`) is loaded, but GGML's device validation is not proceeding. The sequence should be:

1. ✅ `cuInit()` - called and succeeds
2. ✅ `cuDeviceGetCount()` - called, returns 1, succeeds
3. ❌ `cuDeviceGet()` - **NOT CALLED** (this is where it stops)
4. ❌ `cuDeviceGetAttribute()` - **NOT CALLED**
5. ❌ `cuCtxCreate()` - **NOT CALLED**
6. ❌ `cuMemAlloc()` - **NOT CALLED**

## Why `cuDeviceGet()` Might Not Be Called

Possible reasons:
1. **Early return in GGML**: GGML might be checking something before calling `cuDeviceGet()` that's failing
2. **Backend selection logic**: Ollama might be selecting CPU backend before GGML validates the device
3. **Missing function exports**: `cuDeviceGet()` might not be properly exported from our shim
4. **Runtime API path**: GGML might be using Runtime API (`cudaGetDevice()`) instead of Driver API (`cuDeviceGet()`)

## Investigation Steps

1. **Check Runtime API calls**: Look for `cudaGetDevice()`, `cudaGetDeviceProperties()`, `cudaDeviceGetAttribute()` calls
2. **Verify function exports**: Ensure all required Driver API functions are exported from `libvgpu-cuda.so`
3. **Check GGML initialization**: Look for GGML-specific initialization logs
4. **Verify backend selection**: Understand Ollama's backend selection logic

## Potential Solutions

### Solution 1: Ensure Runtime API Functions Are Called
If GGML uses Runtime API instead of Driver API:
- Verify `cudaGetDevice()`, `cudaGetDeviceProperties()` are implemented in `libvgpu-cudart.so`
- Ensure Runtime API shim properly calls Driver API shim

### Solution 2: Force Device Validation
- Add explicit device validation in shim constructor
- Call `cuDeviceGet()` and `cuDeviceGetAttribute()` during initialization
- Log all device queries to understand what GGML expects

### Solution 3: Check Backend Selection
- Investigate why Ollama reports `library=cpu` even with CUDA backend loaded
- May need to ensure all validation passes before backend selection

## Next Actions

1. Check if Runtime API functions (`cudaGetDevice()`, `cudaGetDeviceProperties()`) are being called
2. Verify all required Driver API functions are exported and callable
3. Add more detailed logging to understand GGML's initialization sequence
4. Investigate Ollama's backend selection logic
