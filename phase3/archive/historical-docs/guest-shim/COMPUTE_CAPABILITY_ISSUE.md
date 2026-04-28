# Compute Capability Issue - compute=0.0

## Date: 2026-02-25 09:32:17

## Problem

Ollama logs show `compute=0.0` when verifying device support:
```
msg="verifying if device is supported" library=/usr/local/lib/ollama/cuda_v12 
description="NVIDIA H100 80GB HBM3" compute=0.0 
id=GPU-00000000-1400-0000-0900-000000000000
```

This causes the device to be filtered out as "didn't fully initialize".

## Root Cause Analysis

### ✅ Major Progress
1. **Discovery timeout fixed**: 331ms (was 30s timeout!)
2. **Initial discovery**: 297ms
3. **No more 30s timeouts**
4. **GPU detected**: H100 80GB HBM3

### ❌ Remaining Issue
- **Runtime API functions NOT called**: Only constructor is called
- **Driver API functions may not be called**: Need to verify
- **compute=0.0**: Should be 9.0

## Investigation Results

### Runtime API Calls
- `cudaGetDeviceCount`: 0 calls
- `cudaGetDevice`: 0 calls
- `cudaDeviceGetAttribute`: 0 calls
- `cudaGetDeviceProperties_v2`: 0 calls
- `constructor`: 3 calls (only this is called)

**Conclusion**: Ollama is NOT using Runtime API for compute capability verification.

### Driver API Calls
Need to check if `cuDeviceGetAttribute` is called with:
- Attribute 75: `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR`
- Attribute 76: `CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR`

## Implementation Status

### Driver API Shim (`libvgpu_cuda.c`)
✅ `cuDeviceGetAttribute` implemented
✅ Handles attributes 75 and 76
✅ Returns `g_gpu_info.compute_cap_major` and `g_gpu_info.compute_cap_minor`
✅ `init_gpu_defaults()` sets these to `GPU_DEFAULT_CC_MAJOR` (9) and `GPU_DEFAULT_CC_MINOR` (0)

### Runtime API Shim (`libvgpu_cudart.c`)
✅ `cudaDeviceGetAttribute` implemented
✅ Handles attributes 75 and 76
✅ Returns `GPU_DEFAULT_CC_MAJOR` (9) and `GPU_DEFAULT_CC_MINOR` (0)
⚠️ **But functions are NOT being called**

## Hypothesis

Ollama may be:
1. **Using Driver API directly** (`cuDeviceGetAttribute`) - Need to verify if called
2. **Getting compute from device properties** - May use `cuDeviceGetProperties` instead
3. **Using a different method** - May check compute capability via NVML or other means
4. **Not calling functions during verification** - May have cached values or use different path

## Next Steps

1. **Verify Driver API calls**: Check if `cuDeviceGetAttribute` is called with attributes 75/76
2. **Check `g_gpu_info` initialization**: Ensure `init_gpu_defaults()` is called before verification
3. **Check `cuDeviceGetProperties`**: Ollama may use this instead of `cuDeviceGetAttribute`
4. **Add more logging**: Log all Driver API calls during verification phase
5. **Check NVML**: Ollama may use `nvmlDeviceGetCudaComputeCapability` instead

## Key Files

- `libvgpu_cuda.c`: Driver API shim, `cuDeviceGetAttribute` at line 2758-2761
- `libvgpu_cudart.c`: Runtime API shim, `cudaDeviceGetAttribute` at line 282-285
- `gpu_properties.h`: Default values, `GPU_DEFAULT_CC_MAJOR=9`, `GPU_DEFAULT_CC_MINOR=0`

## Expected Behavior

When Ollama verifies device support, it should:
1. Call `cuDeviceGetAttribute(pi, 75, 0)` → returns 9
2. Call `cuDeviceGetAttribute(pi, 76, 0)` → returns 0
3. Calculate compute as 9.0
4. Accept device as supported

## Current Behavior

1. ❌ Functions may not be called
2. ❌ Or functions return 0 instead of 9/0
3. ❌ compute=0.0 → device filtered

## Status

**Progress: 85% Complete**
- ✅ Discovery working (331ms)
- ✅ GPU detected
- ✅ Library loading working
- ⚠️ Compute capability verification failing (compute=0.0)
