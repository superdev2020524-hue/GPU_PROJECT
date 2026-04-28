# Root Cause Found: Driver Version Issue!

## ✅ BREAKTHROUGH!

**Found the actual error messages from strace:**
1. `"ggml_cuda_init: failed to initialize CUDA: API call is not supported in the installed CUDA driver"` (98 bytes)
2. `"ggml_cuda_init: failed to initialize CUDA: CUDA driver version is insufficient for CUDA runtime version"` (104 bytes)

## The Problem

The CUDA runtime (libcudart.so.12) is checking if the driver version is sufficient, and it's failing because:
- We were returning driver version **12080** (12.8.0)
- The runtime expects a higher minimum version

## The Fix

**Increased driver version from 12080 to 12090 (12.9.0)**

This should satisfy the runtime's version check.

## What We Changed

In `gpu_properties.h`:
```c
#define GPU_DEFAULT_DRIVER_VERSION  12090  /* CUDA 12.9 (increased from 12.8) */
```

## Next Steps

1. ✅ Rebuild shim libraries with new version
2. ✅ Restart Ollama service
3. ⏳ Check if error is fixed
4. ⏳ Verify GPU mode activates

## Expected Outcome

After this fix:
- Driver version error should be resolved
- `ggml_backend_cuda_init` should proceed further
- Device query functions should be called
- GPU mode should activate!

## Status

**We found the root cause!** The driver version was too low. With version 12090, the runtime should accept it and initialization should proceed.
