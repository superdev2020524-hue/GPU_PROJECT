# Driver Version Increased to 13.0

## Current Status

**Found both errors:**
1. "API call is not supported in the installed CUDA driver" (98 bytes)
2. "CUDA driver version is insufficient for CUDA runtime version" (104 bytes)

## The Fix

**Increased driver version from 12090 (12.9) to 13000 (13.0)**

This should satisfy the runtime's version check. The CUDA runtime (libcudart.so.12) checks if the driver version is sufficient, and version 13.0 should be high enough.

## What We Changed

In `gpu_properties.h`:
```c
#define GPU_DEFAULT_DRIVER_VERSION  13000  /* CUDA 13.0 (increased from 12.9) */
```

## Expected Outcome

After this fix:
- Driver version error should be resolved
- "API call is not supported" error might also be resolved (if it was version-related)
- `ggml_backend_cuda_init` should proceed further
- Device query functions should be called
- GPU mode should activate!

## Status

**We've increased the driver version to 13.0.** This should be sufficient for CUDA 12.x runtime. If errors persist, we may need to investigate the "API call is not supported" error more deeply, as it might indicate a missing function rather than a version issue.
