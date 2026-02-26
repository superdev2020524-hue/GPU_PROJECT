# Constructor Fix Attempt

## Date: 2026-02-26

## Problem Identified

The Runtime API shim constructor is not completing because it cannot find `cuInit()` from the Driver API shim.

### Root Cause

1. **Both shims are in LD_PRELOAD** ✓
2. **cuInit() IS exported and visible** ✓ (verified via manual test)
3. **But constructor runs too early** ✗

The issue is the **order of loading**:
- LD_PRELOAD: `libvgpu-exec.so:libvgpu-nvml.so:libvgpu-cudart.so:libvgpu-cuda.so`
- Runtime API shim (`libvgpu-cudart.so`) loads BEFORE Driver API shim (`libvgpu-cuda.so`)
- When Runtime API shim constructor runs, Driver API shim's symbols are not yet in global scope

### Fix Attempted

Modified constructor to try multiple methods:
1. `dlsym(RTLD_DEFAULT, "cuInit")` - Failed (symbol not in scope yet)
2. `dlopen("/usr/lib64/libvgpu-cuda.so", RTLD_NOLOAD)` - Failed (library not loaded yet)
3. `dlopen("/usr/lib64/libvgpu-cuda.so", RTLD_LAZY)` - Failed (creates new instance, not the LD_PRELOAD one)

### Current Status

- ✅ New code deployed and running
- ✅ Constructor completes (logs "Runtime API shim ready")
- ❌ But cuInit() and device count functions are NOT called
- ❌ Device count remains 0
- ❌ Backend scanner still skips CUDA backend

## Next Steps

1. **Change LD_PRELOAD order** - Put Driver API shim BEFORE Runtime API shim
2. **Or delay constructor** - Use lazy initialization instead of constructor
3. **Or use different approach** - Ensure device count functions work when called, not proactively

## Key Insight

**The constructor approach won't work if the Driver API shim loads after the Runtime API shim.** We need to either:
- Change the loading order
- Use lazy initialization
- Or ensure device count functions work when libggml-cuda.so calls them directly
