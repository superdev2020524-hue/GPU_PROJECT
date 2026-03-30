# Constructor Fix Summary

## Date: 2026-02-26

## Problem

Runtime API shim constructor could not find `cuInit()` from Driver API shim, even after fixing LD_PRELOAD order.

## Root Cause

Even though:
- ✅ LD_PRELOAD order is correct (Driver API shim loads before Runtime API shim)
- ✅ cuInit() is exported from Driver API shim
- ✅ Driver API shim is loaded

The constructor still could not find cuInit() via:
- `dlsym(RTLD_DEFAULT, "cuInit")` - Failed
- `dlopen("/usr/lib64/libvgpu-cuda.so", RTLD_LAZY | RTLD_NOLOAD)` - Failed
- `dlopen("/usr/lib64/libvgpu-cuda.so", RTLD_LAZY)` - Failed

This suggests a symbol visibility issue - symbols from LD_PRELOAD libraries might not be visible to `dlsym(RTLD_DEFAULT)`.

## Fix Applied

Updated constructor to:
1. **First try to use function pointer from `init_driver_api_functions()`** - This function already tries to find cuInit() and stores it in `real_cuInit`
2. **If that's NULL, try RTLD_DEFAULT** - Fallback method
3. **If that fails, try explicit dlopen with RTLD_GLOBAL** - RTLD_GLOBAL makes symbols globally visible

### Code Changes

```c
/* Method 1: Use function pointer from init_driver_api_functions() if available */
if (real_cuInit) {
    cuInit_func = real_cuInit;
    // Log success
} else {
    /* Method 2: Try RTLD_DEFAULT */
    cuInit_func = (cuInit_func_t)dlsym(RTLD_DEFAULT, "cuInit");
    if (!cuInit_func) {
        /* Method 3: Try explicit dlopen with RTLD_GLOBAL */
        void *shim_handle = dlopen("/usr/lib64/libvgpu-cuda.so", RTLD_LAZY | RTLD_GLOBAL);
        if (shim_handle) {
            cuInit_func = (cuInit_func_t)dlsym(shim_handle, "cuInit");
        }
    }
}
```

## Status

✅ Constructor code updated
✅ Shim rebuilt and deployed
⏳ Awaiting verification that constructor now finds and calls cuInit()

## Expected Results

After this fix:
1. Constructor uses `real_cuInit` from `init_driver_api_functions()` if available
2. Or finds cuInit() via RTLD_GLOBAL dlopen
3. Calls cuInit() and device count functions
4. Sets device count to 1
5. Enables GPU mode
