# Segfault Issue After Device Discovery

## Date: 2026-02-25

## Status

✅ **Device Discovery**: WORKING
✅ **GPU Detection**: WORKING
⚠ **Segfault**: Occurs after device discovery (separate issue)

## Problem

After device discovery succeeds, Ollama crashes with a segfault. The segfault happens right after `fopen()` is called for `/sys/bus/pci/devices/0000:00:05.0/vendor`.

## What's Working

- ✅ Device discovery finds VGPU-STUB correctly
- ✅ Real values are read: 0x10de, 0x2331, 0x030200
- ✅ GPU defaults applied: H100 80GB CC=9.0
- ✅ `device_found=1`
- ✅ `cuInit()` succeeds with device found

## Segfault Pattern

```
[libvgpu-cuda] fopen() INTERCEPTOR CALLED: /sys/bus/pci/devices/0000:00:05.0/vendor
[libvgpu-cuda] fopen() called: /sys/bus/pci/devices/0000:00:05.0/vendor, skip_flag=0
SEGV - core-dump
```

## Possible Causes

1. **Mutex Initialization**: Fixed (changed from `PTHREAD_MUTEX_INITIALIZER` to lazy initialization)
2. **fprintf/fflush**: May be called when stderr is not available
3. **NULL Pointer**: May be dereferenced in fopen() interceptor
4. **dlsym/dlopen**: May fail or return invalid pointer

## Fixes Applied

1. ✅ Changed `g_skip_flag_mutex` from `PTHREAD_MUTEX_INITIALIZER` to lazy initialization
2. ✅ Added `ensure_skip_flag_mutex_init()` function
3. ✅ Added checks for mutex initialization before use

## Next Steps

1. Check if segfault is in fprintf/fflush calls
2. Add NULL pointer checks
3. Use `write()` syscall instead of fprintf for critical debug messages
4. Check if dlsym/dlopen are causing issues

## Impact

The segfault doesn't affect device discovery - it happens AFTER discovery succeeds. However, it prevents Ollama from running, so it needs to be fixed for full functionality.

## Workaround

Device discovery is working correctly. The segfault is a separate issue that needs investigation but doesn't prevent the core functionality (device discovery) from working.
