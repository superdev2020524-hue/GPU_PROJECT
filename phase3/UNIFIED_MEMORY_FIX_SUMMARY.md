# Unified Memory API Fix Summary

## Problem
Ollama was crashing with "exit status 2" after CUDA backend initialization. The crash was caused by unified memory APIs (`cuMemCreate`, `cuMemMap`, `cuMemRelease`) returning dummy handles without actually allocating memory.

## Solution
Modified `libvgpu_cuda.c` to:
1. **`cuMemCreate`**: Actually allocate memory using `cuMemAlloc_v2()` via transport, return device pointer as handle
2. **`cuMemMap`**: Validate handle and return success (mapping already done via allocation)
3. **`cuMemRelease`**: Free memory using `cuMemFree_v2()` via transport

## Changes Made

### `cuMemCreate` (lines 3343-3375)
- **Before**: Returned dummy handle `0x1000`
- **After**: Calls `cuMemAlloc_v2()` to actually allocate memory on physical GPU
- Returns device pointer as handle (valid since handles are opaque)

### `cuMemMap` (lines 3433-3465)
- **Before**: Always returned success without validation
- **After**: Validates handle is non-zero, returns success (mapping already done)

### `cuMemRelease` (lines 3447-3477)
- **Before**: Always returned success without freeing
- **After**: Calls `cuMemFree_v2()` to actually free memory on physical GPU

## Deployment Status

✅ **Code updated**: `libvgpu_cuda.c` modified with proper transport calls
✅ **Library rebuilt**: `libvgpu-cuda.so.1` compiled successfully
✅ **Library installed**: Updated in both `/usr/lib64/` and `/opt/vgpu/lib/`
⚠️ **Testing**: Ollama service currently failing to start (exit code 127 - investigating)

## Next Steps

1. Fix Ollama service startup issue (exit code 127)
2. Verify unified memory APIs work correctly
3. Test that Ollama no longer crashes
4. Monitor logs for `cuMemCreate() SUCCESS: handle=0x... (allocated ... bytes)`

## Expected Behavior

**Before fix:**
```
[libvgpu-cuda] cuMemCreate returning SUCCESS (dummy handle)
[libvgpu-cuda] cuMemMap returning SUCCESS
[libvgpu-cuda] cuMemRelease returning SUCCESS
→ Crash: exit status 2
```

**After fix:**
```
[libvgpu-cuda] cuMemCreate() CALLED (handle=..., size=16777216, ...)
[libvgpu-cuda] cuMemAlloc_v2() CALLED (size=16777216, ...)
[libvgpu-cuda] cuMemAlloc_v2() SUCCESS: ptr=0x..., size=16777216
[libvgpu-cuda] cuMemCreate() SUCCESS: handle=0x... (allocated 16777216 bytes)
[libvgpu-cuda] cuMemMap() SUCCESS: mapped ptr=0x... to handle=0x...
[libvgpu-cuda] cuMemRelease() SUCCESS: freed handle=0x...
→ No crash, Ollama continues
```
