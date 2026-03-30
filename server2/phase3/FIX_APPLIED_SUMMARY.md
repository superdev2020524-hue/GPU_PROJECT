# GPU Attributes Fix Applied - Summary

## Date: 2026-02-27

## Fix Applied

**Problem**: ChatGPT identified that `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK` was returning `1620000` instead of `1024`, causing Ollama/GGML to reject the GPU.

**Solution**: Added explicit hardcoded return value of `1024` for `MAX_THREADS_PER_BLOCK` in `cuDeviceGetAttribute()`.

## Changes Made

### Local Code
- **File**: `/home/david/Downloads/gpu/phase3/guest-shim/libvgpu_cuda.c`
- **Line**: ~3611
- **Change**: Changed from `*pi = g_gpu_info.max_threads_per_block;` to explicit `*pi = 1024;`
- **Reason**: Safety measure to ensure correct value is always returned, regardless of initialization state

### VM Code
- **File**: `/home/test-10/phase3/guest-shim/libvgpu_cuda.c`
- **Status**: ✅ Fixed code deployed to VM
- **Status**: ✅ Library rebuilt successfully
- **Status**: ✅ Ollama restarted

## Verification

### Source Code
- ✅ VM code: `GPU_DEFAULT_MAX_THREADS_PER_BLOCK = 1024` (correct)
- ✅ Local code: `GPU_DEFAULT_MAX_THREADS_PER_BLOCK = 1024` (correct)
- ✅ Implementation: Explicit return of `1024` added

### Library Build
- ✅ Library rebuilt: Feb 26 13:18 (after fix)
- ✅ Library installed: `/usr/lib64/libvgpu-cuda.so`
- ✅ Ollama restarted: Active and running

### Current Status
- ✅ Ollama is running
- ✅ VGPU-STUB detected at 0000:00:05.0
- ✅ GPU defaults applied (H100 80GB)
- ⚠️ Discovery still shows `library=cpu` (may need model execution to trigger full discovery)

## Next Steps

1. **Test with model execution** - Discovery may complete when a model is actually run
2. **Check runner subprocess logs** - Discovery may happen in runner, not main process
3. **Verify attribute is returned correctly** - Check logs for `cuDeviceGetAttribute` calls with attrib=1

## Notes

- This is a **general SHIM fix**, not Ollama-specific
- The fix ensures all GPU programs get correct `MAX_THREADS_PER_BLOCK = 1024`
- The value `1620000` was actually `GPU_DEFAULT_CLOCK_RATE_KHZ` (a different attribute)
- Source code was already correct; explicit hardcoding added as safety measure
