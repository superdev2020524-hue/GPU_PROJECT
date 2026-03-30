# VM Investigation Summary - GPU Attributes Fix

## Date: 2026-02-27

## Investigation Results

### ✅ Source Code is CORRECT

**VM Code:**
- `/home/test-10/phase3/guest-shim/gpu_properties.h` line 44: `#define GPU_DEFAULT_MAX_THREADS_PER_BLOCK   1024` ✅
- `/home/test-10/phase3/guest-shim/libvgpu_cuda.c` line 2901: `g_gpu_info.max_threads_per_block = GPU_DEFAULT_MAX_THREADS_PER_BLOCK;` ✅
- `/home/test-10/phase3/guest-shim/libvgpu_cuda.c` line 3611: `*pi = g_gpu_info.max_threads_per_block;` ✅

**Local Code:**
- `/home/david/Downloads/gpu/phase3/guest-shim/gpu_properties.h` line 44: `#define GPU_DEFAULT_MAX_THREADS_PER_BLOCK   1024` ✅
- Implementation matches VM code ✅

### Library Status

- **Library location**: `/usr/lib64/libvgpu-cuda.so`
- **Build date**: Feb 26 10:39:23
- **Source code modified**: Feb 26 10:16:40 (before library build)
- **Library was built AFTER source code modification** ✅

### The 1620000 Value

- **1620000 is `GPU_DEFAULT_CLOCK_RATE_KHZ`** (clock rate in kHz)
- This is a **different attribute** (CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13)
- **NOT related to MAX_THREADS_PER_BLOCK** (attribute = 1)

### Code Comparison

**VM vs Local:**
- `gpu_properties.h`: Only difference is comment about driver version (12.9 vs 13.0), value is same
- `libvgpu_cuda.c`: Need to compare full files, but MAX_THREADS_PER_BLOCK handling looks identical

## Conclusion

**The source code is CORRECT on both VM and local.**

If ChatGPT's analysis showed `MAX_THREADS_PER_BLOCK = 1620000`, possible causes:
1. **Library needs rebuild** - Even though build date is after source, maybe it wasn't rebuilt properly
2. **Runtime bug** - Maybe there's a bug where wrong value is returned
3. **Different code path** - Maybe a different function is being called
4. **Test was run on old library** - Maybe the test was done before the fix

## Recommended Actions

1. **Rebuild library on VM** to ensure latest code is compiled
2. **Test with actual Ollama** to see what value it gets
3. **Verify attribute mapping** - Ensure attribute ID 1 maps to MAX_THREADS_PER_BLOCK correctly
4. **Add explicit check** - Hardcode return value of 1024 for MAX_THREADS_PER_BLOCK as safety measure

## Next Steps

1. Sync local code with VM (VM is source of truth)
2. Ensure both have correct values
3. Rebuild library on VM
4. Test with Ollama
5. If still wrong, add explicit hardcoded return value
