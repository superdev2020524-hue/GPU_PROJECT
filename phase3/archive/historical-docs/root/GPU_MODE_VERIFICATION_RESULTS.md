# GPU Mode Verification Results

## Date: 2026-02-26

## Current Status

### ✅ Working
- Device discovery: VGPU-STUB found at 0000:00:05.0
- GPU defaults applied: H100 80GB CC=9.0 VRAM=81920 MB
- cuInit() succeeds with device found
- Symlinks in place: `/usr/local/lib/ollama/cuda_v12/libcuda.so.1` → `/usr/lib64/libvgpu-cuda.so`
- `cuDeviceGetCount()` is exported and implemented correctly

### ❌ Issue
- Runner subprocess reports `initial_count=0`
- Ollama uses CPU mode (`library=cpu`)
- `cuDeviceGetCount()` is NOT being called in runner subprocess

## Verification Results

### Symlinks
```
✓ /usr/local/lib/ollama/cuda_v12/libcuda.so.1 → /usr/lib64/libvgpu-cuda.so
✓ /usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90 → /usr/lib64/libvgpu-cudart.so
✓ Symlinks correctly resolve to our shims
```

### Symbol Exports
```
✓ cuDeviceGetCount is exported (nm shows: 000000000000ea73 T cuDeviceGetCount)
✓ cuDeviceGetCount_v2 is exported
```

### Library Dependencies
```
libggml-cuda.so depends on:
  libcudart.so.12 => /usr/local/lib/ollama/cuda_v12/libcudart.so.12
  libcuda.so.1 => /usr/local/lib/ollama/cuda_v12/libcuda.so.1
```

### cuDeviceGetCount() Implementation
```c
CUresult cuDeviceGetCount(int *count)
{
    /* Logs when called */
    syscall(__NR_write, 2, "[libvgpu-cuda] cuDeviceGetCount() CALLED (pid=%d)\n", ...);
    
    /* Always returns count=1 */
    *count = 1;
    
    /* Logs success */
    syscall(__NR_write, 2, "[libvgpu-cuda] cuDeviceGetCount() SUCCESS: returning count=1 (pid=%d)\n", ...);
    
    return CUDA_SUCCESS;
}
```

## The Problem

**The runner subprocess is NOT calling our `cuDeviceGetCount()` implementation.**

Even though:
- Symlinks are correct
- `cuDeviceGetCount()` is exported
- Implementation is correct

The runner subprocess still reports `initial_count=0`, which means it's calling the real CUDA library, not our shim.

## Possible Causes

1. **Runner subprocess doesn't load our shim**
   - Even though symlinks are in place, the runner might load libraries differently
   - The runner might have a different LD_LIBRARY_PATH
   - The runner might load libraries before symlinks are resolved

2. **Library loading order**
   - The runner might load `libggml-cuda.so` which then loads `libcuda.so.1`
   - If `libggml-cuda.so` was linked against a specific version, it might not follow symlinks

3. **Different code path**
   - The runner might use a different API to get device count
   - The runner might check device count before loading CUDA libraries

## Next Steps

1. **Verify runner subprocess loads our shim**
   - Check process maps when runner is running
   - Verify `libvgpu-cuda.so` is in the process memory

2. **Check library loading order**
   - Use `strace` or `ltrace` to see what libraries the runner loads
   - Verify the order of library loading

3. **Force shim loading**
   - Ensure LD_PRELOAD is set in runner subprocess
   - Or ensure symlinks are followed correctly

4. **Check if different API is used**
   - The runner might use `cudaGetDeviceCount()` (Runtime API) instead of `cuDeviceGetCount()` (Driver API)
   - Check if Runtime API functions are being called

## Conclusion

**Device discovery is working, but GPU mode is not active because the runner subprocess is not using our shim.** The symlinks are correct, but the runner subprocess is still calling the real CUDA library which returns 0 devices. We need to verify that the runner subprocess actually loads our shim and calls our `cuDeviceGetCount()` implementation.
