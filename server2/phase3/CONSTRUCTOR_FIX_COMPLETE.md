# Constructor Fix Complete

## Date: 2026-02-26

## ✅ Success: Constructor is Working!

The Runtime API shim constructor fix is **complete and working correctly**.

### What Was Fixed

1. **LD_PRELOAD Order** ✓
   - Fixed order in `/etc/systemd/system/ollama.service.d/vgpu.conf`
   - Driver API shim (`libvgpu-cuda.so`) now loads BEFORE Runtime API shim (`libvgpu-cudart.so`)
   - Order: `libvgpu-exec.so:libvgpu-syscall.so:libvgpu-cuda.so:libvgpu-nvml.so:libvgpu-cudart.so`

2. **Constructor Finds cuInit()** ✓
   - Uses direct external function call (since both shims are in same process)
   - No longer relies on `dlsym()` which wasn't working
   - Logs show: "cuInit() called directly as external function"

3. **Device Count Functions Called** ✓
   - `cuInit()` called, returns `rc=0` (success)
   - `cuDeviceGetCount()` called, returns `rc=0, count=1`
   - `cudaGetDeviceCount()` called, returns `rc=0, count=1`

### Evidence

```
[libvgpu-cudart] constructor CALLED (initializing Runtime API shim)
[libvgpu-cudart] constructor: cuInit() called directly as external function
[libvgpu-cudart] constructor: cuInit() called, rc=0
[libvgpu-cudart] constructor: cuDeviceGetCount() called, rc=0, count=1
[libvgpu-cudart] constructor: cudaGetDeviceCount() called, rc=0, count=1
[libvgpu-cudart] constructor: Runtime API shim ready
```

## ⚠️ Remaining Issue

**Discovery still shows `initial_count=0`**

Even though:
- Constructor is working ✓
- Device count functions return count=1 ✓
- All functions called successfully ✓

Ollama's discovery still reports `initial_count=0` and uses CPU mode.

### Root Cause

Based on previous documentation:
- `libggml-cuda.so` IS being opened during discovery
- But `ggml_cuda_init()` is FAILING
- This causes discovery to timeout or fail

### Why This Matters

Ollama's discovery:
1. Loads `libggml-cuda.so` ✓
2. Calls `ggml_cuda_init()` ✗ **FAILS!**
3. If initialization fails, discovery doesn't proceed
4. Falls back to CPU mode

## Next Steps

1. **Identify what `ggml_cuda_init()` calls**
   - Check which CUDA functions it needs
   - Verify all required functions are implemented
   - Ensure functions return correct values (not errors)

2. **Fix any missing or failing functions**
   - `cuDeviceGet()` - to get device handle
   - `cuCtxCreate()` - to create CUDA context
   - `cuDeviceGetAttribute()` - to get device properties
   - Other functions that might be called

3. **Verify initialization succeeds**
   - Test `ggml_cuda_init()` manually
   - Check error messages
   - Ensure all prerequisites are met

## Key Achievement

**The constructor fix is working!** Device count functions are being called and returning count=1. Now we need to ensure `libggml-cuda.so` can initialize successfully so Ollama's discovery can proceed.
