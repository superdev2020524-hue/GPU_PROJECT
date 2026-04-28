# Root Cause and Solution

## Root Cause Identified

From log analysis:
- ✅ `cuInit()` is called and succeeds
- ✅ Runtime API shim is loaded (constructor called)
- ❌ `cudaRuntimeGetVersion()` is NOT being called
- ❌ `libggml-cuda.so` does NOT link to `libcudart` at link time
- ❌ `libcudart` is NOT loaded at runtime

**Conclusion**: `cudaRuntimeGetVersion()` is never called because `libggml-cuda.so` doesn't have access to the Runtime API library.

## Why This Happens

`libggml-cuda.so` is compiled to use CUDA Driver API directly, not Runtime API. It may:
1. Not link to `libcudart` at all
2. Load `libcudart` dynamically only if needed
3. Use Driver API functions exclusively

Since `cudaRuntimeGetVersion()` is a Runtime API function, it's never called if `libggml-cuda.so` doesn't load `libcudart`.

## Solution Options

### Option 1: Ensure libcudart is Loaded (Recommended)

Make sure `libggml-cuda.so` can find and load `libcudart`:

1. **Check if libggml-cuda.so uses dlopen() for libcudart**:
   - If yes, ensure the path includes our shim location
   - Create symlinks in the expected location

2. **Ensure LD_LIBRARY_PATH includes our shim**:
   - Already done in systemd config
   - Verify it's working

3. **Replace actual libcudart.so.12.8.90 with our shim**:
   - Backup original: `/usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90`
   - Create symlink: `libcudart.so.12.8.90 -> /usr/lib64/libvgpu-cudart.so`

### Option 2: Force Load libcudart

If `libggml-cuda.so` doesn't load `libcudart` automatically, we can force it:

1. **Use LD_PRELOAD to preload libcudart**:
   - Add `libvgpu-cudart.so` to LD_PRELOAD
   - This ensures it's loaded before `libggml-cuda.so`

2. **Use constructor to force load**:
   - In our Runtime API shim constructor, call `dlopen("libcudart.so.12", RTLD_GLOBAL)`
   - This makes it available globally

### Option 3: Check if ggml_backend_cuda_init Actually Needs Runtime API

Maybe `ggml_backend_cuda_init` doesn't actually call `cudaRuntimeGetVersion()`. It might:
- Only use Driver API functions
- Check runtime version through Driver API
- Not check runtime version at all

If this is the case, we need to focus on ensuring Driver API device queries work instead.

## Recommended Next Steps

1. **Check if libggml-cuda.so loads libcudart dynamically**:
   ```bash
   strace -e trace=openat,open ollama list 2>&1 | grep cudart
   ```

2. **Force preload libcudart in systemd**:
   - Add `libvgpu-cudart.so` to LD_PRELOAD (already done, but verify order)

3. **Replace libcudart.so.12.8.90 with symlink**:
   - This ensures if `libggml-cuda.so` looks for the specific version, it finds our shim

4. **Check if device queries work without Runtime API**:
   - Maybe `ggml_backend_cuda_init` uses Driver API exclusively
   - Focus on ensuring `cuDeviceGetCount()`, `cuDeviceGetAttribute()` work

## Current Status

- ✅ All fixes implemented and deployed
- ✅ Version compatibility logic in place
- ✅ Proactive device count initialization in place
- ⚠️ `cudaRuntimeGetVersion()` not being called (library not loaded)
- ⚠️ Device query functions not being called (need to investigate why)

## Next Action

Since `cudaRuntimeGetVersion()` isn't being called, we should:
1. Focus on ensuring Driver API device queries work
2. Check if `cuDeviceGetCount()` and `cuDeviceGetAttribute()` are being called
3. If not, investigate why `ggml_backend_cuda_init` isn't calling them

The Runtime API fixes are in place and will work once `libcudart` is loaded, but the immediate priority is ensuring device queries via Driver API work.
