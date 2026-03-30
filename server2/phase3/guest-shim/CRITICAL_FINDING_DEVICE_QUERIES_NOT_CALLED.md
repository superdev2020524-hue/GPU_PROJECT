# Critical Finding: Device Query Functions Not Being Called

## Current Status

✅ **What's Working**:
- `cuInit()` is called and succeeds
- All shims are loaded correctly
- Functions are properly exported

❌ **What's NOT Working**:
- `cuGetProcAddress()` is NOT being called (ggml_backend_cuda_init uses direct linking)
- `cudaRuntimeGetVersion()` is NOT being called (libcudart not loaded)
- `cuDeviceGetCount()` is NOT being called
- `cuDeviceGetAttribute()` is NOT being called
- `cuDeviceGetProperties()` is NOT being called
- Still showing `initial_count=0` and `library=cpu`

## Root Cause Analysis

`ggml_backend_cuda_init` is **failing silently** after `cuInit()` succeeds but before it calls any device query functions.

### Why This Happens

Since `cuGetProcAddress()` is not being called, `ggml_backend_cuda_init` uses **direct linking** to resolve CUDA functions. This means:

1. Functions must be available at link time
2. Functions must be properly exported
3. Functions must be found via the dynamic linker

### Possible Failure Points

1. **Error checking after cuInit()**:
   - `ggml_backend_cuda_init` might call `cuGetErrorString()` or `cuGetLastError()`
   - If these return errors, initialization fails
   - We added logging but haven't seen calls yet

2. **Version compatibility check**:
   - `ggml_backend_cuda_init` might check driver/runtime version compatibility
   - If versions don't match expected range, initialization fails
   - We return driver=12090, runtime=12080 (should be compatible)

3. **Function not found via direct linking**:
   - A required function might not be exported or found
   - Dynamic linker can't resolve the symbol
   - Initialization fails silently

4. **Internal error in ggml_backend_cuda_init**:
   - Some internal check or initialization step fails
   - No error is logged or propagated
   - Initialization stops before device queries

## Next Steps

### Step 1: Check Error Function Calls

Verify if `cuGetErrorString()` or `cuGetLastError()` are being called:

```bash
sudo journalctl -u ollama -n 500 --no-pager | grep -i "cuGetError"
```

If these are called, check what error codes they're returning.

### Step 2: Check Version Compatibility

Verify if `cuDriverGetVersion()` is being called and what version is returned:

```bash
sudo journalctl -u ollama -n 500 --no-pager | grep -i "cuDriverGetVersion"
```

Ensure the returned version is compatible with what `ggml_backend_cuda_init` expects.

### Step 3: Verify Function Exports

Check if all required functions are properly exported:

```bash
nm -D /usr/lib64/libvgpu-cuda.so | grep -E "(cuDeviceGetCount|cuDeviceGetAttribute|cuDeviceGetProperties)"
```

### Step 4: Check Direct Linking

Verify if `libggml-cuda.so` can find our functions via direct linking:

```bash
ldd /usr/local/lib/ollama/libggml-cuda.so | grep libcuda
objdump -p /usr/local/lib/ollama/libggml-cuda.so | grep NEEDED
```

### Step 5: Add More Logging

If the above don't reveal the issue, add logging to:
- `cuGetErrorString()` - to see if error checking is happening
- `cuGetLastError()` - to see if error checking is happening
- `cuDriverGetVersion()` - to see if version check is happening
- All device query functions - to see if they're being called but logs aren't appearing

## Recommended Immediate Action

**Check if error checking functions are being called**. This is the most likely cause - `ggml_backend_cuda_init` might be checking for errors after `cuInit()` and failing if it finds any.

We already added logging to `cuGetErrorString()`, so check the logs to see if it's being called and what error codes it's seeing.

## Expected Outcome

Once we identify why `ggml_backend_cuda_init` is failing after `cuInit()`, we can:
1. Fix the specific issue (error check, version check, function not found, etc.)
2. Ensure device query functions are called
3. Verify GPU detection works (`library=cuda`, `compute=9.0`)
