# Direct Linking Analysis - Why dlsym Interception Doesn't Work

## Key Discovery

**libggml-cuda.so uses DIRECT linking to libcuda.so.1, NOT dlsym()**

```
libcuda.so.1 => /usr/lib64/libcuda.so.1
```

This means:
- Symbols are resolved at **load time** by the dynamic linker
- NOT via `dlsym()` calls at runtime
- Our `dlsym()` interceptor **will not catch** these function calls

## Why dlsym Interception Doesn't Work

1. **libggml-cuda.so** is compiled with direct linkage to `libcuda.so.1`
2. When it loads, the dynamic linker resolves CUDA symbols to our shim (via LD_PRELOAD)
3. But it doesn't call `dlsym()` - symbols are already resolved
4. Our `dlsym()` interceptor is never invoked

## Current Status

- ✅ dlsym interception implemented and built
- ✅ Library installed correctly
- ✅ LD_PRELOAD configured in systemd service
- ✅ Shim loaded in Ollama process (confirmed via /proc/PID/maps)
- ❌ dlsym interceptor is NOT being called (expected - library doesn't use dlsym)
- ❌ Device query functions are NOT being called
- ❌ compute=0.0 still appears in logs

## Root Cause Analysis

The real issue is that **device query functions are not being called at all**. This suggests:

1. **libggml-cuda.so initialization fails early** - before it calls device queries
2. **Or** it calls functions we're not intercepting correctly
3. **Or** it gets compute capability from a different source

## What Should Happen

Since libggml-cuda.so uses direct linking:
1. When libggml-cuda.so loads, the dynamic linker resolves CUDA symbols
2. Our shim functions (via LD_PRELOAD) should be used
3. When libggml-cuda.so calls `cuDeviceGetAttribute()`, it should call our shim
4. Our shim should return compute capability 9.0

## What's Actually Happening

1. libggml-cuda.so loads ✓
2. Symbols resolve to our shim (via LD_PRELOAD) ✓
3. But device query functions are never called ✗
4. compute=0.0 appears in logs ✗

## Possible Explanations

1. **ggml_backend_cuda_init fails before device queries**
   - Error message: "ggml_cuda_init: failed to initialize CUDA: [reason]"
   - Happens right after cuInit() and cuDriverGetVersion() succeed
   - Before any device query functions are called

2. **Ollama gets compute capability from a different source**
   - Maybe it uses Runtime API instead of Driver API
   - Maybe it reads from a different function
   - Maybe it caches the value from a previous failed attempt

3. **Functions are called but logs aren't captured**
   - Write interceptor may not be working in systemd context
   - Error log files aren't being created
   - Can't verify if functions are actually being called

## Next Steps

1. **Verify write interceptor is working**
   - Check if error log files are created
   - Test write interceptor in systemd context
   - Ensure stderr is being captured

2. **Check if functions are actually being called**
   - Use strace/ltrace to see function calls
   - Check if cuDeviceGetAttribute is invoked
   - Verify symbol resolution is working

3. **Ensure all initialization functions succeed**
   - Verify cuInit() succeeds
   - Verify cuDriverGetVersion() succeeds
   - Check if any function returns an error

4. **Check if Ollama uses Runtime API**
   - Maybe it calls cudaDeviceGetAttribute() instead
   - Need to ensure Runtime API shim also returns correct values

5. **Get full error message from ggml_backend_cuda_init**
   - The [reason] in the error message is the key
   - Need to see what's actually failing

## Conclusion

dlsym interception was the wrong approach because libggml-cuda.so uses direct linking. Our shim functions should work via direct symbol resolution (LD_PRELOAD), but device query functions aren't being called, suggesting that ggml_backend_cuda_init fails before it gets to them.

The next step is to understand why ggml_backend_cuda_init fails and ensure all initialization functions succeed so it can proceed to device queries.
