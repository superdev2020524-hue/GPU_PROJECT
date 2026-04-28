# dlsym Interception Findings

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
- ❌ dlsym interceptor is NOT being called (expected - library doesn't use dlsym)
- ❌ Device query functions are NOT being called
- ❌ compute=0.0 still appears in logs

## Root Cause

The real issue is that **device query functions are not being called at all**. This suggests:

1. **libggml-cuda.so initialization fails early** - before it calls device queries
2. **Or** it calls functions we're not intercepting correctly
3. **Or** it gets compute capability from a different source

## Next Steps

Since dlsym interception won't work (library doesn't use dlsym), we need to:

1. **Verify our shim functions are being called** when libggml-cuda.so loads
2. **Check what functions ARE being called** during initialization
3. **Ensure all initialization functions return success** so it proceeds to device queries
4. **Verify compute capability functions return correct values** when called

The shim functions should be called via direct symbol resolution (LD_PRELOAD), not dlsym interception.
