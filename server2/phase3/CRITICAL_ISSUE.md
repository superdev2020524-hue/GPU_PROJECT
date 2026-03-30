# Critical Issue: cuDeviceGetCount() Never Called

## Current Status

✅ **All Infrastructure Complete:**
- cuDeviceGetCount() simplified to always return count=1 immediately
- nvmlDeviceGetCount_v2() returns count=1 immediately
- All functions implemented and working
- Early initialization in constructors
- Device discovery working

❌ **But:**
- cuDeviceGetCount() is NEVER called during discovery
- ggml_cuda_init() fails with "failed to initialize" (truncated)
- Discovery times out after 30 seconds

## The Mystery

**Why is cuDeviceGetCount() never called?**

We've:
1. Made it return immediately ✓
2. Made it work during early loading ✓
3. Verified it can be found via dlsym() ✓
4. Verified it works when called directly ✓
5. Simplified it to always return count=1 ✓

But it's STILL never called!

## Possible Explanations

1. **ggml_cuda_init() doesn't call cuDeviceGetCount()** - Uses different mechanism
2. **ggml_cuda_init() fails before calling it** - Checks something else first that fails
3. **ggml_cuda_init() is called in subprocess without shims** - Runner doesn't have LD_PRELOAD
4. **ggml_cuda_init() calls a different function first that fails** - Missing function
5. **Different CUDA version library** - Maybe cuda_v13 instead of cuda_v12?

## What We Need

1. **Full error message** - See exactly why ggml_cuda_init() fails
2. **Understand ggml_cuda_init() behavior** - What does it actually do?
3. **Check if function is called but logging fails** - Maybe syscall doesn't work?
4. **Verify all required functions** - Is something missing?

## Next Steps

1. Check if cuDeviceGetCount() is called but logging fails
2. Check if different CUDA version library is being used
3. Get full error message from ggml_cuda_init()
4. Check if ggml_cuda_init() calls a function we don't have
