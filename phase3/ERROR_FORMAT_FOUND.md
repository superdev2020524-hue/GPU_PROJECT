# Error Format Found!

## Critical Discovery

**Found the error message format in libggml-cuda.so:**
```
%s: failed to initialize CUDA: %s
```

This means the actual error message is:
```
ggml_cuda_init: failed to initialize CUDA: [reason]
```

The 98-byte message is this format with the [reason] filled in.

## What This Tells Us

1. **The error is formatted** - It's not a static string, it's formatted with a reason
2. **The reason is the key** - We need to find what the [reason] is
3. **It happens in ggml_backend_cuda_init** - The function that's failing

## Possible Reasons

Since the error happens right after `cuInit()` and `cuDriverGetVersion()` succeed, but before device query functions are called, the reason might be:

1. **CUDA runtime function fails** - Maybe calls `cudaGetDeviceCount()` or similar
2. **Device count is 0** - Maybe checks count and gets 0 (but we return 1)
3. **Missing function** - Maybe calls a function we don't have
4. **Runtime library issue** - Maybe a runtime function fails

## Next Steps

1. **Intercept CUDA runtime functions** - Maybe need to shim `cudaGetDeviceCount()` etc.
2. **Get the actual reason** - Need to see what the second `%s` is
3. **Check what ggml_backend_cuda_init calls** - Understand its internal flow

## Key Insight

The error format suggests there's a specific reason for the failure. If we can find what that reason is, we can fix it!
