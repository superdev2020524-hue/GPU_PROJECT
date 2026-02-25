# Next Step: Investigating Discovery Timeout

## Current Status

✅ **Driver Version Errors Fixed:**
- Increased driver version to 13000 (13.0)
- "CUDA driver version is insufficient" error is GONE
- "API call is not supported" error is GONE

✅ **Initialization Working:**
- cuInit() is called and succeeds
- nvmlInit_v2() is called and succeeds
- Device found at 0000:00:05.0

❌ **Discovery Still Timing Out:**
- Discovery starts: "discovering available GPUs..."
- Times out after 30 seconds: "failed to finish discovery before timeout"
- Device functions are NOT called (cuDeviceGetCount, nvmlDeviceGetCount_v2)
- GPU mode still CPU: `library=cpu`

## The Problem

**Discovery times out, but we don't know why.**

Possible reasons:
1. **libggml-cuda.so loads but initialization fails** - Maybe still failing for a different reason
2. **libggml-cuda.so doesn't load** - Maybe discovery doesn't try to load it
3. **Discovery waits for something else** - Maybe waits for a function call that never happens
4. **Device functions are called but fail silently** - Maybe called but return errors

## Next Steps

1. **Check if libggml-cuda.so is being loaded** - Use strace to see if dlopen() is called
2. **Check if ggml_backend_cuda_init still fails** - Maybe new error after driver version fix
3. **Check if device functions are called but logging fails** - Maybe called but we don't see logs
4. **Check discovery mechanism** - Understand how Ollama actually does discovery

## Hypothesis

Since device functions aren't called, Ollama might:
- Try to load libggml-cuda.so directly (without checking device count first)
- If libggml-cuda.so loads, call ggml_backend_cuda_init
- If ggml_backend_cuda_init fails, discovery times out

We fixed driver version errors, so maybe ggml_backend_cuda_init succeeds now, but discovery still times out for another reason?
