# Complete Investigation Summary

## Date: 2026-02-26

## What We Found

### ✅ What's Working
1. **Crashes fixed** - Ollama runs stable (20+ minutes uptime)
2. **Discovery is running** - Bootstrap discovery completes in ~240ms
3. **cuDeviceGetCount() works** - Returns 1 in main process
4. **All symlinks correct** - CUDA and NVML shims symlinked correctly
5. **Configuration correct** - OLLAMA_LIBRARY_PATH, OLLAMA_LLM_LIBRARY set

### ❌ What's Not Working
1. **libggml-cuda.so NOT loading** - No "verifying if device is supported" message
2. **nvmlDeviceGetCount_v2() NOT called** - No logs showing it's called
3. **initial_count=0** - Discovery reports no GPU
4. **GPU mode is CPU** - `library=cpu` instead of `cuda`

## Root Cause

**`libggml-cuda.so` is not loading during discovery, which causes `initial_count=0`.**

### Evidence

**Feb 25 (Working):**
```
bootstrap discovery took 302ms
verifying if device is supported  ← This message comes from libggml-cuda.so
library=/usr/local/lib/ollama/cuda_v12
initial_count=1
```

**Now:**
```
bootstrap discovery took 239ms
(No "verifying" message)  ← libggml-cuda.so is NOT loading
initial_count=0
library=cpu
```

## Why libggml-cuda.so Doesn't Load

Possible reasons:
1. **Discovery skips loading when OLLAMA_LLM_LIBRARY=cuda_v12 is set**
   - Maybe this setting bypasses discovery
   - Maybe discovery assumes library is already loaded

2. **Library loading fails silently**
   - Maybe symbol resolution fails
   - Maybe dependencies are missing
   - Maybe version script issue

3. **Discovery uses different mechanism**
   - Maybe doesn't use OLLAMA_LIBRARY_PATH
   - Maybe checks something else first that fails

4. **Prerequisite check fails**
   - Maybe checks NVML first and fails
   - Maybe checks CUDA first and fails
   - Maybe checks PCI devices first and fails

## Next Steps

1. **Check if OLLAMA_LLM_LIBRARY=cuda_v12 affects discovery**
   - Maybe try without this setting
   - Maybe this setting prevents discovery from running

2. **Check if libggml-cuda.so can be loaded manually**
   - Test loading the library directly
   - Check for symbol resolution errors

3. **Compare with Feb 25 working state**
   - What was different when it worked?
   - What configuration/files were present?

4. **Check discovery logs for errors**
   - Look for any errors preventing library loading
   - Check if discovery is skipping CUDA backend

## Conclusion

**The root cause is that `libggml-cuda.so` is not loading during discovery.**

This is why `initial_count=0` and GPU mode is not active. We need to investigate why the library is not loading even though all prerequisites are in place.
