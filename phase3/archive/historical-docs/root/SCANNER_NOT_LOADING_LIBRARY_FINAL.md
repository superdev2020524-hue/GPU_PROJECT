# Scanner Not Loading Library - Final Analysis

## Date: 2026-02-26

## Key Finding

**The backend scanner is not loading `libggml-cuda.so`, even though:**
- ✅ Library is loadable (manual test succeeds)
- ✅ Device count functions work (return count=1)
- ✅ All prerequisites are in place
- ✅ Symlink exists and is valid
- ✅ Environment variables set correctly

## Evidence

### What's Working

1. **Manual Load Test** ✅
   - Library can be loaded via `ctypes.CDLL()`
   - Device count functions are called and return count=1
   - Shims are working correctly

2. **All Prerequisites** ✅
   - Version script fix applied
   - Symlink created: `/usr/local/lib/ollama/libggml-cuda.so` → `cuda_v12/libggml-cuda.so`
   - All dependencies resolved
   - Environment variables set (`OLLAMA_LLM_LIBRARY=cuda_v12`, `OLLAMA_NUM_GPU=999`)
   - Shims loaded (Driver API and Runtime API)

3. **Scanner Access** ✅
   - `OLLAMA_LIBRARY_PATH` includes both top-level and cuda_v12/
   - Scanner has access to both locations

### What's Not Working

1. **Scanner Not Loading Library** ❌
   - `libggml-cuda.so` NOT in process memory maps
   - No "verifying if device is supported" log (this appears AFTER library loads)
   - No dlopen/dlsym interception logs for libggml-cuda.so

2. **Discovery Results** ❌
   - `initial_count=0`
   - `library=cpu`
   - `pci_id=""`

## Comparison with Working State

### When Working (BREAKTHROUGH_SUMMARY.md)

Logs showed:
```
msg="verifying if device is supported" 
library=/usr/local/lib/ollama/cuda_v12 
description="NVIDIA H100 80GB HBM3" 
pci_id=99fff950:99fff9
```

**Key point**: The "verifying if device is supported" message appears AFTER the library is loaded. This message is currently missing, which confirms the library is not being loaded.

### Current State

- No "verifying if device is supported" message
- No library path in discovery logs
- Scanner is skipping cuda_v13 and vulkan (correct behavior)
- But no log about finding/loading cuda_v12

## Root Cause Hypothesis

**The scanner is not even attempting to load `libggml-cuda.so`**, which suggests:

1. **Scanner may have a prerequisite check that fails**
   - Scanner may check device count (via NVML) before loading
   - If NVML device count is 0, scanner skips CUDA backend
   - But NVML device count is not being called (OLLAMA_LLM_LIBRARY bypasses NVML)

2. **OLLAMA_LLM_LIBRARY=cuda_v12 behavior**
   - According to command.txt, this bypasses NVML discovery
   - But may also change scanner behavior in unexpected ways
   - Scanner may not scan directories when this is set

3. **Scanner may not be checking cuda_v12/ subdirectory**
   - Scanner logs show it checking cuda_v13 and vulkan
   - But no log about checking cuda_v12/
   - Scanner may skip the requested library directory

## What Was NOT Changed

Per user's instruction to avoid breaking working parts:
- ✅ Only verification performed
- ✅ No code changes made
- ✅ No configuration changes made
- ✅ All working parts preserved

## Next Steps (For Reference)

1. **Check if scanner needs NVML discovery first**
   - Temporarily remove `OLLAMA_LLM_LIBRARY=cuda_v12` to see if scanner finds cuda_v12
   - This is a safe test (can be reverted immediately)

2. **Check scanner source code**
   - Understand how scanner identifies backend libraries
   - Verify what conditions must be met for scanner to load a library

3. **Compare with working state**
   - Review what was different when it was working
   - Check if there were additional steps or configuration

## Conclusion

**All documented fixes are in place, but the backend scanner is not loading `libggml-cuda.so`.** The library is loadable and shims work correctly, but the scanner is not even attempting to load it. This suggests a scanner behavior or condition issue, not a code issue.

**The key missing log is "verifying if device is supported"** - this appears AFTER the library loads, and its absence confirms the library is not being loaded.
