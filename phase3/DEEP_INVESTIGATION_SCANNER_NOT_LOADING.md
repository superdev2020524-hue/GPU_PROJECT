# Deep Investigation: Scanner Not Loading libggml-cuda.so

## Date: 2026-02-26

## Investigation Summary

### ✅ What's Confirmed

1. **Library is Loadable** ✓
   - Manual load test succeeds
   - Device count functions work (return count=1)
   - Shims work correctly
   - All dependencies resolved

2. **All Prerequisites in Place** ✓
   - Symlink exists: `/usr/local/lib/ollama/libggml-cuda.so` → `cuda_v12/libggml-cuda.so`
   - Target file exists: `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so` (1.6GB)
   - Environment variables set: `OLLAMA_LLM_LIBRARY=cuda_v12`, `OLLAMA_NUM_GPU=999`
   - Shims loaded: `libvgpu-cuda.so`, `libvgpu-cudart.so`
   - Driver version: 13000 (13.0)

3. **Discovery Completes** ✓
   - Discovery completes quickly (~228ms)
   - No errors or timeouts
   - But library is not loaded

### ❌ What's Not Working

1. **libggml-cuda.so NOT Loaded** ❌
   - NOT in process memory maps
   - No "verifying if device is supported" message
   - Discovery shows `initial_count=0`, `library=cpu`

2. **Scanner Not Loading Library** ❌
   - No scanner activity logs
   - No library loading attempts visible
   - Scanner appears to skip CUDA backend

## Key Finding: Comparison with Working State

### When Working (BREAKTHROUGH_SUMMARY.md)

Logs showed:
```
time=2026-02-25T09:16:56.934-05:00 level=DEBUG source=runner.go:437 
msg="bootstrap discovery took" duration=302.578653ms 
OLLAMA_LIBRARY_PATH="[/usr/local/lib/ollama /usr/local/lib/ollama/cuda_v12]"

time=2026-02-25T09:17:26.935-05:00 level=DEBUG source=runner.go:146 
msg="verifying if device is supported" 
library=/usr/local/lib/ollama/cuda_v12 
description="NVIDIA H100 80GB HBM3" 
pci_id=99fff950:99fff9
```

**Key observation**: 30-second delay between:
- "bootstrap discovery took" (09:16:56)
- "verifying if device is supported" (09:17:26)

This delay suggests library loading and initialization.

### Current State

Logs show:
```
time=2026-02-26T04:45:48.892-05:00 level=DEBUG source=runner.go:437 
msg="bootstrap discovery took" duration=228.959725ms 
OLLAMA_LIBRARY_PATH="[/usr/local/lib/ollama /usr/local/lib/ollama/cuda_v12]"

time=2026-02-26T04:45:48.892-05:00 level=DEBUG source=runner.go:124 
msg="evaluating which, if any, devices to filter out" initial_count=0

time=2026-02-26T04:45:48.892-05:00 level=INFO source=types.go:60 
msg="inference compute" id=cpu library=cpu
```

**Key observation**: NO delay, NO "verifying if device is supported" message.

This confirms: **Library is not being loaded at all.**

## Root Cause Hypothesis

### Hypothesis 1: Scanner Checks Device Count First

According to `BACKEND_SCANNER_INVESTIGATION.md`:
- "Scanner may check device availability BEFORE loading"
- "If no devices found, skips loading CUDA backend"

**With `OLLAMA_LLM_LIBRARY=cuda_v12`:**
- NVML discovery is bypassed
- `nvmlDeviceGetCount()` may not be called
- Scanner may see device count as 0 or undefined
- Scanner skips CUDA backend

**Evidence:**
- No `nvmlDeviceGetCount()` calls in logs
- `initial_count=0` in discovery results
- Library is not loaded

### Hypothesis 2: Scanner Behavior Changed

- Scanner may have different behavior with `OLLAMA_LLM_LIBRARY=cuda_v12`
- May skip directory scanning when this is set
- May require different conditions

### Hypothesis 3: Library Load Fails Silently

- Scanner may try to load library
- Load fails silently (no error logs)
- Scanner continues without it

## Investigation Results

### 1. NVML Device Count Not Called

**Test**: Checked logs for `nvmlDeviceGetCount()` calls
**Result**: NO calls found
**Conclusion**: With `OLLAMA_LLM_LIBRARY=cuda_v12`, NVML discovery is bypassed

### 2. Scanner Activity Logs Missing

**Test**: Checked for scanner activity logs
**Result**: NO scanner logs found (no "skipping", "available library", etc.)
**Conclusion**: Scanner may not be logging, or may not be running

### 3. Library Loadable Manually

**Test**: Manual load test with `ctypes.CDLL()`
**Result**: Library loads successfully, device count functions work
**Conclusion**: Library itself is fine, issue is with scanner

### 4. Symlink and File Correct

**Test**: Verified symlink and target file
**Result**: Both exist and are correct
**Conclusion**: File system prerequisites are met

## Possible Solutions

### Solution 1: Ensure NVML Device Count Returns 1

Even with `OLLAMA_LLM_LIBRARY=cuda_v12`, ensure `nvmlDeviceGetCount()` is called and returns 1.

**Implementation**: NVML shim already returns count=1, but it's not being called.

### Solution 2: Remove OLLAMA_LLM_LIBRARY Temporarily

Test if removing `OLLAMA_LLM_LIBRARY=cuda_v12` allows scanner to find cuda_v12.

**Risk**: Low (can be reverted immediately)
**Benefit**: Will show if this setting is the issue

### Solution 3: Force Scanner to Check CUDA Device Count

Ensure scanner checks CUDA device count instead of (or in addition to) NVML.

**Implementation**: May require understanding Ollama's scanner source code

### Solution 4: Pre-load libggml-cuda.so

Force library to load before discovery runs.

**Implementation**: May require LD_PRELOAD or similar mechanism

## Next Steps

1. **Test removing OLLAMA_LLM_LIBRARY**: See if scanner finds cuda_v12 without it
2. **Check Ollama source code**: Understand scanner behavior and prerequisites
3. **Force NVML device count**: Ensure it's called even with OLLAMA_LLM_LIBRARY set
4. **Check for silent failures**: Look for any hidden errors or conditions

## Conclusion

**The backend scanner is not loading `libggml-cuda.so` despite all prerequisites being met.** The most likely cause is that the scanner checks device count first (via NVML), and with `OLLAMA_LLM_LIBRARY=cuda_v12`, NVML discovery is bypassed, so the scanner sees device count as 0 and skips the CUDA backend.

**The key missing element is the "verifying if device is supported" message**, which appears AFTER the library loads. Its absence confirms the library is not being loaded.

## References

- `SCANNER_NOT_LOADING_LIBRARY_FINAL.md` - Initial analysis
- `BACKEND_SCANNER_INVESTIGATION.md` - Scanner behavior documentation
- `BREAKTHROUGH_SUMMARY.md` - Working state reference
- `DRIVER_VERSION_13_VERIFICATION_RESULTS.md` - Driver version upgrade results
