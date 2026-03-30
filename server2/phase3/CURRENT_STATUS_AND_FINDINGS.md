# Current Status and Findings

## Date: 2026-02-26

## Current Status

### ✅ What's Working
1. **Ollama runs stable** - No crashes (20+ minutes uptime)
2. **All shim libraries loaded** - libvgpu-exec, libvgpu-cuda, libvgpu-nvml, libvgpu-cudart
3. **Configuration correct** - OLLAMA_LIBRARY_PATH, OLLAMA_LLM_LIBRARY set
4. **All symlinks correct** - CUDA and NVML shims symlinked properly
5. **Discovery runs** - Bootstrap discovery completes in ~240ms

### ❌ What's Not Working
1. **libggml-cuda.so NOT loading** - Confirmed via lsof
2. **GPU not detected** - initial_count=0
3. **GPU mode is CPU** - library=cpu
4. **No "verifying" message** - Library never loads

## Fixes Attempted

1. ✅ **Fixed crashes** - Removed libvgpu-syscall.so, removed force_load_shim
2. ✅ **Added OLLAMA_LIBRARY_PATH** - Tells scanner where to find libraries
3. ✅ **Added nvmlDeviceGetCount_v2() call in constructor** - To ensure discovery sees device count=1

**Result**: Fixes applied but issue persists. Library still not loading.

## Root Cause Analysis

**The core issue**: `libggml-cuda.so` is not loading during discovery.

### Evidence

**Feb 25 (Working):**
- bootstrap discovery took 302ms
- "verifying if device is supported" message appeared
- library=/usr/local/lib/ollama/cuda_v12
- initial_count=1

**Now:**
- bootstrap discovery took ~240ms
- No "verifying" message
- library=cpu
- initial_count=0
- libggml-cuda.so NOT in process memory

### Why Library Doesn't Load

When `OLLAMA_LLM_LIBRARY=cuda_v12` is set:
- Discovery skips scanning `cuda_v12/` directory
- Assumes library is already handled
- But library is NOT loaded
- Discovery doesn't check device count before assuming library is available

## Key Questions

1. **What triggers library loading when OLLAMA_LLM_LIBRARY is set?**
   - Discovery skips the directory
   - But library should still load somehow
   - What mechanism loads it?

2. **What was different on Feb 25?**
   - Same configuration
   - But library WAS loading
   - What changed?

3. **Does discovery need device count > 0 to load library?**
   - If so, why isn't device count > 0?
   - Device count functions return 1 in main process
   - But discovery runs in runner subprocess

## Next Investigation Steps

1. **Check if library needs to be pre-loaded**
   - Maybe discovery expects it to be already in memory
   - Or maybe it needs to be in a specific location

2. **Investigate runner subprocess**
   - Discovery runs in runner subprocess
   - Maybe runner doesn't have shims loaded
   - Or maybe runner doesn't call device count functions

3. **Compare with Feb 25 working state**
   - What files were present?
   - What was the exact configuration?
   - What triggered library loading?

4. **Check if there's a missing prerequisite**
   - Maybe a file or symlink is missing
   - Or maybe a specific environment variable

## Conclusion

**The issue persists**: `libggml-cuda.so` is not loading, preventing GPU detection and GPU mode.

All attempted fixes have been applied but haven't resolved the issue. Need to investigate what actually triggers library loading when `OLLAMA_LLM_LIBRARY=cuda_v12` is set.
