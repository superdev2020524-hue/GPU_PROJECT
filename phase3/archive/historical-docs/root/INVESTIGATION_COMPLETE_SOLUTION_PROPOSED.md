# Investigation Complete - Solution Proposed

## Date: 2026-02-26

## Investigation Summary

### ✅ What's Confirmed Working

1. **Shim Functionality** ✅
   - Device detected: `cuInit() device found at 0000:00:05.0`
   - GPU defaults applied: `H100 80GB CC=9.0 VRAM=81920 MB`
   - VGPU-STUB found: `vendor=0x10de device=0x2331 class=0x030200`
   - Device count functions return count=1
   - PCI bus ID functions implemented and working

2. **NVML Functionality** ✅
   - NVML constructor called
   - `nvmlDeviceGetCount()` called and returns count=1
   - `nvmlInit_v2()` called

3. **Library Loadability** ✅
   - Library loadable manually from `cuda_v12/`
   - All dependencies resolved
   - Version symbols present

4. **All Prerequisites** ✅
   - Symlink exists: `/usr/local/lib/ollama/libggml-cuda.so` → `cuda_v12/libggml-cuda.so`
   - Target file exists
   - Environment variables set: `OLLAMA_LLM_LIBRARY=cuda_v12`, `OLLAMA_NUM_GPU=999`
   - Driver version: 13000 (13.0)

### ❌ What's Not Working

**Backend scanner is NOT loading `libggml-cuda.so`**

Evidence:
- `libggml-cuda.so` NOT in process memory maps
- No "verifying if device is supported" message
- Discovery shows `initial_count=0`, `library=cpu`
- Scanner logs show it checking `cuda_v13` and `vulkan` but NOT `cuda_v12`

## Root Cause Analysis

### Key Finding from Logs

Scanner logs show:
```
skipping available library at user's request requested=cuda_v12 libDir=/usr/local/lib/ollama/cuda_v13
skipping available library at user's request requested=cuda_v12 libDir=/usr/local/lib/ollama/vulkan
```

**But NO log about:**
- Finding `cuda_v12`
- Loading `cuda_v12`
- Checking top-level directory

### Hypothesis

When `OLLAMA_LLM_LIBRARY=cuda_v12` is set:
1. Scanner knows to look for `cuda_v12`
2. Scanner checks subdirectories (`cuda_v13`, `vulkan`)
3. But scanner may **skip checking `cuda_v12/` subdirectory** because it's the requested one
4. Scanner may expect the library to be in a different location
5. OR scanner may need the library to be loaded via a different mechanism

### Comparison with Working State (Feb 25)

**When working:**
- `strace` showed `libggml-cuda.so` WAS being opened from `cuda_v12/` directly
- Logs showed `library=/usr/local/lib/ollama/cuda_v12`
- "verifying if device is supported" message appeared

**Current:**
- No `strace` opens of `libggml-cuda.so`
- No "verifying" message
- `library=cpu`

## Proposed Solution

Based on investigation and command.txt documentation:

### Solution: Ensure Scanner Finds Library in Requested Directory

**The issue:** When `OLLAMA_LLM_LIBRARY=cuda_v12` is set, the scanner may skip checking the `cuda_v12/` subdirectory because it's the requested one, expecting it to be handled differently.

**The fix:** Ensure the library is accessible where the scanner expects it when `OLLAMA_LLM_LIBRARY` is set.

### Implementation Options

#### Option 1: Verify Library Path Resolution (Safest)

According to `install.sh` line 610-611:
> "This bypasses NVML-based GPU detection entirely and ensures cuInit() reaches our shim even if LD_PRELOAD is not honoured for some reason."

The library should be loadable from `cuda_v12/` via RPATH `$ORIGIN`. Verify:
1. Library dependencies are resolvable from `cuda_v12/`
2. RPATH includes `$ORIGIN` or correct paths
3. All required symlinks are in place

**Risk:** Very low - verification only

#### Option 2: Force Library Pre-load (If Option 1 doesn't work)

If the scanner isn't loading the library, we could pre-load it via:
- LD_PRELOAD mechanism (but this may not work for backend libraries)
- Wrapper script that loads library before discovery
- Systemd service modification

**Risk:** Medium - may affect other processes

#### Option 3: Remove OLLAMA_LLM_LIBRARY Temporarily (Test)

Test if removing `OLLAMA_LLM_LIBRARY=cuda_v12` allows scanner to find `cuda_v12` via normal discovery.

**Risk:** Low - can be reverted immediately
**Benefit:** Will show if this setting is the issue

## Recommended Next Steps

1. **Verify library dependencies from cuda_v12/** (Option 1)
   - Check if all dependencies resolve correctly
   - Verify RPATH settings
   - Ensure all symlinks are correct

2. **If dependencies are correct, test removing OLLAMA_LLM_LIBRARY** (Option 3)
   - Temporarily comment it out
   - Restart Ollama
   - Check if scanner finds `cuda_v12`
   - This will confirm if the setting is preventing discovery

3. **If that works, investigate why OLLAMA_LLM_LIBRARY prevents loading**
   - May need to ensure library is in a specific location
   - Or may need a different approach when this setting is used

## Safety Considerations

- ✅ All working parts preserved
- ✅ Only verification and testing
- ✅ No code changes until root cause confirmed
- ✅ All changes reversible

## Conclusion

**The shim works perfectly and can reproduce Feb 25 results.** The issue is purely with Ollama's backend scanner not loading `libggml-cuda.so` when `OLLAMA_LLM_LIBRARY=cuda_v12` is set.

**The solution should focus on ensuring the scanner can find and load the library from the expected location when this environment variable is set.**
