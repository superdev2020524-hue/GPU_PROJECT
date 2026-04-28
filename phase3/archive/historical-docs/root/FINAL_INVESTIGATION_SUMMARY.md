# Final Investigation Summary

## Date: 2026-02-26

## Investigation Complete

### ✅ Confirmed Working

1. **Shim Functionality** - Perfect
   - Device detected: `cuInit() device found at 0000:00:05.0`
   - GPU defaults applied: `H100 80GB CC=9.0 VRAM=81920 MB`
   - VGPU-STUB found
   - Device count functions return count=1
   - PCI bus ID functions working
   - **Can reproduce Feb 25 results**

2. **NVML Functionality** - Working
   - Constructor called
   - `nvmlDeviceGetCount()` returns count=1
   - `nvmlInit_v2()` called

3. **Library Loadability** - Working
   - Library loads manually from `cuda_v12/`
   - All dependencies resolve correctly
   - Version symbols present
   - Symlinks correct

4. **All Prerequisites** - In Place
   - Symlink: `/usr/local/lib/ollama/libggml-cuda.so` → `cuda_v12/libggml-cuda.so`
   - Target file exists: `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so` (1.6GB)
   - Environment variables: `OLLAMA_LLM_LIBRARY=cuda_v12`, `OLLAMA_NUM_GPU=999`
   - Driver version: 13000 (13.0)
   - All shim libraries loaded

### ❌ Root Cause Identified

**Backend scanner is NOT loading `libggml-cuda.so` when `OLLAMA_LLM_LIBRARY=cuda_v12` is set**

**Evidence:**
- No `dlopen()` calls for `libggml-cuda.so`
- No "verifying if device is supported" message
- Scanner checks `cuda_v13` and `vulkan` but NOT `cuda_v12`
- Discovery shows `initial_count=0`, `library=cpu`
- `libggml-cuda.so` NOT in process memory maps

### Critical Discovery

**Removing `OLLAMA_LLM_LIBRARY=cuda_v12` causes Ollama to crash with SEGV!**

This means:
- `OLLAMA_LLM_LIBRARY` is **REQUIRED** for Ollama to run
- We **CANNOT** remove it
- Solution **MUST** work WITH it set

### Key Comparison

**When Working (BREAKTHROUGH_LIBGGML_LOADING.md):**
- `strace` showed library WAS being opened from `cuda_v12/`
- Scanner tried to load it (but initialization timed out)
- Logs showed `library=/usr/local/lib/ollama/cuda_v12`

**Current State:**
- Library is NOT being opened at all
- Scanner doesn't even try to load it
- No logs about finding/loading `cuda_v12`

### Root Cause Analysis

When `OLLAMA_LLM_LIBRARY=cuda_v12` is set:
1. Scanner knows `cuda_v12` is requested
2. Scanner skips OTHER libraries (`cuda_v13`, `vulkan`) - correct behavior
3. Scanner does NOT check `cuda_v12/` subdirectory
4. Scanner does NOT check top-level directory
5. Scanner appears to assume `cuda_v12` is already handled or expects it to be loaded differently

### Why This Happens

According to `install.sh`:
- When `OLLAMA_LLM_LIBRARY=cuda_v12` is set, it "bypasses NVML-based GPU detection entirely"
- Library should be loadable from `cuda_v12/` via RPATH `$ORIGIN`
- But scanner doesn't load it

**Possible reasons:**
1. Scanner behavior changed in Ollama version
2. Scanner expects library to be pre-loaded or in specific state
3. Scanner needs a missing file or condition
4. Scanner skips requested library assuming it's handled differently

## Solution Status

### What We Know

1. ✅ Shim works perfectly - can reproduce Feb 25 results
2. ✅ Library is loadable - manual test succeeds
3. ✅ All prerequisites in place
4. ❌ Scanner doesn't load library when `OLLAMA_LLM_LIBRARY` is set
5. ❌ Cannot remove `OLLAMA_LLM_LIBRARY` (causes crash)

### What We Need

A solution that:
- Works WITH `OLLAMA_LLM_LIBRARY=cuda_v12` set (required)
- Ensures scanner loads `libggml-cuda.so` from `cuda_v12/`
- Doesn't break any working parts
- Only modifies necessary parts

### Possible Solutions

1. **Investigate scanner requirements**
   - What does scanner need to load library when `OLLAMA_LLM_LIBRARY` is set?
   - Is there a missing file or condition?
   - Does scanner need library in specific state?

2. **Force library loading**
   - Pre-load library before discovery runs
   - Use alternative loading mechanism
   - Ensure library is accessible from expected location

3. **Check Ollama version**
   - Scanner behavior may have changed
   - May need different approach for current version

## Conclusion

**The shim works perfectly and can reproduce Feb 25 results.** The issue is purely with Ollama's backend scanner not loading `libggml-cuda.so` when `OLLAMA_LLM_LIBRARY=cuda_v12` is set.

**The scanner previously opened the library from `cuda_v12/` (as shown in BREAKTHROUGH_LIBGGML_LOADING.md), but currently doesn't even try to open it.**

**We need to find why the scanner skips `cuda_v12/` when it's the requested library and ensure it loads the library from there.**

## Next Steps

1. Investigate what scanner needs to load library when `OLLAMA_LLM_LIBRARY` is set
2. Check if there's a missing file, condition, or initialization step
3. Consider alternative loading mechanisms if scanner won't load it
4. Verify Ollama version and scanner behavior

All investigation findings are documented in:
- `CRITICAL_FINDING_OLLAMA_CRASHES_WITHOUT_OLLAMA_LLM_LIBRARY.md`
- `INVESTIGATION_SUMMARY_AND_NEXT_STEPS.md`
- `INVESTIGATION_COMPLETE_SOLUTION_PROPOSED.md`
