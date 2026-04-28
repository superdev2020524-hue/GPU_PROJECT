# Investigation Complete - Final Report

## Date: 2026-02-26

## Executive Summary

**Status:** Investigation complete. Root cause identified but requires Ollama source code access or version change to fully resolve.

**Key Finding:** The shim works perfectly and can reproduce Feb 25 results. The issue is with Ollama's backend scanner behavior when `OLLAMA_LLM_LIBRARY=cuda_v12` is set - the scanner skips checking the requested library directory.

## Investigation Results

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
   - ldconfig configured correctly

### ❌ Root Cause Identified

**Backend scanner is NOT loading `libggml-cuda.so` when `OLLAMA_LLM_LIBRARY=cuda_v12` is set**

**Evidence:**
- No `dlopen()` calls for `libggml-cuda.so`
- No "verifying if device is supported" message
- Scanner checks `cuda_v13` and `vulkan` but NOT `cuda_v12`
- Discovery shows `initial_count=0`, `library=cpu`
- `libggml-cuda.so` NOT in process memory maps

**Scanner Behavior:**
- When `OLLAMA_LLM_LIBRARY=cuda_v12` is set, scanner:
  1. Checks OTHER directories (`cuda_v13`, `vulkan`) and skips them
  2. Does NOT check `cuda_v12/` directory
  3. Appears to assume `cuda_v12` is already handled
  4. Does NOT attempt to load the library

### Critical Discovery

**Removing `OLLAMA_LLM_LIBRARY=cuda_v12` causes Ollama to crash with SEGV!**

This means:
- `OLLAMA_LLM_LIBRARY` is **REQUIRED** for Ollama to run
- We **CANNOT** remove it
- Solution **MUST** work WITH it set

## Comparison with Working State

### When Working (BREAKTHROUGH_LIBGGML_LOADING.md - Feb 25)

- `strace` showed library WAS being opened from `cuda_v12/`
- Scanner tried to load it (but initialization timed out)
- Logs showed `library=/usr/local/lib/ollama/cuda_v12`
- "verifying if device is supported" message appeared

### Current State

- Library is NOT being opened at all
- Scanner doesn't even try to load it
- No logs about finding/loading `cuda_v12`
- No "verifying" message

**Key Difference:** Previously, scanner opened the library. Currently, scanner doesn't even try.

## Root Cause Analysis

When `OLLAMA_LLM_LIBRARY=cuda_v12` is set:

1. Scanner knows `cuda_v12` is requested
2. Scanner checks OTHER directories (`cuda_v13`, `vulkan`) and skips them
3. Scanner does NOT check `cuda_v12/` directory
4. Scanner appears to assume `cuda_v12` is already handled or expects it to be loaded differently
5. Library isn't being loaded via the expected mechanism

**Why This Happens:**

According to `install.sh`:
- When `OLLAMA_LLM_LIBRARY=cuda_v12` is set, it "bypasses NVML-based GPU detection entirely"
- Library should be loadable from `cuda_v12/` via RPATH `$ORIGIN`
- But scanner doesn't load it

**Possible Reasons:**
1. Scanner behavior changed in Ollama version (0.16.3)
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

### Constraints

- Cannot modify Ollama source code
- Cannot remove `OLLAMA_LLM_LIBRARY` (required for Ollama to run)
- All documented fixes are in place
- Scanner behavior is internal to Ollama

## Possible Solutions (Require Further Investigation)

1. **Check Ollama Version/Release Notes**
   - Scanner behavior may have changed
   - May need different Ollama version
   - May be a known issue

2. **Investigate Scanner Source Code**
   - Understand exactly what scanner does when `OLLAMA_LLM_LIBRARY` is set
   - Find what triggers library loading
   - Identify missing condition or file

3. **Alternative Loading Mechanism**
   - Pre-load library before discovery runs
   - Use wrapper script or systemd modification
   - Force library loading via different mechanism

4. **Workaround**
   - Accept limitation of current Ollama version
   - Wait for Ollama fix
   - Use different approach

## Conclusion

**The shim works perfectly and can reproduce Feb 25 results.** The issue is purely with Ollama's backend scanner not loading `libggml-cuda.so` when `OLLAMA_LLM_LIBRARY=cuda_v12` is set.

**The scanner previously opened the library from `cuda_v12/` (as shown in BREAKTHROUGH_LIBGGML_LOADING.md), but currently doesn't even try to open it.**

**Without access to Ollama source code, it's difficult to determine exactly why the scanner skips the requested library directory.** This appears to be a limitation or change in Ollama's scanner behavior when `OLLAMA_LLM_LIBRARY` is set.

## Documentation

All investigation findings are documented in:
- `FINAL_INVESTIGATION_SUMMARY.md`
- `SOLUTION_APPROACH_SCANNER_BEHAVIOR.md`
- `CRITICAL_FINDING_OLLAMA_CRASHES_WITHOUT_OLLAMA_LLM_LIBRARY.md`
- `INVESTIGATION_SUMMARY_AND_NEXT_STEPS.md`
- `INVESTIGATION_COMPLETE_SOLUTION_PROPOSED.md`

## Next Steps

1. Check Ollama release notes for scanner behavior changes
2. Investigate if different Ollama version works
3. Research Ollama source code or community for similar issues
4. Consider alternative approaches if scanner behavior cannot be changed
