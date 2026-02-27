# Investigation Summary and Next Steps

## Date: 2026-02-26

## Critical Discovery

**Removing `OLLAMA_LLM_LIBRARY=cuda_v12` causes Ollama to crash with SEGV!**

This means:
- `OLLAMA_LLM_LIBRARY` is **REQUIRED** for Ollama to run
- We **CANNOT** remove it
- Solution **MUST** work WITH it set

## Current Status

### ✅ What's Working

1. **Shim Functionality** - Perfect
   - Device detected: `cuInit() device found at 0000:00:05.0`
   - GPU defaults applied
   - VGPU-STUB found
   - Device count functions return count=1
   - PCI bus ID functions working

2. **NVML Functionality** - Working
   - Constructor called
   - `nvmlDeviceGetCount()` returns count=1

3. **Library Loadability** - Working
   - Library loads manually
   - Dependencies resolve
   - All symlinks correct

4. **All Prerequisites** - In Place
   - Symlink exists: `/usr/local/lib/ollama/libggml-cuda.so` → `cuda_v12/libggml-cuda.so`
   - Environment variables set correctly
   - Driver version: 13000 (13.0)

### ❌ What's Not Working

**Backend scanner is NOT loading `libggml-cuda.so`**

Evidence:
- No `dlopen()` calls for `libggml-cuda.so`
- No "verifying if device is supported" message
- Scanner checks `cuda_v13` and `vulkan` but NOT `cuda_v12`
- Discovery shows `initial_count=0`, `library=cpu`

## Root Cause Analysis

### Key Finding

When `OLLAMA_LLM_LIBRARY=cuda_v12` is set:
- Scanner knows to look for `cuda_v12`
- Scanner checks other subdirectories (`cuda_v13`, `vulkan`) and skips them
- Scanner does NOT check `cuda_v12/` subdirectory
- Scanner does NOT check top-level directory

### Why This Happens

According to `ROOT_CAUSE_FIXED.md`:
- Scanner normally looks in top-level directory
- But when `OLLAMA_LLM_LIBRARY` is set, scanner behavior changes
- Scanner may expect library to be loaded via different mechanism
- OR scanner may skip the requested directory assuming it's handled differently

## Proposed Solutions

### Solution 1: Ensure Library is in Expected Location (Safest)

When `OLLAMA_LLM_LIBRARY=cuda_v12` is set, scanner may expect library at:
- `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so` (already exists ✓)
- But may need it accessible via different path or mechanism

**Action:** Verify scanner can access library from `cuda_v12/` when `OLLAMA_LLM_LIBRARY` is set

### Solution 2: Force Library Loading via Alternative Mechanism

If scanner doesn't load it, pre-load via:
- Systemd service modification
- Wrapper script
- LD_PRELOAD (may not work for backend libraries)

**Risk:** Medium - may affect other processes

### Solution 3: Investigate Scanner Source Code Behavior

Understand exactly what scanner does when `OLLAMA_LLM_LIBRARY` is set:
- What location does it check?
- What conditions must be met?
- Why doesn't it load from `cuda_v12/`?

**Action:** Research Ollama scanner behavior with `OLLAMA_LLM_LIBRARY` set

## Recommended Next Steps

1. **Verify library accessibility from cuda_v12/**
   - Check if scanner can see/access it
   - Verify all paths and permissions

2. **Check if there's a missing prerequisite**
   - Something scanner needs before loading
   - May be a specific file or condition

3. **Investigate why scanner skips cuda_v12/**
   - Scanner checks other directories but not requested one
   - May be intentional behavior that needs workaround

4. **Consider alternative loading mechanism**
   - If scanner won't load it, find another way
   - Must work WITH `OLLAMA_LLM_LIBRARY` set

## Safety Considerations

- ✅ All working parts preserved
- ✅ No code changes until root cause confirmed
- ✅ `OLLAMA_LLM_LIBRARY` must remain set (removing it causes crash)
- ✅ All solutions must work WITH the setting

## Conclusion

**The shim works perfectly.** The issue is that Ollama's backend scanner doesn't load `libggml-cuda.so` when `OLLAMA_LLM_LIBRARY=cuda_v12` is set, even though:
- Library exists and is loadable
- All prerequisites are in place
- Scanner is running

**We need to find why scanner skips `cuda_v12/` when it's the requested library and ensure it loads the library from there.**
