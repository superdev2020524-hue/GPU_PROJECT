# Root Cause: libggml-cuda.so Not Loading

## Date: 2026-02-26

## Problem Identified

**libggml-cuda.so is NOT being loaded during Ollama discovery**, even though:
- ✅ Symlink exists: `/usr/local/lib/ollama/libggml-cuda.so` → `cuda_v12/libggml-cuda.so`
- ✅ Target file exists: `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so` (1.6GB)
- ✅ All dependencies resolved (verified via `ldd`)
- ✅ No loading errors in logs
- ✅ dlsym/dlopen interception implemented and rebuilt

## Evidence

1. **Process memory check**: `libggml-cuda.so` NOT in `/proc/PID/maps`
2. **Discovery results**: `initial_count=0`, `library=cpu`, `pci_id=""`
3. **No dlopen/dlsym interception logs**: Suggests Ollama is not calling `dlopen("libggml-cuda.so")`

## Why This Matters

According to `ROOT_CAUSE_FIXED.md`:
- Ollama's backend scanner looks in top-level directory (`/usr/local/lib/ollama/`)
- Symlink allows scanner to find `libggml-cuda.so`
- Without it loading, no CUDA backend is available
- Result: `initial_count=0`, `library=cpu`

## What's Working

✅ **All prerequisites in place**:
- Symlink created (as per ROOT_CAUSE_FIXED.md)
- Version script fix applied (as per BREAKTHROUGH_SUMMARY.md)
- All symlinks in cuda_v12/ correct
- Environment variables set (`OLLAMA_LLM_LIBRARY=cuda_v12`, `OLLAMA_NUM_GPU=999`)
- dlsym/dlopen interception implemented
- Driver API shim loaded (`libvgpu-cuda.so` in process memory)
- Runtime API shim loaded (`libvgpu-cudart.so` in process memory)

## What's Not Working

❌ **Backend scanner is not loading libggml-cuda.so**:
- File exists and is accessible
- Dependencies resolved
- No loading errors
- But scanner is not finding/loading it

## Possible Causes

1. **Backend scanner not running**
   - Scanner may only run under certain conditions
   - Or may be disabled/failing silently

2. **OLLAMA_LLM_LIBRARY=cuda_v12 behavior**
   - Setting this may change how scanner works
   - Scanner may look in subdirectory directly (not top-level)
   - Or may skip scanning entirely

3. **Timing issue**
   - Scanner may run before symlink is accessible
   - Or may cache previous results

4. **Scanner conditions**
   - Scanner may require device count > 0 before loading
   - But device count is 0 because library isn't loaded
   - Circular dependency

## Safe Next Steps (No Code Changes)

1. **Check backend scanner logs**
   - Look for "scanning", "available library", "skipping" messages
   - Verify scanner is actually running

2. **Check if OLLAMA_LLM_LIBRARY affects scanner**
   - May need to remove it temporarily to test
   - Or verify it's working as expected

3. **Check discovery sequence**
   - Verify when scanner runs relative to other initialization
   - Check if there are prerequisite checks that fail

## Files Verified (No Changes Made)

- ✅ `/usr/local/lib/ollama/libggml-cuda.so` (symlink)
- ✅ `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so` (target)
- ✅ Dependencies (via `ldd`)
- ✅ Process memory maps (via `/proc/PID/maps`)
- ✅ Logs (no loading errors)

## Conclusion

**All documented fixes are in place, but libggml-cuda.so is still not loading.** The issue appears to be that the backend scanner is not finding or loading the library, despite the symlink and all prerequisites being correct.

**Next step**: Investigate why the backend scanner is not loading libggml-cuda.so, focusing on scanner behavior and conditions, without modifying any working code.
