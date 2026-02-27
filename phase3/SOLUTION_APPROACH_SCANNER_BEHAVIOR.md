# Solution Approach: Scanner Behavior with OLLAMA_LLM_LIBRARY

## Date: 2026-02-26

## Key Finding

**When `OLLAMA_LLM_LIBRARY=cuda_v12` is set:**
- Scanner checks OTHER directories (`cuda_v13`, `vulkan`) and skips them
- Scanner does NOT check `cuda_v12/` directory
- Scanner appears to assume `cuda_v12` is already handled

## Evidence

### Scanner Logs Show:
```
skipping available library at user's request requested=cuda_v12 libDir=/usr/local/lib/ollama/cuda_v13
skipping available library at user's request requested=cuda_v12 libDir=/usr/local/lib/ollama/vulkan
```

**But NO log about:**
- Finding `cuda_v12`
- Loading `cuda_v12`
- Checking `cuda_v12/` directory

### Directory Comparison

Both `cuda_v12/` and `cuda_v13/` have:
- `libggml-cuda.so` file
- Similar structure
- All required libraries

**Difference:** Scanner checks `cuda_v13/` but not `cuda_v12/` when it's the requested library.

## Hypothesis

When `OLLAMA_LLM_LIBRARY=cuda_v12` is set:

1. **Scanner skips scanning phase for requested library**
   - Scanner knows `cuda_v12` is requested
   - Scanner checks OTHER directories to skip them
   - Scanner assumes `cuda_v12` is already available/loaded
   - Scanner doesn't check `cuda_v12/` because it's the requested one

2. **Library should be loaded via different mechanism**
   - When `OLLAMA_LLM_LIBRARY` is set, library may be loaded directly
   - Not via scanner discovery
   - But this mechanism isn't working

3. **Scanner expects library in specific state**
   - Library may need to be pre-loaded
   - Or library may need to be accessible from different path
   - Or scanner needs specific condition to be met

## Comparison with Working State

**When Working (BREAKTHROUGH_SUMMARY.md - Feb 25):**
- Logs showed: `library=/usr/local/lib/ollama/cuda_v12`
- "verifying if device is supported" message appeared
- Library WAS loaded and verified

**Current State:**
- No "verifying" message
- No library path in logs
- Library not loaded

**Key Difference:**
- Previously: Library was loaded (even if initialization had issues)
- Currently: Library is not even being opened

## Possible Solutions

### Solution 1: Ensure Library is Pre-loaded

If scanner assumes library is already loaded when `OLLAMA_LLM_LIBRARY` is set:
- Pre-load library before discovery runs
- Use LD_PRELOAD or similar mechanism
- Ensure library is in memory when scanner needs it

**Risk:** Medium - may affect other processes

### Solution 2: Force Scanner to Check cuda_v12/

If scanner skips requested library directory:
- Find way to make scanner check it anyway
- May require understanding scanner source code
- Or may need specific file/marker in directory

**Risk:** Low - investigation only

### Solution 3: Load Library via Alternative Path

If scanner doesn't load it when `OLLAMA_LLM_LIBRARY` is set:
- Load library from different location
- Or use different loading mechanism
- Or ensure library is accessible from expected path

**Risk:** Low - testing only

### Solution 4: Check for Missing Condition

If scanner needs specific condition:
- Check what was different when it was working
- Verify all files and conditions are in place
- Ensure nothing is missing

**Risk:** Very low - verification only

## Recommended Approach

Since scanner previously loaded library from `cuda_v12/` (BREAKTHROUGH_LIBGGML_LOADING.md), and all files are in place:

1. **Investigate what triggers library loading when OLLAMA_LLM_LIBRARY is set**
   - Does scanner load it directly?
   - Or does it need to be pre-loaded?
   - What condition must be met?

2. **Check if there's a missing file or initialization step**
   - Something that was present when it was working
   - That's missing now

3. **Consider forcing library to load**
   - If scanner won't load it, find alternative way
   - Must work WITH `OLLAMA_LLM_LIBRARY` set

## Next Steps

1. Check if library needs to be pre-loaded before discovery
2. Investigate what triggers library loading when `OLLAMA_LLM_LIBRARY` is set
3. Verify all conditions are met that were present when it was working
4. Consider alternative loading mechanisms if scanner won't load it

## Conclusion

**The scanner skips `cuda_v12/` when it's the requested library, assuming it's already handled.** But the library isn't being loaded, which suggests either:
- The loading mechanism that should work with `OLLAMA_LLM_LIBRARY` isn't working
- A condition or file is missing
- The library needs to be in a different state

**We need to find what triggers the library to load when `OLLAMA_LLM_LIBRARY` is set and ensure that mechanism works.**
