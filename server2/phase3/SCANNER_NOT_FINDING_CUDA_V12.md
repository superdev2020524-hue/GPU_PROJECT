# Scanner Not Finding cuda_v12

## Date: 2026-02-26

## Key Finding

**Backend scanner is running but NOT finding cuda_v12**, even though:
- ✅ Symlink exists in top-level: `/usr/local/lib/ollama/libggml-cuda.so`
- ✅ Scanner is active (logs show "skipping available library")
- ✅ Scanner is skipping cuda_v13 and vulkan (because `OLLAMA_LLM_LIBRARY=cuda_v12`)

## Evidence

**Logs show**:
```
skipping available library at user's request requested=cuda_v12 libDir=/usr/local/lib/ollama/cuda_v13
skipping available library at user's request requested=cuda_v12 libDir=/usr/local/lib/ollama/vulkan
```

**But NO log about**:
- Finding cuda_v12
- Loading cuda_v12
- Scanning top-level directory

## Analysis

### When Working (BREAKTHROUGH_SUMMARY.md)
Logs showed: `library=/usr/local/lib/ollama/cuda_v12`

### Current State
- Scanner is skipping cuda_v13 and vulkan (correct behavior)
- But no log about finding/loading cuda_v12
- Scanner appears to only check subdirectories (cuda_v13, vulkan)
- Not checking top-level directory where symlink is

## Possible Causes

1. **Scanner behavior with OLLAMA_LLM_LIBRARY=cuda_v12**
   - When set, scanner may look in `cuda_v12/` subdirectory directly
   - Not in top-level directory
   - Symlink is in top-level, but scanner doesn't check there

2. **Scanner directory scanning**
   - Scanner may only scan subdirectories (cuda_v13, vulkan)
   - Not scanning top-level directory
   - Or scanning top-level but not finding libggml-cuda.so

3. **Symlink resolution**
   - Scanner may resolve symlink and look in cuda_v12/
   - But libggml-cuda.so may not be loadable from there
   - Or scanner may not follow symlinks

## What's Verified (No Changes Made)

✅ Symlink exists and is correct
✅ Target file exists
✅ Dependencies resolved
✅ No loading errors
✅ Scanner is running
✅ Scanner is skipping other libraries correctly

## Safe Next Steps

1. **Check if scanner looks in top-level**
   - Verify scanner behavior when OLLAMA_LLM_LIBRARY is set
   - Check if it scans top-level or only subdirectories

2. **Check if libggml-cuda.so needs to be in cuda_v12/**
   - According to BREAKTHROUGH_SUMMARY.md, working state showed:
     `library=/usr/local/lib/ollama/cuda_v12`
   - This is the subdirectory path, not top-level
   - May need to verify scanner looks in subdirectory

3. **Verify scanner conditions**
   - Check if scanner requires certain conditions
   - Or if there are prerequisite checks that fail

## Conclusion

**The backend scanner is running but not finding cuda_v12.** The scanner is checking subdirectories (cuda_v13, vulkan) but there's no evidence it's checking the top-level directory where the symlink is, or the cuda_v12 subdirectory.

**This is a scanner behavior issue, not a code issue.** All prerequisites are in place, but the scanner is not finding/loading libggml-cuda.so.
