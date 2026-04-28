# Critical Finding: Ollama Crashes Without OLLAMA_LLM_LIBRARY

## Date: 2026-02-26

## Critical Discovery

**Removing `OLLAMA_LLM_LIBRARY=cuda_v12` causes Ollama to crash with SEGV (segmentation fault)!**

### Evidence

When `OLLAMA_LLM_LIBRARY=cuda_v12` was commented out:
```
Feb 26 05:23:36 ollama.service: Main process exited, code=dumped, status=11/SEGV
Feb 26 05:23:36 ollama.service: Failed with result 'core-dump'.
```

Ollama crashes repeatedly and cannot start.

### What This Means

1. **OLLAMA_LLM_LIBRARY is REQUIRED** - Removing it breaks Ollama
2. **We cannot remove it** - It's necessary for stability
3. **We must find a solution that works WITH it set**

## Impact on Solution

### Previous Hypothesis
- Hypothesis: Removing `OLLAMA_LLM_LIBRARY` would allow scanner to find `cuda_v12`
- Test: Commented out `OLLAMA_LLM_LIBRARY`
- Result: **Ollama crashes - test invalidated**

### New Understanding

`OLLAMA_LLM_LIBRARY=cuda_v12` is not optional - it's required for Ollama to run. Therefore:
- We must keep it set
- We need to find why scanner doesn't load `libggml-cuda.so` when it's set
- We need a solution that works WITH the setting, not without it

## Revised Solution Approach

Since we cannot remove `OLLAMA_LLM_LIBRARY`:

1. **Investigate scanner behavior with OLLAMA_LLM_LIBRARY set**
   - Why doesn't scanner load `libggml-cuda.so`?
   - What location does scanner check when this is set?
   - What conditions must be met?

2. **Ensure library is accessible where scanner expects it**
   - Scanner may look in a specific location when `OLLAMA_LLM_LIBRARY` is set
   - Ensure library is there and loadable

3. **Force library loading via alternative mechanism**
   - If scanner doesn't load it, find another way to ensure it's loaded
   - May require pre-loading or different approach

## Safety Note

**DO NOT remove `OLLAMA_LLM_LIBRARY=cuda_v12`** - It causes Ollama to crash.

All solutions must work WITH this setting in place.

## Next Steps

1. Investigate what location scanner checks when `OLLAMA_LLM_LIBRARY=cuda_v12` is set
2. Ensure `libggml-cuda.so` is accessible from that location
3. Check if there are other prerequisites for scanner to load it
4. Consider alternative loading mechanisms if scanner doesn't work
