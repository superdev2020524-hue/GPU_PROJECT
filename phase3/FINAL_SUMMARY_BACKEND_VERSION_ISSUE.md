# Final Summary: Backend Version/Loading Issue

## Date: 2026-02-27

## ChatGPT's Analysis

**This is NOT a CUDA detection issue - it's Ollama's backend selection logic.**

### Key Finding

Discovery logs show:
- ✅ Skips `cuda_v13` (expected - we requested `cuda_v12`)
- ✅ Skips `vulkan` (expected)
- ❌ **NO message about loading or trying `cuda_v12`**
- ❌ `initial_count=0` immediately

**This means bootstrap never even considers `cuda_v12` as eligible.**

### What We Found

1. **Both backend directories exist:**
   - `/usr/local/lib/ollama/cuda_v12/` - has `libggml-cuda.so` (1.6GB)
   - `/usr/local/lib/ollama/cuda_v13/` - has `libggml-cuda.so` (380MB)

2. **Binary has CUDA support:**
   - Contains "CUDA" strings
   - But no specific version strings found

3. **Discovery behavior:**
   - Scans directories
   - Skips incompatible versions
   - But doesn't attempt to load `cuda_v12`

## Possible Root Causes

1. **Backend init function fails silently**
   - Library loads but init returns failure
   - No error logged

2. **Missing entry point**
   - Backend requires specific init symbol
   - Symbol not found or wrong signature

3. **Pre-validation fails**
   - Ollama checks something before loading
   - Check fails silently

4. **Version mismatch**
   - Binary expects different backend structure
   - Doesn't recognize `cuda_v12` as valid

## Next Steps for ChatGPT

Need to understand:
1. What does Ollama check before loading a backend?
2. What could cause backend loading to be skipped silently?
3. Is there a backend init function that must succeed?
4. How can we force backend loading or see why it's skipped?
