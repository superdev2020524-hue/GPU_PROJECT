# Complete Backend Version Analysis

## Date: 2026-02-27

## Findings

### Binary Analysis
- ✅ Binary contains "CUDA" strings (has CUDA support)
- ❌ No specific version strings like "cuda_v12" or "cuda_v13" found
- This suggests version might be determined at runtime or via directory scanning

### Available Backends
- ✅ `/usr/local/lib/ollama/cuda_v12/` exists (has `libggml-cuda.so`)
- ✅ `/usr/local/lib/ollama/cuda_v13/` exists (directory present)
- ✅ `/usr/local/lib/ollama/vulkan/` exists

### Discovery Behavior
- ✅ Skips `cuda_v13` (expected - we requested `cuda_v12`)
- ✅ Skips `vulkan` (expected)
- ❌ **NO message about loading `cuda_v12`**
- ❌ `initial_count=0` immediately

## Hypothesis

The binary might be:
1. **Scanning directories** and finding `cuda_v12` but not loading it
2. **Checking for init function** that fails silently
3. **Requiring a specific file** that's missing
4. **Version mismatch** - binary expects different structure

## Next Steps

1. Check what's in `cuda_v13` directory (maybe it's empty?)
2. Check if there's a specific init function or entry point required
3. Check if discovery logs show any errors about `cuda_v12`
