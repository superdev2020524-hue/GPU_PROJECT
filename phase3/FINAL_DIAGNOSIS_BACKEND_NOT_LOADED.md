# Final Diagnosis: Backend Not Loaded During Bootstrap

## Date: 2026-02-27

## Critical Finding

**The CUDA backend library (`libggml-cuda.so`) is NOT being loaded during bootstrap discovery.**

### Evidence

1. **Library exists and can be loaded:**
   - ✅ `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so` exists
   - ✅ Manual load succeeds: `ctypes.CDLL('libggml-cuda.so')` → SUCCESS
   - ✅ All dependencies resolve correctly

2. **But NOT loaded during bootstrap:**
   - ❌ `LD_DEBUG=libs ollama list` shows NO ggml loading
   - ❌ No `dlopen` of `libggml-cuda.so` during discovery
   - ❌ Discovery reports `initial_count=0`

3. **Discovery logs show:**
   - ✅ Skips cuda_v13 (expected - we requested cuda_v12)
   - ✅ Skips vulkan (expected)
   - ❌ **NO message about loading or trying cuda_v12**
   - ❌ `initial_count=0` immediately after bootstrap

## Root Cause

**Ollama's bootstrap discovery is NOT loading `libggml-cuda.so` even though:**
- Library exists
- Library can be loaded manually
- Dependencies resolve
- `OLLAMA_LLM_LIBRARY=cuda_v12` is set

## Possible Reasons

1. **Backend init fails silently** - Library loads but init function fails
2. **Symbol lookup fails** - Backend init symbol not found
3. **Pre-validation fails** - Something checks before loading that fails
4. **Path issue** - Discovery can't find the library in the expected location

## Next Steps for ChatGPT

Need to understand:
1. What does Ollama check before loading a backend?
2. What could cause backend loading to be skipped silently?
3. How can we force backend loading during bootstrap?
