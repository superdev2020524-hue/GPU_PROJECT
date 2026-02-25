# Ollama Compute Capability Source Research

## Research Summary

### Web Search Results
Web searches for Ollama source code did not return directly accessible results. The searches returned generic CUDA documentation and forum posts, but not the actual Ollama GitHub repository code.

### Key Findings from Log Analysis
Based on the error logs and behavior observed:

1. **Ollama logs show**: `compute=0.0` and `"didn't fully initialize"`
2. **Source locations from logs**:
   - `source=runner.go:146` - "verifying if device is supported"
   - `source=types.go:60` - Device type definitions

### Hypothesis
Ollama likely gets compute capability from:
1. **libggml-cuda.so's internal state** - The library may query compute capability during its own initialization
2. **Default/fallback value** - libggml-cuda.so may default to 0.0 if initialization fails or compute can't be determined
3. **ggml_backend_cuda_init return value** - Ollama may check the return value or status from this function

### What We Know
- Our shim functions (`cuDeviceGetAttribute`, `cudaGetDeviceProperties_v2`, etc.) are correctly implemented
- These functions are **NOT being called** during device verification
- `libggml-cuda.so` loads successfully
- `cuInit()` is called (we intercept it)
- Device query functions are never called

### Next Steps
1. Implement dlsym interception (Route 2) to catch indirect CUDA function lookups
2. Add comprehensive logging to see what functions libggml-cuda.so is looking for
3. Check if libggml-cuda.so uses dlsym/dlopen to resolve CUDA functions directly

## Implementation Status
- Route 2 (dlsym interception) has been implemented
- Route 1 (source code research) requires direct access to Ollama GitHub repository
