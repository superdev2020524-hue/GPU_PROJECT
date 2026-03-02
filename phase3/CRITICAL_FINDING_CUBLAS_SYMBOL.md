# Critical Finding: CUBLAS Symbol Version Issue

## Problem Discovered

When trying to load `libggml-cuda.so`, we get:
```
OSError: /usr/local/lib/ollama/libggml-cuda.so: undefined symbol: cublasGetStatusString, version libcublas.so.12
```

## Root Cause

**Our CUBLAS stub library doesn't export `cublasGetStatusString` with the correct version tag.**

The real CUBLAS library exports it as:
```
cublasGetStatusString@@libcublas.so.12
```

But our stub was exporting it as:
```
cublasGetStatusString@@CUBLAS_12.0
```

## Solution

1. **Create version script** with correct version name: `libcublas.so.12`
2. **Compile with SONAME**: `libcublas.so.12`
3. **Export all cublas* functions** with the version tag

## Status

- ✅ Function `cublasGetStatusString` exists in our stub
- ✅ Version script created with `libcublas.so.12`
- ✅ SONAME set correctly
- ⏳ Testing if library loads now

## Impact

**This is THE blocker preventing GGML CUDA backend from loading!**

If `libggml-cuda.so` can't load because of missing symbols, then:
- CUDA backend never initializes
- Ollama falls back to CPU
- No CUDA calls are made
- Our shims are never used
- No data sent to GPU

## Next Steps

1. Verify library loads successfully
2. Test `ggml_backend_cuda_init()` call
3. Check if Ollama uses CUDA backend
4. Verify CUDA calls are made (and intercepted by our shims)
