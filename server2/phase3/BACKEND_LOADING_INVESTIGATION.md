# Backend Loading Investigation

## Date: 2026-02-27

## ChatGPT's Hypothesis

**The CUDA backend library (`libggml-cuda.so`) is NOT being loaded during bootstrap discovery.**

This explains:
- No CUDA API calls during discovery
- No NVML calls during discovery
- `initial_count=0`

## Findings

### ✅ Library Can Be Loaded
- `python3 -c "ctypes.CDLL('libggml-cuda.so')"` → **SUCCESS**
- All dependencies resolve correctly
- Our shim (`libcuda.so.1`) is found

### ✅ Dependencies Resolve
```
libcuda.so.1 => /usr/lib64/libcuda.so.1 (our shim) ✅
libcudart.so.12 => /usr/lib64/libcudart.so.12 (our shim) ✅
libcublas.so.12 => /usr/local/lib/ollama/libcublas.so.12 ✅
```

### ❌ But NOT Loaded During Bootstrap
- `LD_DEBUG=libs ollama list` shows NO ggml loading
- No `dlopen` of `libggml-cuda.so` during discovery
- Discovery reports `initial_count=0`

## Hypothesis

Ollama's discovery logic decides NOT to load the CUDA backend before even trying.

Possible reasons:
1. **Environment variable check** - Maybe `OLLAMA_LLM_LIBRARY=cuda_v12` isn't respected during bootstrap
2. **Library path check** - Maybe discovery can't find the library
3. **Pre-validation** - Maybe discovery checks something else first that fails
4. **Configuration** - Maybe there's a config that disables CUDA backend

## Next Steps

1. Check if `libggml-cuda.so` is in the expected location
2. Check if discovery logs show why it's skipped
3. Check if there's a condition that prevents loading
