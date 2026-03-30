# Library Loading Investigation

## Date: 2026-02-26

## Current Status

**Ollama is running stable, but `libggml-cuda.so` is not loaded in the main process.**

### What's Working

1. ✅ **Ollama is running stable** (no crashes)
2. ✅ **All shim libraries are loaded:**
   - `libvgpu-exec.so` ✓
   - `libvgpu-cuda.so` ✓
   - `libvgpu-nvml.so` ✓
   - `libvgpu-cudart.so` ✓
3. ✅ **Configuration is correct:**
   - `OLLAMA_LIBRARY_PATH` is set
   - `OLLAMA_LLM_LIBRARY=cuda_v12` is set
   - Symlinks are correct
4. ✅ **Library exists and is accessible:**
   - `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so` exists
   - All dependencies are available
   - Library can be loaded manually

### Current Issue

**`libggml-cuda.so` is NOT loaded in the main Ollama process (PID 154237).**

### Investigation Results

1. **Library dependencies:** ✅ All available
   - `libcuda.so.1` → `/usr/lib64/libcuda.so.1` → points to shim ✓
   - `libcudart.so.12` → `/usr/lib64/libcudart.so.12` → points to shim ✓
   - All other dependencies available ✓

2. **Symlinks:** ✅ Correct
   - `/usr/lib64/libcuda.so.1` → `/usr/lib64/libvgpu-cuda.so` ✓
   - `/usr/lib64/libcudart.so.12` → `/usr/lib64/libvgpu-cudart.so` ✓
   - `/usr/local/lib/ollama/cuda_v12/libcuda.so.1` → shim ✓
   - `/usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90` → shim ✓

3. **Manual loading:** ✅ Works
   - Library can be loaded manually with Python
   - No errors when loading

4. **Discovery logs:** ⚠ None found
   - No bootstrap discovery logs
   - No scanner logs
   - No library loading logs

### Possible Explanations

1. **Library loads in subprocess (runner), not main process**
   - Ollama uses a runner subprocess for model execution
   - Library might only load when model is actually run
   - Main process might not need the library until model execution

2. **Discovery hasn't run yet**
   - Discovery might run on first actual model execution
   - Or it might run in the runner subprocess
   - Main process might not trigger discovery

3. **OLLAMA_LLM_LIBRARY setting**
   - When `OLLAMA_LLM_LIBRARY=cuda_v12` is set, scanner may skip discovery
   - But library still needs to be loaded when model runs
   - May only load in runner subprocess

### Next Steps

1. **Check runner subprocess:**
   - When a model is run, check if library loads in runner process
   - Runner process might have the library loaded

2. **Verify model execution:**
   - Actually run a model (not just request)
   - Check if library loads during execution
   - Check runner process maps

3. **Check if GPU is detected:**
   - Even if library isn't in main process, GPU might be detected
   - Check if models use GPU layers
   - Verify GPU mode is active

## Summary

**Status:**
- ✅ Ollama running stable
- ✅ All shims loaded
- ✅ Configuration correct
- ⚠ `libggml-cuda.so` not in main process
- ⚠ May load in runner subprocess when model runs

**The crash issue is completely resolved!** The library may load in the runner subprocess when models are actually executed, not in the main process.
