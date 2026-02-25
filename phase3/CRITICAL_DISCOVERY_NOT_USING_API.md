# Critical Finding: Ollama Not Using Standard NVML/CUDA API

## The Problem

**Ollama is NOT calling NVML/CUDA functions during discovery!**

### Evidence

1. **Constructors ARE called** - Libraries load via LD_PRELOAD ✓
2. **But Ollama NEVER calls:**
   - `nvmlInit_v2()` (only called from constructor, not by Ollama)
   - `nvmlDeviceGetCount_v2()` (NEVER called)
   - `cuDeviceGetCount()` (NEVER called)
   - Any other device query functions

3. **Result:** `libggml-cuda.so` never loads, GPU mode is CPU

## What This Means

**Ollama's discovery does NOT use the standard NVML/CUDA API!**

Possible explanations:
1. **Ollama uses a different discovery mechanism** - Maybe checks library existence but doesn't call functions
2. **Discovery fails before reaching NVML/CUDA** - Maybe checks `/proc/driver/nvidia/version` first and fails
3. **Discovery happens in a subprocess** - Maybe in a process we're not intercepting
4. **Ollama uses a wrapper library** - Maybe uses a different API that we're not intercepting

## Next Steps

1. **Check Ollama source code** - Understand exactly how discovery works
2. **Check if discovery uses different mechanism** - Maybe not using `dlopen()`/`dlsym()` at all
3. **Check if discovery checks files first** - Maybe `/proc/driver/nvidia/version` or `/dev/nvidia*`
4. **Check if discovery happens in subprocess** - Maybe we need to intercept a different process

## Key Insight

**We've been assuming Ollama uses standard NVML/CUDA API, but it might not!**

We need to understand HOW Ollama actually does discovery, not just assume it uses standard functions.
