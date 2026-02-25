# Critical Finding: libggml-cuda.so Not Loaded

## The Problem

**`libggml-cuda.so` is NOT loaded in the Ollama process!**

This is why GPU mode is still CPU. Ollama's discovery must succeed and load `libggml-cuda.so` before it can use GPU mode.

## Current Status

✅ CUDA initialization: `cuInit()` is called and succeeds
✅ NVML initialization: `nvmlInit_v2()` is called and succeeds  
✅ Device discovery: VGPU device found at 0000:00:05.0
❌ **libggml-cuda.so: NOT LOADED**
❌ **Device query functions: NOT CALLED** (`cuDeviceGetCount`, `nvmlDeviceGetCount_v2`)

## Why Discovery Fails

Ollama's discovery process:
1. Calls `nvmlInit_v2()` ✓ (we intercept this)
2. Calls `nvmlDeviceGetCount_v2()` ❌ (NOT CALLED - this is the problem!)
3. If count > 0, loads `libggml-cuda.so` ❌ (never happens)
4. If `libggml-cuda.so` loads, uses GPU mode ❌ (never happens)

## The Root Cause

**Ollama's discovery is failing before it calls device count functions.**

Possible reasons:
1. **Discovery checks something else first** that fails
2. **Symbol versioning issue** - functions exist but can't be resolved
3. **Discovery uses a different mechanism** - maybe checks library loading differently
4. **NVML discovery fails silently** - returns error before calling device count

## Next Steps

1. **Add logging to ALL NVML functions** to see what Ollama actually calls
2. **Check if dlsym() can find our functions** - maybe symbol resolution fails
3. **Verify symbol versions** - maybe Ollama expects specific symbol versions
4. **Check if discovery happens in a subprocess** - maybe we're not intercepting the right process
