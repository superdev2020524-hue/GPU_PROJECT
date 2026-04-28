# Root Cause Identified

## The Chain of Events

1. **Ollama performs GPU discovery** using NVML first
2. **If NVML discovery succeeds**, Ollama loads `libggml-cuda.so`
3. **`libggml-cuda.so` depends on `libcuda.so.1`** (our shim)
4. **When `libggml-cuda.so` loads, it loads `libcuda.so.1`** (our shim) as a dependency
5. **Our shim initializes CUDA** and provides GPU information

## The Problem

**Discovery is failing, so `libggml-cuda.so` is never loaded, so our shim is never loaded.**

## Why Discovery is Failing

Discovery uses **NVML first**. If NVML discovery fails, Ollama never proceeds to load CUDA libraries.

### Evidence

1. **Both main and runner processes have 0 CUDA/NVML libraries**
   - This means discovery failed
   - If discovery succeeded, `libggml-cuda.so` would be loaded
   - If `libggml-cuda.so` was loaded, `libcuda.so.1` (our shim) would be loaded

2. **No shim messages in logs**
   - Our NVML shim should print messages when called
   - No messages = shim is never called
   - Shim is never called = NVML discovery is not happening or failing before it reaches our shim

3. **GPU mode is CPU**
   - This confirms discovery failed
   - If discovery succeeded, GPU mode would be CUDA

## The Real Question

**Why is NVML discovery failing?**

Possible reasons:
1. **NVML shim is not being found**
   - Ollama calls `dlopen("libnvidia-ml.so.1")`
   - If it can't find our shim, discovery fails
   
2. **NVML shim is found but initialization fails**
   - `nvmlInit_v2()` is called but fails
   - Discovery fails and doesn't proceed to CUDA
   
3. **NVML discovery happens but returns no devices**
   - `nvmlDeviceGetCount_v2()` returns 0
   - Ollama thinks there are no GPUs
   - Doesn't proceed to CUDA loading

4. **Discovery happens in a different way**
   - Maybe not using `dlopen()` at all
   - Maybe using a different mechanism
   - Our shims are never called

## Next Steps

1. **Verify NVML shim can be found**
   - Test `dlopen("libnvidia-ml.so.1")` manually
   - Check if it resolves to our shim

2. **Test NVML shim manually**
   - Load shim and call `nvmlInit_v2()`
   - See if it works

3. **Check if discovery is actually calling NVML**
   - Trace `dlopen()` calls during discovery
   - See if `libnvidia-ml.so.1` is being opened

4. **Check discovery logs in detail**
   - See what happens during "discovering available GPUs..."
   - Check if there are any errors

## Key Finding

**The root cause is that GPU discovery is failing, so CUDA libraries are never loaded, so our shim is never called.**

We need to fix NVML discovery first, then CUDA will load automatically.
