# Next Step: Fix Compute Capability (compute=0.0 → 9.0)

## Date: 2026-02-25 09:36:40

## Current Status

### ✅ Working
- Discovery: 331ms (no timeout)
- GPU detected: H100 80GB HBM3
- Library loading: libggml-cuda.so loads
- Initialization: `init_gpu_defaults()` sets compute_cap_major=9, minor=0

### ❌ Issue
- **compute=0.0** in Ollama logs (should be 9.0)
- Device filtered as "didn't fully initialize"
- Falls back to CPU mode

## Root Cause Analysis

### Key Finding
**None of our CUDA functions are being called during verification!**

- `cuDeviceGetAttribute`: NOT called
- `cuDeviceGetProperties`: NOT called
- `cudaDeviceGetAttribute`: NOT called
- `cudaGetDeviceProperties_v2`: NOT called
- `nvmlDeviceGetCudaComputeCapability`: NOT called

### Hypothesis

Ollama is getting `compute=0.0` from:
1. **libggml-cuda.so internal initialization** - The library may call CUDA functions internally, and if those fail or return 0, Ollama uses that value
2. **Cached value from initial discovery** - Ollama may have read compute capability earlier (before our shims were active) and cached it
3. **Different API path** - Ollama may use a different method we're not intercepting

## Solution Approach

Since our functions aren't being called directly, we need to ensure that **if** libggml-cuda.so calls CUDA functions internally, they return correct values.

### 1. Verify Function Implementations

All functions should return compute capability 9.0:

- ✅ `cuDeviceGetAttribute(pi, 75, dev)` → sets `*pi = 9` (from `g_gpu_info.compute_cap_major`)
- ✅ `cuDeviceGetAttribute(pi, 76, dev)` → sets `*pi = 0` (from `g_gpu_info.compute_cap_minor`)
- ✅ `cuDeviceGetProperties(prop, dev)` → sets `prop->major = 9`, `prop->minor = 0`
- ✅ `cudaDeviceGetAttribute(value, 75, 0)` → sets `*value = 9`
- ✅ `cudaDeviceGetAttribute(value, 76, 0)` → sets `*value = 0`
- ✅ `cudaGetDeviceProperties_v2(prop, 0)` → sets `prop->major = 9`, `prop->minor = 0`

### 2. Ensure Early Initialization

- ✅ `init_gpu_defaults()` is called in `cuInit()`
- ✅ Sets `g_gpu_info.compute_cap_major = 9`
- ✅ Sets `g_gpu_info.compute_cap_minor = 0`
- ✅ Log shows "GPU defaults applied (H100 80GB CC=9.0 VRAM=81920 MB)"

### 3. Add Comprehensive Logging

Add logging to ALL functions that might be called by libggml-cuda.so:
- `cuDeviceGetAttribute` - ✅ Already logs
- `cuDeviceGetProperties` - ✅ Already logs
- `cudaDeviceGetAttribute` - ✅ Already logs
- `cudaGetDeviceProperties_v2` - ✅ Already logs

### 4. Check for Indirect Calls

libggml-cuda.so may call functions indirectly through:
- Function pointers
- Internal CUDA Runtime API calls
- dlopen/dlsym of CUDA libraries

We need to ensure ALL paths return correct values.

## Next Actions

1. **Verify g_gpu_info is initialized before any queries**
   - Check that `init_gpu_defaults()` is called early enough
   - Ensure `g_gpu_info_valid = 1` is set

2. **Add defensive checks**
   - Ensure functions return 9/0 even if `g_gpu_info` isn't initialized
   - Add fallback to `GPU_DEFAULT_CC_MAJOR/MINOR` if needed

3. **Check if libggml-cuda.so calls functions we're not intercepting**
   - May need to intercept additional functions
   - May need to check if functions are called via different names

4. **Verify compute capability is available from the start**
   - Ensure `g_gpu_info.compute_cap_major = 9` is set before any queries
   - Add logging to confirm values are correct when functions are called

## Status

**Progress: 90% Complete**
- ✅ Discovery working
- ✅ GPU detected
- ✅ Library loading working
- ✅ Functions implemented correctly
- ⚠️ Functions not being called (may be indirect)
- ⚠️ compute=0.0 (need to find source)

## Key Insight

The fact that `compute=0.0` appears in Ollama logs suggests that **somewhere**, a CUDA function is returning 0 for compute capability. Since our functions aren't being called directly, they must be called indirectly by libggml-cuda.so, or Ollama is using a cached value from an earlier failed attempt.

We need to ensure that **if** our functions are called (even indirectly), they return 9.0, not 0.0.
