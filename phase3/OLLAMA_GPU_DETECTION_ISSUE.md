# Ollama GPU Detection Issue - Root Cause Analysis

## Date: 2026-02-27

## Current Status

### ✅ What Works
- **vGPU works for general GPU projects** - Python, CUDA, etc. all work correctly
- **All 3 shim libraries built and installed** - libvgpu-cuda.so, libvgpu-nvml.so, libvgpu-cudart.so
- **Libraries can be loaded manually** - Test programs work
- **Symlinks in place** - Bundled libraries point to shims

### ❌ What Doesn't Work
- **Ollama doesn't detect GPU** - `initial_count=0`, `library=cpu`
- **libggml-cuda.so not loaded** - CUDA backend never loads
- **Discovery doesn't call device query functions** - `cuDeviceGetCount()`, `nvmlDeviceGetCount_v2()` never called

## Key Findings from Documentation

### 1. Ollama Uses NVML First
- Discovery starts with NVML, not CUDA
- If NVML discovery fails, `libggml-cuda.so` never loads
- NVML discovery must succeed for GPU mode to activate

### 2. Ollama Doesn't Call Standard Functions
- Ollama doesn't call `cuDeviceGetCount()` or `nvmlDeviceGetCount_v2()`
- Discovery mechanism is different from standard CUDA usage
- May check library loading instead of device count

### 3. libggml-cuda.so Loading is Conditional
- Only loads if NVML discovery succeeds
- May try to load but initialization fails
- May hang during initialization

### 4. Discovery May Happen in Subprocess
- May occur in runner subprocess, not main process
- Subprocess may not have shims loaded
- Environment variables may not be inherited

## Current Issues Found

### Issue 1: Function Name Mismatch
- **Problem**: `call_libvgpu_set_skip_interception()` called but function defined as `call_call_libvgpu_set_skip_interception()`
- **Location**: `cuda_transport.c` line 992
- **Impact**: Undefined symbol error when loading libraries
- **Status**: ✅ Fixed (renamed function)

### Issue 2: Ollama Not Running
- **Problem**: Ollama service not running on test VM
- **Impact**: Cannot test GPU detection
- **Status**: Need to set up Ollama

### Issue 3: No Configuration
- **Problem**: No `vgpu.conf` found
- **Impact**: Ollama may not have correct environment variables
- **Status**: Need to create configuration

## Investigation Plan

### Phase 1: Fix Immediate Issues
1. ✅ Fix function name mismatch
2. Rebuild libraries with fix
3. Test library loading

### Phase 2: Set Up Ollama
1. Install Ollama if needed
2. Create systemd configuration
3. Set up environment variables
4. Create symlinks for bundled libraries

### Phase 3: Test Discovery
1. Start Ollama with strace
2. Monitor library loading
3. Check what functions Ollama calls
4. Verify NVML discovery

### Phase 4: Fix Discovery
1. Ensure NVML discovery succeeds
2. Ensure `libggml-cuda.so` loads
3. Ensure initialization completes
4. Verify GPU mode activates

## Next Steps

1. **Fix function name** - Already done
2. **Rebuild libraries** - Need to rebuild with fix
3. **Set up Ollama** - Install and configure
4. **Test discovery** - Monitor what Ollama does
5. **Fix issues found** - Address any problems

## Expected Discovery Flow

1. Ollama starts → logs "discovering available GPUs..."
2. Ollama loads `libnvidia-ml.so.1` (our NVML shim)
3. Ollama calls `nvmlInit_v2()` → should succeed
4. Ollama calls `nvmlDeviceGetCount_v2()` → should return 1
5. If count > 0, Ollama loads `libggml-cuda.so`
6. `libggml-cuda.so` loads `libcuda.so.1` (our CUDA shim)
7. CUDA initialization happens
8. GPU mode activates

## Current Hypothesis

**NVML discovery is failing, preventing `libggml-cuda.so` from loading.**

Possible reasons:
1. NVML shim not found by Ollama
2. `nvmlInit_v2()` fails
3. `nvmlDeviceGetCount_v2()` returns 0 or not called
4. Discovery happens in subprocess without shims
5. Symbol resolution fails

---

**Status**: Investigation in progress
**Next Action**: Rebuild libraries with fix, then set up Ollama for testing
