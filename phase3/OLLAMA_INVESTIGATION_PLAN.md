# Ollama GPU Detection Investigation Plan

## Date: 2026-02-27

## Current Status

### ✅ What Works
- vGPU works for general GPU projects (Python, CUDA, etc.)
- All 3 shim libraries built and installed
- Symlinks in place
- Libraries can be loaded manually

### ❌ What Doesn't Work
- Ollama doesn't detect GPU
- `libggml-cuda.so` not loaded
- GPU mode is CPU

## Key Findings from Documentation

1. **Ollama uses NVML first** - Discovery starts with NVML, not CUDA
2. **Conditional loading** - `libggml-cuda.so` only loads if NVML discovery succeeds
3. **Different discovery mechanism** - Ollama doesn't call standard device query functions
4. **Bundled libraries** - Ollama uses bundled CUDA libraries, not system libraries
5. **Discovery in subprocess** - May happen in runner subprocess, not main process

## Investigation Steps

### Step 1: Verify Ollama Installation and Configuration
- [ ] Check if Ollama is installed
- [ ] Check systemd service configuration
- [ ] Verify environment variables
- [ ] Check bundled library symlinks

### Step 2: Test NVML Discovery
- [ ] Test if NVML shim can be found by Ollama
- [ ] Test if `nvmlInit_v2()` is called
- [ ] Test if `nvmlDeviceGetCount_v2()` is called
- [ ] Verify NVML discovery returns device count > 0

### Step 3: Test libggml-cuda.so Loading
- [ ] Check if `libggml-cuda.so` exists and is accessible
- [ ] Test if it can be loaded manually
- [ ] Check if dependencies are resolved
- [ ] Verify symlinks point to shims

### Step 4: Trace Ollama Discovery Process
- [ ] Use strace to see what libraries Ollama tries to load
- [ ] Check what functions Ollama calls
- [ ] Verify if discovery happens in main or runner process
- [ ] Check if there are any errors during discovery

### Step 5: Verify Shim Functions
- [ ] Ensure all required NVML functions are implemented
- [ ] Ensure all required CUDA functions are implemented
- [ ] Verify function return values are correct
- [ ] Check if symbol versions match what Ollama expects

## Expected Discovery Flow

1. Ollama starts → logs "discovering available GPUs..."
2. Ollama tries to load `libnvidia-ml.so.1` (our NVML shim)
3. Ollama calls `nvmlInit_v2()` → should succeed
4. Ollama calls `nvmlDeviceGetCount_v2()` → should return 1
5. If count > 0, Ollama loads `libggml-cuda.so`
6. `libggml-cuda.so` loads `libcuda.so.1` (our CUDA shim) as dependency
7. CUDA initialization happens
8. GPU mode activates

## Current Hypothesis

**NVML discovery is failing, so `libggml-cuda.so` never loads.**

Possible reasons:
1. NVML shim not found by Ollama
2. `nvmlInit_v2()` fails
3. `nvmlDeviceGetCount_v2()` returns 0
4. Discovery happens in subprocess without shims
5. Symbol resolution fails

## Next Actions

1. Connect to VM and check Ollama status
2. Verify configuration files
3. Test NVML discovery manually
4. Trace Ollama's discovery process
5. Fix any issues found
