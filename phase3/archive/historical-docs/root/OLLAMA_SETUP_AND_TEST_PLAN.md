# Ollama Setup and Test Plan

## Date: 2026-02-27

## Objective
Set up Ollama properly and investigate why GPU detection isn't working, even though vGPU works for general GPU projects.

## Current Status

### ✅ Working
- vGPU works for general GPU projects (Python, CUDA, etc.)
- All 3 shim libraries built and installed
- Libraries can be loaded manually

### ❌ Not Working
- Ollama doesn't detect GPU
- `libggml-cuda.so` not loaded
- GPU mode is CPU

## Issues Fixed

### Issue 1: Function Name Mismatch ✅
- **Problem**: Function defined as `call_call_libvgpu_set_skip_interception()` but called as `call_libvgpu_set_skip_interception()`
- **Fix**: Renamed function to match calls
- **Status**: ✅ Fixed

## Next Steps

### Step 1: Rebuild Libraries ✅
- Rebuild with function name fix
- Verify all 3 libraries build successfully
- Test library loading

### Step 2: Set Up Ollama Configuration
1. Create systemd service override directory
2. Create `vgpu.conf` with:
   - `LD_PRELOAD` for shim libraries
   - `LD_LIBRARY_PATH` for bundled libraries
   - `OLLAMA_LIBRARY_PATH` for scanner
   - `OLLAMA_LLM_LIBRARY=cuda_v12` (if needed)
   - `OLLAMA_NUM_GPU=999` (if needed)
3. Create symlinks for bundled libraries:
   - `/usr/local/lib/ollama/cuda_v12/libcuda.so.1` → `/usr/lib64/libvgpu-cuda.so`
   - `/usr/local/lib/ollama/cuda_v12/libnvidia-ml.so.1` → `/usr/lib64/libvgpu-nvml.so`
   - `/usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90` → `/usr/lib64/libvgpu-cudart.so`
4. Create top-level symlink:
   - `/usr/local/lib/ollama/libggml-cuda.so` → `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so`

### Step 3: Test Ollama Discovery
1. Start Ollama with strace
2. Monitor library loading:
   - Check if `libnvidia-ml.so.1` is loaded
   - Check if `libcuda.so.1` is loaded
   - Check if `libggml-cuda.so` is loaded
3. Monitor function calls:
   - Check if `nvmlInit_v2()` is called
   - Check if `nvmlDeviceGetCount_v2()` is called
   - Check if `cuInit()` is called
   - Check if `cuDeviceGetCount()` is called
4. Check discovery logs:
   - Look for "discovering available GPUs..."
   - Look for device count
   - Look for library selection

### Step 4: Fix Issues Found
Based on what we discover:
1. If NVML shim not found → Fix library paths
2. If `nvmlInit_v2()` fails → Fix initialization
3. If `nvmlDeviceGetCount_v2()` returns 0 → Fix device count
4. If `libggml-cuda.so` doesn't load → Fix loading conditions
5. If initialization hangs → Fix blocking operations

## Expected Discovery Flow

1. Ollama starts → logs "discovering available GPUs..."
2. Ollama loads `libnvidia-ml.so.1` (our NVML shim)
3. Ollama calls `nvmlInit_v2()` → should succeed
4. Ollama calls `nvmlDeviceGetCount_v2()` → should return 1
5. If count > 0, Ollama loads `libggml-cuda.so`
6. `libggml-cuda.so` loads `libcuda.so.1` (our CUDA shim) as dependency
7. CUDA initialization happens
8. GPU mode activates (`initial_count=1`, `library=cuda`)

## Key Questions to Answer

1. **Does Ollama load NVML shim?**
   - Check process memory maps
   - Check strace output

2. **Does Ollama call NVML functions?**
   - Check function call logs
   - Check strace output

3. **Does NVML discovery succeed?**
   - Check return values
   - Check device count

4. **Does libggml-cuda.so load?**
   - Check process memory maps
   - Check strace output

5. **Does libggml-cuda.so initialize?**
   - Check for initialization errors
   - Check for hanging operations

## Files to Create/Modify

1. `/etc/systemd/system/ollama.service.d/vgpu.conf` - Service configuration
2. Symlinks in `/usr/local/lib/ollama/cuda_v12/` - Bundled library symlinks
3. Symlink `/usr/local/lib/ollama/libggml-cuda.so` - Top-level symlink

## Status

- ✅ Function name fix applied
- ⏳ Libraries rebuilding
- ⏳ Ollama setup pending
- ⏳ Discovery testing pending

---

**Next Action**: Complete library rebuild, then set up Ollama configuration
