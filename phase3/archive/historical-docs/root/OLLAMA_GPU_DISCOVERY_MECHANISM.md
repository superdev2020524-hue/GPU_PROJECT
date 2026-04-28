# Ollama GPU Discovery Mechanism - Deep Analysis

## Critical Finding: Ollama Uses Bundled CUDA Libraries

### Discovery

Ollama does **NOT** use system CUDA libraries. Instead, it uses **bundled CUDA libraries** located in:
- `/usr/local/lib/ollama/cuda_v12/`
- `/usr/local/lib/ollama/cuda_v13/`

### Bundled Library Structure

```
/usr/local/lib/ollama/
├── cuda_v12/
│   ├── libcuda.so -> /usr/lib64/libvgpu-cuda.so (our shim ✓)
│   ├── libcuda.so.1 -> /usr/lib64/libvgpu-cuda.so (our shim ✓)
│   ├── libcudart.so.12 -> libcudart.so.12.8.90
│   ├── libcublas.so.12 -> libcublas.so.12.8.4.1
│   └── libggml-cuda.so (CUDA backend)
├── cuda_v13/
│   ├── libcuda.so -> /usr/lib64/libvgpu-cuda.so (our shim ✓)
│   ├── libcuda.so.1 -> /usr/lib64/libvgpu-cuda.so (our shim ✓)
│   ├── libcudart.so.13 -> libcudart.so.13.0.96
│   ├── libcublas.so.13 -> libcublas.so.13.1.0.3
│   └── libggml-cuda.so (CUDA backend)
└── libcuda.so.1 -> /usr/lib64/libvgpu-cuda.so (our shim ✓)
```

### Why This Matters

1. **Ollama loads libraries from bundled directories first**
   - These directories are likely in `LD_LIBRARY_PATH` or hardcoded paths
   - System libraries in `/usr/lib64` are NOT used for CUDA

2. **Our shims are already in place**
   - All bundled `libcuda.so` and `libcuda.so.1` point to our shim
   - This was done previously (Feb 24 02:59)

3. **But libraries still aren't loading**
   - Despite correct symlinks, libraries don't appear in process memory
   - This suggests a different issue

## Ollama's Discovery Flow

Based on logs and investigation:

### Phase 1: Main Process (`ollama serve`)

1. **Startup**: Ollama binary starts
2. **Discovery Trigger**: Logs "discovering available GPUs..." from `runner.go:67`
3. **Library Loading**: 
   - Should load `libnvidia-ml.so.1` for NVML discovery
   - Should load bundled CUDA libraries when CUDA backend is needed
4. **Runner Spawn**: Spawns `ollama runner` subprocesses

### Phase 2: Runner Subprocess (`ollama runner`)

1. **GPU Discovery**: 
   - Uses NVML to enumerate GPUs
   - Matches PCI devices with NVML devices
   - Loads CUDA backend if GPU found
2. **Backend Loading**:
   - Loads `libggml-cuda.so` from bundled directory
   - This library depends on `libcuda.so.1` (our shim)
3. **Initialization**:
   - Calls `cuInit()` via our shim
   - Should match PCI bus IDs

## Current Status

### ✅ What's Working

1. **Bundled library symlinks**: All point to our shim ✓
2. **System library symlinks**: All point to our shim ✓
3. **SONAMEs**: Both CUDA and NVML shims have correct SONAME ✓
4. **Real libraries**: Moved to backup ✓

### ⚠ What's Not Working

1. **Libraries not in process memory**: 0 libraries found
2. **GPU mode still CPU**: No CUDA backend loaded
3. **No initialization messages**: No `cuInit()` or `nvmlInit()` logs

## Why Libraries Still Aren't Loading

### Hypothesis 1: Discovery Happens in Runner, Not Main Process

- Main process might not load CUDA libraries
- Discovery and backend loading might happen in runner subprocess
- Runner subprocesses might not inherit library paths correctly

**Investigation needed**: Check if runner subprocesses have libraries loaded

### Hypothesis 2: Ollama Uses Direct Library Loading

- Ollama might use `dlopen()` with absolute paths to bundled libraries
- If path is hardcoded, symlinks might not be followed
- Need to check if Ollama uses `dlopen("/usr/local/lib/ollama/cuda_v12/libcuda.so.1", ...)`

**Investigation needed**: Trace `dlopen()` calls to see what paths are used

### Hypothesis 3: Library Loading is Conditional

- Ollama might only load CUDA libraries if certain conditions are met
- If NVML discovery fails, CUDA libraries might never be loaded
- Need to verify NVML shim is working

**Investigation needed**: Check if NVML discovery succeeds

### Hypothesis 4: Go Runtime Bypasses Library Resolution

- Go runtime might use direct syscalls for library loading
- Might bypass normal `dlopen()` mechanism
- Might not follow symlinks correctly

**Investigation needed**: Check if Go uses different mechanism

## Next Steps

1. **Trace dlopen() calls**: Use `strace` to see what paths Ollama uses
2. **Check runner subprocesses**: Verify if libraries load in runner, not main
3. **Test NVML shim**: Verify NVML discovery succeeds
4. **Check library loading order**: See if there's a specific order required
5. **Investigate Go runtime**: Check if Go uses different library loading mechanism

## Key Questions

1. **Does Ollama call `dlopen()` at all?**
   - Or does it use a different mechanism?

2. **Where does discovery happen?**
   - Main process or runner subprocess?

3. **What triggers CUDA library loading?**
   - Is it conditional on NVML discovery success?

4. **Are bundled libraries loaded via absolute paths?**
   - If so, do symlinks work?

5. **Does Go runtime interfere?**
   - Does it bypass normal library resolution?

## Files Modified

All bundled library symlinks already point to our shim:
- `/usr/local/lib/ollama/libcuda.so.1` → `/usr/lib64/libvgpu-cuda.so`
- `/usr/local/lib/ollama/cuda_v12/libcuda.so` → `/usr/lib64/libvgpu-cuda.so`
- `/usr/local/lib/ollama/cuda_v12/libcuda.so.1` → `/usr/lib64/libvgpu-cuda.so`
- `/usr/local/lib/ollama/cuda_v13/libcuda.so` → `/usr/lib64/libvgpu-cuda.so`
- `/usr/local/lib/ollama/cuda_v13/libcuda.so.1` → `/usr/lib64/libvgpu-cuda.so`

## Conclusion

The discovery that Ollama uses bundled libraries explains why system library symlinks weren't enough. However, even with bundled library symlinks pointing to our shim, libraries still aren't loading. This suggests the issue is deeper - either:

1. Libraries are loaded in runner subprocesses (not main process)
2. Ollama uses a different loading mechanism
3. Library loading is conditional and failing
4. Go runtime interferes with library loading

Further investigation is needed to determine which of these is the actual cause.
