# Strace Findings: Discovery Opens Our Symlink

## Key Discovery

**strace shows discovery opens `/usr/local/lib/ollama/libcuda.so.1` - our symlink!**

### Evidence from strace

```
55047 openat(AT_FDCWD, "/usr/local/lib/ollama/libcuda.so.1", O_RDONLY|O_CLOEXEC) = 9
55047 openat(AT_FDCWD, "/usr/local/lib/ollama/cuda_v12/libggml-cuda.so", O_RDONLY|O_CLOEXEC) = 9
```

Discovery is:
1. Opening `/usr/local/lib/ollama/libcuda.so.1` (our symlink ✓)
2. Opening `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so` (depends on libcuda.so.1)

### Dependency Chain

```
libggml-cuda.so
  └─> depends on libcuda.so.1
      └─> /usr/local/lib/ollama/libcuda.so.1 -> /usr/lib64/libvgpu-cuda.so (our shim ✓)
```

### What We Verified

1. **All symlinks are correct:**
   - `/usr/lib64/libcuda.so.1` → `/usr/lib64/libvgpu-cuda.so` ✓
   - `/usr/local/lib/libcuda.so.1` → `/usr/lib64/libvgpu-cuda.so` ✓
   - `/usr/local/lib/ollama/libcuda.so.1` → `/usr/lib64/libvgpu-cuda.so` ✓
   - `/usr/local/lib/ollama/cuda_v12/libcuda.so.1` → `/usr/lib64/libvgpu-cuda.so` ✓
   - `/usr/local/lib/ollama/cuda_v13/libcuda.so.1` → `/usr/lib64/libvgpu-cuda.so` ✓

2. **Libraries are loading:**
   - Via LD_PRELOAD: NVML and CUDA shims in process memory ✓
   - Via dependency resolution: libggml-cuda.so should load our shim ✓

3. **Discovery is happening:**
   - Log shows: "discovering available GPUs..." ✓
   - strace shows: opens libcuda.so.1 ✓

## The Mystery

**Everything looks correct, but GPU mode is still CPU!**

### Possible Reasons

1. **Functions aren't being called**
   - Libraries load but functions aren't invoked
   - Discovery might check library existence but not call functions
   - Or functions are called but return errors

2. **Initialization isn't happening**
   - Our constructors are empty (for safety)
   - Lazy initialization only happens when functions are called
   - Maybe discovery doesn't call functions that trigger initialization

3. **Discovery fails silently**
   - Discovery might fail before calling our functions
   - Or discovery succeeds but reports CPU anyway
   - Or there's an error condition we're not seeing

4. **NVML discovery fails first**
   - Discovery might use NVML first
   - If NVML fails, it might skip CUDA entirely
   - Even though NVML shim is loaded, it might not be called

## Next Steps

1. **Verify if functions are called**
   - Add more aggressive logging
   - Check if stderr is captured
   - Verify if messages are suppressed

2. **Check NVML discovery**
   - strace didn't show NVML opens
   - Maybe NVML is loaded via dlopen() (doesn't show in openat)
   - Or NVML discovery happens differently

3. **Force function calls**
   - If discovery doesn't call functions, we might need early initialization
   - But must be extremely careful to avoid VM crashes
   - Test thoroughly before deploying

## Key Insight

**Discovery IS opening our symlink!**

This means:
- Symlinks work ✓
- Discovery finds our library ✓
- Dependency resolution should work ✓

But something is still preventing GPU mode from activating. Need to investigate why functions aren't being called or why discovery still reports CPU.
