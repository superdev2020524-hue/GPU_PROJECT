# ELF Loader Investigation - dlopen() Resolution Issue

## Date: 2026-02-27

## ChatGPT's Analysis

**The problem is NOT CUDA anymore - it's ELF dynamic loader behavior.**

### Key Insight

GGML uses `dlopen()` to load `libcuda.so.1` at runtime, not link-time.

**Loader resolution order:**
1. DT_RPATH (hardcoded in binary)
2. DT_RUNPATH
3. LD_LIBRARY_PATH
4. Default system paths

If Ollama binary has embedded RUNPATH pointing to `/usr/lib64` first, it can override `LD_LIBRARY_PATH` ordering.

### Why We Don't See Shim Logs From Runner

**No constructor log = loader never touched our .so**

If the shim were loaded, we would see:
```
[libvgpu-cuda] constructor CALLED
```

No constructor log means the loader resolved to system libcuda instead of our shim.

## Investigation Steps

1. Check Ollama binary for RPATH/RUNPATH
2. Check what runner process actually is
3. Run LD_DEBUG to see loader resolution
4. Determine if we should replace system libcuda or patch RPATH

## Solutions

### Solution 1: Replace System libcuda (Recommended)
- Move real NVIDIA libcuda: `/usr/lib64/libcuda.so.1` → `/usr/lib64/libcuda.so.1.real`
- Copy shim: `/opt/vgpu/lib/libcuda.so.1` → `/usr/lib64/libcuda.so.1`
- Run `ldconfig`
- Most deterministic solution

### Solution 2: Patch RPATH/RUNPATH
- Use `patchelf` to remove or rewrite RUNPATH in Ollama binary
- Force loader to use our path
