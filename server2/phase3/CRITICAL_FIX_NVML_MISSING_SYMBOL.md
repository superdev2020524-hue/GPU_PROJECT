# Critical Fix: NVML Missing Symbol

## Date: 2026-02-27

## Root Cause Identified by ChatGPT

**The NVML shim has an undefined symbol that causes backend loading to fail.**

### The Error

```
libnvidia-ml.so.1: undefined symbol: libvgpu_set_skip_interception
```

### What's Happening

1. Ollama loads `libggml-cuda.so` during bootstrap
2. `libggml-cuda.so` depends on `libnvidia-ml.so.1` (our NVML shim)
3. Dynamic linker loads NVML shim
4. NVML shim has undefined symbol `libvgpu_set_skip_interception`
5. Symbol is defined in `libvgpu-cuda.so` but not loaded yet
6. Dynamic linker fails → backend init never runs
7. Ollama reports `initial_count=0`

### Why It Fails

The NVML shim was built with `-Wl,--allow-shlib-undefined` expecting both libraries to be loaded together via `LD_PRELOAD`. However, when `libggml-cuda.so` loads the NVML shim as a dependency, the CUDA shim might not be loaded yet.

### The Fix

Add a stub implementation of `libvgpu_set_skip_interception` directly in the NVML shim:

```c
void libvgpu_set_skip_interception(int skip)
{
    /* Stub implementation - does nothing */
    /* cuda_transport.c will use dlsym to find the real implementation
     * from libvgpu-cuda.so if it's available */
    (void)skip;
}
```

This ensures:
- Symbol is always available when NVML shim loads
- If CUDA shim is loaded later, `cuda_transport.c` will use dlsym to find the real implementation
- Backend loading succeeds

### Files Modified

- `libvgpu_nvml.c` - Added stub implementation

### Next Steps

1. Rebuild NVML shim with stub
2. Deploy to VM
3. Test backend loading
4. Verify GPU detection
