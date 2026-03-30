# Critical Finding: Shim IS Loaded in Runner Subprocess!

## Date: 2026-02-27

## Discovery

**The shim IS being loaded in the runner subprocess!**

### Evidence from Logs

```
[libvgpu-cudart] cudaGetDeviceCount() CALLED (pid=212717)
[libvgpu-cudart] cudaGetDeviceCount() SUCCESS: returning count=1 (pid=212717)
[libvgpu-cuda] cuDeviceGet() SUCCESS: device=...
[libvgpu-cuda] cuDeviceGetAttribute() CALLED (attrib=102, dev=0, pid=212717)
[libvgpu-nvml] constructor CALLED (initializing early for discovery)
```

**pid=212717 is the runner subprocess, and it's calling our shim functions!**

## The Real Problem

**This is NOT a loader issue anymore!**

The shim is loading correctly. The issue must be:

1. **Discovery happens BEFORE these CUDA calls**
   - Discovery runs at bootstrap (232ms)
   - CUDA calls happen later during model execution
   - Discovery may be checking something else

2. **Discovery uses different code path**
   - Maybe discovery uses NVML instead of CUDA?
   - Or uses a different validation that fails?

3. **Timing issue**
   - Discovery runs before shim is fully initialized?
   - Or before context is created?

## Next Investigation

1. Check when discovery runs vs when CUDA calls happen
2. Check if discovery uses NVML or CUDA
3. Check if there's a validation step that's failing
4. Look at the actual discovery code path in GGML

## Status

- ✅ Shim loads in runner subprocess
- ✅ CUDA functions are called and work
- ❌ Discovery still reports `initial_count=0`
- ❌ This is NOT a loader issue - it's a discovery/validation issue
