# Constructor Priority Fix Attempted

## Date: 2026-02-26

## Attempted Fix

Tried to update constructor priorities to 101 (early) so they run BEFORE Ollama's discovery:

1. **Updated Runtime API shim constructor** - `libvgpu_cudart.c`
2. **Updated Driver API shim constructor** - `libvgpu_cuda.c`
3. **Updated NVML shim constructor** - `libvgpu_nvml.c`

Changed from:
```c
__attribute__((constructor))
```

To:
```c
__attribute__((constructor(101)))
```

## Issue Encountered

The `sed` command used to update the constructors broke `libvgpu_cuda.c`:
- Syntax error at line 2012
- Syntax error at line 2188
- File needs to be restored from working version

## Status

- ✅ `libvgpu_cudart.c` - Updated successfully, rebuilt
- ✅ `libvgpu_nvml.c` - Updated successfully, rebuilt
- ❌ `libvgpu_cuda.c` - File corrupted, needs restoration

## Next Steps

1. **Restore `libvgpu_cuda.c` from working version**
2. **Apply constructor priority fix more carefully** (use proper editor or more precise sed)
3. **Rebuild and test**

## Constructor Priority Explanation

GCC supports constructor priorities:
- Priority 101 = runs early (before default 65535)
- Default 65535 = runs late
- Lower number = runs earlier

By using priority 101, constructors should run BEFORE Ollama's discovery, ensuring device count = 1 is available when discovery checks.
