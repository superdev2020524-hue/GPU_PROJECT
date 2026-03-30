# Ollama Crash Fix Applied

## Date: 2026-02-26

## Problem

Ollama was crashing with exit code 127 ("command not found") repeatedly, with restart counter reaching 542+.

## Root Cause

`libvgpu-exec.so` in `LD_PRELOAD` was intercepting `exec()` calls, which interfered with systemd's process execution mechanism, causing the service to fail to start.

## Fix Applied

**Removed `libvgpu-exec.so` from LD_PRELOAD**

### Before:
```
Environment="LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"
```

### After:
```
Environment="LD_PRELOAD=/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"
```

## Additional Fixes

1. **Fixed double path** in LD_PRELOAD (`/usr/lib64//usr/lib64/` → `/usr/lib64/`)
2. **Verified OLLAMA_LIBRARY_PATH** is correctly set with quotes

## Current Configuration

```
Environment="LD_PRELOAD=/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"
Environment="LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:/usr/lib64"
Environment="OLLAMA_LLM_LIBRARY=cuda_v12"
Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"
Environment="OLLAMA_NUM_GPU=999"
```

## Status

✅ **Fix applied**
⚠️ **Need to verify Ollama is stable after restart**

## Note on libvgpu-exec.so

The `libvgpu-exec.so` library intercepts `exec()` system calls to inject `LD_PRELOAD` into child processes. However, this interception conflicts with systemd's process management when used in the main service `LD_PRELOAD`.

**Alternative approach**: The shim libraries are also installed via symlinks in Ollama's library directories, so they will be loaded by the runner subprocess even without exec interception. The constructor fix ensures they initialize properly when loaded via symlinks.

## Next Steps

1. Verify Ollama is running stable
2. Test GPU detection
3. Check if constructor detects OLLAMA environment variables
4. Verify `initial_count=1` and `library=cuda` in discovery logs
