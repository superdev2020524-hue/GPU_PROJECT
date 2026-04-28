# Final Status: Library Path Approach Implementation

## Date: 2026-02-27

## Implementation Summary

### ✅ Completed

1. **Created `/opt/vgpu/lib/` with shim libraries:**
   - `libcuda.so.1` → shim (127KB)
   - `libcuda.so` → symlink to libcuda.so.1
   - `libcudart.so.12` → shim (31KB)

2. **Updated systemd service:**
   - `LD_LIBRARY_PATH=/opt/vgpu/lib:...` (first in path)
   - Verified: `systemctl show ollama` shows correct environment

3. **Verified library loading:**
   - `ldd libggml-cuda.so` with `/opt/vgpu/lib` first → correctly loads shim ✅

### Current Status

**Main process:**
- ✅ Has `LD_LIBRARY_PATH` with `/opt/vgpu/lib` first
- ✅ Calls `cuDeviceGetCount()` → returns 1
- ✅ Shim loaded and working

**Runner subprocess:**
- ✅ Has `LD_LIBRARY_PATH` (includes `/opt/vgpu/lib`, but Ollama reorders it)
- ⚠️ Still reports `initial_count=0` in discovery logs
- ⚠️ No shim logs from runner subprocess PID visible

### Key Finding

**Ollama reorders `LD_LIBRARY_PATH` for runner subprocess:**
```
Systemd sets: /opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:...
Ollama uses:  /usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12:/opt/vgpu/lib:...
```

This puts Ollama's directories first, but `/opt/vgpu/lib` is still in the path.

### Next Investigation

1. Check if runner subprocess is actually calling CUDA functions
2. Verify if symlinks in `/usr/local/lib/ollama/cuda_v12/` are interfering
3. Consider removing/replacing symlinks in Ollama directories to force use of `/opt/vgpu/lib`
4. Check if discovery happens before library loading completes

### Files Created

- `/opt/vgpu/lib/libcuda.so.1` - Shim library
- `/opt/vgpu/lib/libcuda.so` - Symlink
- `/opt/vgpu/lib/libcudart.so.12` - CUDART shim
- `/etc/systemd/system/ollama.service.d/vgpu.conf` - Updated service config

### Architecture

This follows ChatGPT's recommended approach:
- ✅ No LD_PRELOAD dependency
- ✅ Proper libcuda.so.1 replacement
- ✅ Subprocess-safe via LD_LIBRARY_PATH
- ✅ Industry-standard approach
