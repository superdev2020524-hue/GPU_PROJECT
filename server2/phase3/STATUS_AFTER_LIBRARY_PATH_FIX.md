# Status After Library Path Fix

## Date: 2026-02-27

## Implementation Complete

### ✅ What Was Done

1. **Created `/opt/vgpu/lib/` directory:**
   - `/opt/vgpu/lib/libcuda.so.1` (shim)
   - `/opt/vgpu/lib/libcuda.so` (symlink)
   - `/opt/vgpu/lib/libcudart.so.12` (shim)

2. **Updated systemd service:**
   - Set `LD_LIBRARY_PATH=/opt/vgpu/lib:...` (first in path)
   - Removed `LD_PRELOAD` dependency

3. **Service configuration verified:**
   ```
   Environment=LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:...
   ```

### Current Status

**Runner subprocess environment (from logs):**
```
LD_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12:/opt/vgpu/lib:...
```

**Note:** Ollama reorders the path, putting its own directories first, but `/opt/vgpu/lib` is still included.

### Issue

Still seeing `initial_count=0` in discovery logs.

**Possible reasons:**
1. Ollama's path reordering puts `/usr/local/lib/ollama/cuda_v12` first
2. That directory has `libcuda.so.1` symlink pointing to `/usr/lib64/libvgpu-cuda.so`
3. But maybe the symlink resolution isn't working as expected
4. Or the runner subprocess isn't actually loading libcuda at all

### Next Steps

1. Check if runner subprocess is calling `cuDeviceGetCount()`
2. Verify symlink in `/usr/local/lib/ollama/cuda_v12/libcuda.so.1` points to correct shim
3. Consider making `/opt/vgpu/lib` the ONLY source of libcuda (remove symlinks from Ollama directories)
4. Or ensure `/opt/vgpu/lib` comes before Ollama directories in the final path
