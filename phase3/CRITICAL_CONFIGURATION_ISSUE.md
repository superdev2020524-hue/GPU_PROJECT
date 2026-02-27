# Critical Configuration Issue Found

## Date: 2026-02-26

## Issue Discovered

During GPU detection verification, I found that the main Ollama process environment still contains:
- `libvgpu-exec.so` in `LD_PRELOAD` (should be removed)
- `libvgpu-syscall.so` in `LD_PRELOAD` (should be removed)
- `OLLAMA_LIBRARY_PATH` is **MISSING** from the environment

## Current Main Process Environment

```
LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-syscall.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so
OLLAMA_DEBUG=1
OLLAMA_LLM_LIBRARY=cuda_v12
OLLAMA_NUM_GPU=999
```

**Missing**: `OLLAMA_LIBRARY_PATH`

## Required Fixes

### 1. Remove Problematic Libraries from LD_PRELOAD

The `vgpu.conf` file still contains:
```
Environment="LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-syscall.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"
```

Should be:
```
Environment="LD_PRELOAD=/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"
```

### 2. Add OLLAMA_LIBRARY_PATH

Add this line to `vgpu.conf`:
```
Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"
```

## Manual Fix Commands

```bash
# Remove problematic libraries
sudo sed -i "s|libvgpu-exec.so:||g; s|libvgpu-syscall.so:||g" /etc/systemd/system/ollama.service.d/vgpu.conf

# Add OLLAMA_LIBRARY_PATH if missing
grep -q "OLLAMA_LIBRARY_PATH" /etc/systemd/system/ollama.service.d/vgpu.conf || \
  echo 'Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"' | \
  sudo tee -a /etc/systemd/system/ollama.service.d/vgpu.conf

# Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

## Expected Result

After fixing, the main process environment should have:
```
LD_PRELOAD=/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so
OLLAMA_DEBUG=1
OLLAMA_LLM_LIBRARY=cuda_v12
OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12
OLLAMA_NUM_GPU=999
```

## Impact

Without `OLLAMA_LIBRARY_PATH`, the runner subprocess may not be able to find the shim libraries when loaded via symlinks. This prevents the constructor from detecting the runner process and initializing the shim, resulting in `initial_count=0` and `library=cpu`.

## Status

- ⚠️ Configuration needs to be fixed
- ⚠️ `libvgpu-exec.so` and `libvgpu-syscall.so` still in LD_PRELOAD
- ⚠️ `OLLAMA_LIBRARY_PATH` missing
- ✅ Shim libraries are working correctly
- ✅ GPU device is detected by shim (device count = 1)
- ❌ Discovery still shows `initial_count=0` and `library=cpu`
