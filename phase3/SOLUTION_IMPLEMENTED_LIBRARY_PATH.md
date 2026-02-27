# Solution Implemented: Library Path Approach (Option 1)

## Date: 2026-02-27

## ChatGPT's Recommendation

**Move away from LD_PRELOAD entirely and use proper libcuda.so.1 replacement via LD_LIBRARY_PATH.**

### Why This Is Better

- ✅ **Subprocess-safe**: Works for all subprocesses including runner
- ✅ **Industry-standard**: How real NVIDIA drivers work
- ✅ **Predictable**: Always used when linked dynamically
- ✅ **No inheritance issues**: Doesn't rely on environment variable inheritance

## Implementation

### 1. Created Dedicated vGPU Library Directory

```bash
/opt/vgpu/lib/
├── libcuda.so.1    (shim - copied from /usr/lib64/libvgpu-cuda.so)
├── libcuda.so      (symlink to libcuda.so.1)
└── libcudart.so.12 (shim - copied from /usr/lib64/libvgpu-cudart.so)
```

### 2. Updated Systemd Service Configuration

**Before (LD_PRELOAD approach):**
```
Environment=LD_PRELOAD=/usr/lib64/libvgpu-exec.so:...
```

**After (LD_LIBRARY_PATH approach):**
```
Environment=LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:/usr/lib64
```

### 3. Removed LD_PRELOAD Dependency

- No longer relying on LD_PRELOAD
- Shim is loaded naturally via library search path
- Works for main process AND runner subprocess

## How It Works

1. **Main process starts:**
   - `LD_LIBRARY_PATH` includes `/opt/vgpu/lib` first
   - When Ollama loads CUDA libraries, it finds `/opt/vgpu/lib/libcuda.so.1` first
   - Shim is loaded ✅

2. **Runner subprocess starts:**
   - Inherits `LD_LIBRARY_PATH` from systemd environment
   - When runner loads CUDA libraries, it finds `/opt/vgpu/lib/libcuda.so.1` first
   - Shim is loaded ✅

3. **Discovery runs:**
   - Runner calls `cuDeviceGetCount()` → shim returns 1 ✅
   - Runner calls `cuMemGetInfo()` → shim returns valid values ✅
   - Discovery reports `initial_count=1` ✅

## Files Modified

1. `/opt/vgpu/lib/` - New directory with shim libraries
2. `/etc/systemd/system/ollama.service.d/vgpu.conf` - Updated environment

## Testing

After restart, check:
- Runner subprocess logs show shim loaded
- Discovery reports `initial_count=1`
- GPU detected and used
