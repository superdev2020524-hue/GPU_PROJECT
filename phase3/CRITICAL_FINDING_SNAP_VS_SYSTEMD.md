# Critical Finding: Snap vs Systemd Service

## Problem Identified

You were absolutely right - I was checking the **wrong service**! 

### What Was Wrong

1. **Ollama was installed as a regular systemd service** (`ollama.service`), NOT as a snap
2. I was checking logs for `snap.ollama.listener.service` which **doesn't exist**
3. The systemd service had **no LD_PRELOAD or LD_LIBRARY_PATH configured**
4. The shims were **never being loaded** because the service wasn't configured

### What Was Fixed

1. ✅ Created `/etc/systemd/system/ollama.service.d/vgpu.conf` with:
   - `LD_PRELOAD=/opt/vgpu/lib/libcuda.so.1:/opt/vgpu/lib/libcudart.so.12`
   - `LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:/usr/lib64`

2. ✅ **Critical fix**: Changed `LD_PRELOAD` to point to `/opt/vgpu/lib/` shims instead of `/lib/x86_64-linux-gnu/` (which was pointing to real NVIDIA library)

3. ✅ Restarted service and confirmed shims are now loading

### Current Status

**Shims ARE NOW LOADING:**
- ✅ Driver API shim constructor runs (`[libvgpu-cuda] cuInit() CALLED`)
- ✅ `cuInit()` succeeds (rc=0) and finds device at `0000:00:05.0`
- ✅ `cuDeviceGetCount()` returns 1
- ✅ Runtime API shim loads (`[libvgpu-cudart] constructor CALLED`)

**But Ollama still reports `library=cpu`:**
- ❌ `cudaGetDeviceProperties` is NOT being called during discovery
- ❌ No GGML logs appear
- ❌ Discovery still reports CPU

### Next Steps

The shims are loading correctly now, but Ollama's discovery phase is not calling `cudaGetDeviceProperties`, which means it's not getting device properties (compute capability) needed for validation. Need to investigate why discovery isn't calling this function.
