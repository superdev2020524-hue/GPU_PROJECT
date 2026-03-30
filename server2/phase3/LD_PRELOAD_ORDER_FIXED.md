# LD_PRELOAD Order Fix

## Date: 2026-02-26

## Problem

The Runtime API shim constructor could not find `cuInit()` from the Driver API shim because the Runtime API shim was loading BEFORE the Driver API shim in LD_PRELOAD.

### Root Cause

LD_PRELOAD order in `/etc/systemd/system/ollama.service.d/vgpu.conf` was:
```
libvgpu-exec.so:libvgpu-nvml.so:libvgpu-cudart.so:libvgpu-cuda.so
```

This meant:
- Runtime API shim (`libvgpu-cudart.so`) loaded BEFORE Driver API shim (`libvgpu-cuda.so`)
- When Runtime API shim constructor ran, Driver API shim symbols were not yet in global scope
- Constructor could not find `cuInit()` via `dlsym(RTLD_DEFAULT)`
- Device count functions were never called
- Device count remained 0
- Backend scanner skipped CUDA backend

## Fix

Updated `/etc/systemd/system/ollama.service.d/vgpu.conf` with correct order:
```
libvgpu-exec.so:libvgpu-syscall.so:libvgpu-cuda.so:libvgpu-nvml.so:libvgpu-cudart.so
```

### Key Changes

1. **Driver API shim (`libvgpu-cuda.so`) is now BEFORE Runtime API shim (`libvgpu-cudart.so`)**
2. Added `libvgpu-syscall.so` which was missing
3. Ensured `[Service]` section is present in the file

### Complete vgpu.conf File

```ini
[Service]
# CRITICAL: libvgpu-exec.so MUST be first to intercept exec calls
# and inject LD_PRELOAD into subprocesses (like runner)
# CRITICAL: libvgpu-cuda.so (Driver API) MUST be before libvgpu-cudart.so (Runtime API)
# so Runtime API shim constructor can find Driver API symbols
Environment="LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-syscall.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"
# CRITICAL: Add ollama directories so libggml-cuda.so can find ALL dependencies
Environment="LD_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama/cuda_v13:/usr/lib64:/usr/lib/x86_64-linux-gnu"
ExecStart=
ExecStart=/usr/local/bin/ollama serve
Environment="OLLAMA_DEBUG=1"
```

## Expected Results

After restarting Ollama with the correct order:

1. Driver API shim loads first
2. Runtime API shim constructor runs
3. Constructor finds `cuInit()` via `dlsym(RTLD_DEFAULT)` ✓
4. Constructor calls `cuInit()` ✓
5. Constructor calls `cuDeviceGetCount()` and `cudaGetDeviceCount()` ✓
6. Device count is set to 1 ✓
7. Backend scanner sees device count = 1
8. Backend scanner loads `libggml-cuda.so` ✓
9. GPU mode is active (`library=cuda`) ✓

## Verification

After applying the fix and restarting Ollama:
- Check LD_PRELOAD order in process: `cat /proc/$(pidof ollama)/environ | tr '\0' '\n' | grep LD_PRELOAD`
- Check constructor logs: `journalctl -u ollama | grep "libvgpu-cudart.*cuInit"`
- Check GPU mode: `journalctl -u ollama | grep "initial_count\|library="`

## Status

✅ LD_PRELOAD order fixed in vgpu.conf
✅ File restored with complete [Service] section
⏳ Awaiting Ollama restart to verify fix works
