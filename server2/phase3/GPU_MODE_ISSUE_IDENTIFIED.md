# GPU Mode Issue Identified

## Date: 2026-02-25

## Current Status

### ✅ Working Components

1. **Device Discovery**: ✓ WORKING
   - VGPU-STUB detected at 0000:00:05.0
   - Correct vendor/device/class IDs (0x10de, 0x2331, 0x030200)
   - Logs show: `Found VGPU-STUB at 0000:00:05.0`

2. **GPU Initialization**: ✓ WORKING
   - `cuInit()` succeeds
   - `nvmlInit()` succeeds
   - GPU defaults applied: H100 80GB, CC=9.0, VRAM=81920 MB

3. **Ollama Service**: ✓ RUNNING
   - Service status: `active (running)`
   - No segfaults
   - Process stable

4. **Shim Libraries**: ✓ DEPLOYED
   - All shim libraries present in `/usr/lib64/`
   - LD_PRELOAD configured in service drop-in

### ❌ Issue Identified

**Ollama is using CPU mode instead of GPU mode**

**Symptoms:**
- Logs show: `msg="inference compute" id=cpu library=cpu`
- Bootstrap discovery shows: `initial_count=0` (no GPUs detected)
- `cuDeviceGetCount()` is NOT being called (no logs found)

**Root Cause:**
Ollama's runner subprocess (which performs GPU discovery) is NOT loading our shim libraries. This means:
1. `cuDeviceGetCount()` is not intercepted → returns 0 (no GPUs)
2. `initial_count=0` → Ollama thinks there are no GPUs
3. Result: Ollama falls back to CPU mode

## Technical Details

### Discovery Process

Ollama performs GPU discovery in two phases:

1. **Main Process**: 
   - Loads our shims (LD_PRELOAD is set)
   - Device discovery works (VGPU-STUB found)
   - GPU initialization works (cuInit, nvmlInit succeed)

2. **Runner Subprocess**:
   - Spawned by Ollama for bootstrap discovery
   - Should inherit LD_PRELOAD via libvgpu-exec.so
   - **PROBLEM**: Not loading our shims
   - Calls real CUDA library → returns 0 devices
   - Result: `initial_count=0`

### Why Runner Subprocess Doesn't Load Shims

Possible causes:

1. **libvgpu-exec.so not intercepting exec()**: 
   - The library should intercept `execve()` and inject LD_PRELOAD
   - No logs from libvgpu-exec.so found
   - May not be intercepting the specific exec call

2. **Ollama using different spawning mechanism**:
   - Ollama may use Go's runtime to spawn subprocess
   - May bypass standard exec() calls
   - May set environment differently

3. **LD_PRELOAD not inherited**:
   - Runner subprocess may clear environment
   - Or set its own environment without LD_PRELOAD

## Solution Approaches

### Approach 1: Verify libvgpu-exec.so is Working

1. Add logging to libvgpu-exec.so to confirm it's intercepting exec calls
2. Verify LD_PRELOAD is being injected into runner subprocess
3. Check if runner subprocess actually has LD_PRELOAD set

### Approach 2: Alternative Injection Method

1. Use `force_load_shim` wrapper (already in service file)
2. Ensure it properly sets LD_PRELOAD for subprocesses
3. Or modify Ollama's subprocess spawning to inherit LD_PRELOAD

### Approach 3: Direct Library Loading

1. Use `dlopen()` to load shims in runner subprocess
2. Or modify libggml-cuda.so to load our shims
3. Or use LD_AUDIT to intercept library loading

## Next Steps

1. **Verify libvgpu-exec.so interception**:
   - Add debug logging to confirm exec() interception
   - Check if LD_PRELOAD is injected into runner subprocess

2. **Check runner subprocess environment**:
   - Verify LD_PRELOAD is set in runner process
   - Check if shim libraries are loaded in runner

3. **Test alternative approaches**:
   - Try different LD_PRELOAD injection methods
   - Or modify how Ollama spawns runner subprocess

## Files Involved

- `phase3/guest-shim/libvgpu_exec.c` - Exec interception library
- `/etc/systemd/system/ollama.service.d/vgpu.conf` - Service configuration
- `phase3/guest-shim/libvgpu_cuda.c` - CUDA shim (cuDeviceGetCount implementation)

## Status Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Device Discovery | ✅ Working | VGPU-STUB detected |
| GPU Initialization | ✅ Working | cuInit, nvmlInit succeed |
| Ollama Service | ✅ Running | No crashes |
| Shim Libraries | ✅ Deployed | All libraries present |
| Runner Subprocess | ❌ Not Loading Shims | cuDeviceGetCount not intercepted |
| GPU Mode | ❌ Not Active | Using CPU mode |

## Conclusion

The infrastructure is in place and working correctly. The issue is that Ollama's runner subprocess is not loading our shim libraries, causing it to use the real CUDA library (which returns 0 devices) instead of our shims (which should return 1 device). This results in Ollama falling back to CPU mode.

The solution requires ensuring that the runner subprocess loads our shim libraries, either by fixing libvgpu-exec.so interception or using an alternative method to inject LD_PRELOAD into the subprocess.
