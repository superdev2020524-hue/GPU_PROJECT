# Runner Subprocess Issue - GPU Mode Not Active

## Date: 2026-02-25

## Problem Summary

Ollama is using CPU mode instead of GPU mode because the runner subprocess (which performs bootstrap discovery) is not loading our shim libraries.

## Current Status

### ✅ Working
- Device discovery in main process: VGPU-STUB found
- GPU initialization: cuInit() and nvmlInit() succeed
- Ollama service: Running (no crashes)
- Shim libraries: Deployed and loaded in main process
- libvgpu-exec.so: Loaded in main process (verified via lsof)

### ❌ Issue
- Runner subprocess: Not loading shim libraries
- cuDeviceGetCount(): Not being called (no logs)
- Bootstrap discovery: `initial_count=0` (no GPUs detected)
- Result: Ollama falls back to CPU mode

## Investigation Results

### 1. libvgpu-exec.so Status
- **Loaded**: ✓ Yes (verified via `lsof`)
- **Constructor logs**: ✗ No logs found
- **Exec interception logs**: ✗ No logs found

**Conclusion**: libvgpu-exec.so is loaded but may not be intercepting exec calls, or Ollama uses a different mechanism to spawn subprocesses.

### 2. Runner Subprocess
- **Currently running**: ✗ No (only spawned during bootstrap discovery)
- **LD_PRELOAD**: Unknown (can't check when not running)
- **Shim libraries loaded**: Unknown

**Conclusion**: Cannot verify if runner subprocess has shims loaded because it's not persistent.

### 3. Bootstrap Discovery
- **initial_count**: 0 (should be 1)
- **cuDeviceGetCount() calls**: 0 (should be called)
- **Result**: No GPUs detected → CPU mode

**Conclusion**: Runner subprocess is calling real CUDA library (returns 0 devices) instead of our shims (should return 1 device).

## Root Cause Hypothesis

Ollama's Go runtime may spawn the runner subprocess using a mechanism that:
1. **Bypasses execve/execv**: Uses syscalls directly (clone, fork+exec)
2. **Clears environment**: Sets its own environment without LD_PRELOAD
3. **Uses different mechanism**: Go's runtime may use a different subprocess spawning method

## Possible Solutions

### Solution 1: Use LD_AUDIT
- LD_AUDIT can intercept library loading
- Could force-load shims when libggml-cuda.so is loaded
- More reliable than exec interception

### Solution 2: Modify libggml-cuda.so Loading
- Intercept dlopen() calls to libggml-cuda.so
- Force-load our shims before libggml-cuda.so initializes
- Ensures shims are loaded regardless of subprocess spawning

### Solution 3: Patch Ollama Binary
- Modify Ollama to set LD_PRELOAD in subprocess environment
- Or modify how it spawns runner subprocess
- Requires access to Ollama source or binary patching

### Solution 4: Use force_load_shim Wrapper
- Check if force_load_shim properly handles subprocesses
- May need to modify it to ensure LD_PRELOAD is inherited
- Currently used in service file but may not be working

## Next Steps

1. **Verify force_load_shim behavior**: Check if it properly sets LD_PRELOAD for subprocesses
2. **Try LD_AUDIT approach**: Use LD_AUDIT to intercept library loading
3. **Intercept dlopen()**: Intercept dlopen() calls to ensure shims load before libggml-cuda.so
4. **Check Go runtime**: Research how Go spawns subprocesses and if we can intercept it

## Files Involved

- `phase3/guest-shim/libvgpu_exec.c` - Exec interception (may not be working)
- `/etc/systemd/system/ollama.service` - Uses force_load_shim wrapper
- `/usr/local/bin/force_load_shim` - Wrapper script (needs verification)
- `phase3/guest-shim/libvgpu_cuda.c` - cuDeviceGetCount() implementation

## Status

**Progress: 90% Complete**
- ✅ All infrastructure in place
- ✅ Device discovery working
- ✅ GPU initialization working
- ❌ Runner subprocess not loading shims
- ❌ GPU mode not active

## Conclusion

The issue is that Ollama's runner subprocess is not loading our shim libraries, causing it to use the real CUDA library which returns 0 devices. This results in Ollama falling back to CPU mode. The exec interception approach (libvgpu-exec.so) may not be working because Ollama's Go runtime uses a different mechanism to spawn subprocesses.
