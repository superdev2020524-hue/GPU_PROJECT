# Backend Scanner Investigation

## Date: 2026-02-26

## Current Situation

### ✅ What's Working
- Device discovery: VGPU-STUB found
- Runtime API shim: ✓ Loaded (`libvgpu-cudart.so` in process memory)
- Driver API shim: ✓ Loaded (`libvgpu-cuda.so` in process memory)
- Constructor: ✓ Called (logs show it's executing)
- All symlinks correct:
  - `libggml-cuda.so` copied (not symlinked) in top-level directory
  - `libcublas.so.12` and `libcublasLt.so.12` symlinked
  - All CUDA library symlinks point to our shims

### ❌ What's Not Working
- `libggml-cuda.so` is NOT being opened during discovery
- `initial_count=0` (no GPUs detected)
- GPU mode is CPU (`library=cpu`)
- Device count functions are NOT being called

## Key Finding

**Previous work (BREAKTHROUGH_LIBGGML_LOADING.md) showed:**
- ✅ `libggml-cuda.so` WAS being opened from `cuda_v12/` directly
- ❌ But discovery timed out (initialization issue)

**Current situation:**
- ❌ `libggml-cuda.so` is NOT being opened at all
- ❌ This is DIFFERENT from previous work

## Investigation Results

### 1. Backend Scanner Behavior
- Strace shows NO opens of `libggml-cuda.so` (neither from top-level nor from `cuda_v12/`)
- Other `libggml-*.so` files are NOT being opened either (only the binary itself)
- This suggests backend scanner may not be running, or is skipping all backends

### 2. Shim Loading
- Runtime API shim: ✓ Loaded via LD_PRELOAD
- Driver API shim: ✓ Loaded via LD_PRELOAD
- Constructor: ✓ Called (initial log message appears)
- But: No logs from constructor about cuInit() or device counts

### 3. Constructor Behavior
- Constructor log file shows only initial "constructor CALLED" message
- No subsequent logs about cuInit(), cuDeviceGetCount(), or cudaGetDeviceCount()
- This suggests either:
  - cuInit() is not found via RTLD_DEFAULT
  - Code fails before reaching device count calls
  - Logging is not working properly

### 4. File Status
- `libggml-cuda.so` exists as regular file (1.6GB) in top-level directory
- File is valid ELF shared object
- Permissions are correct (same as CPU backends)
- File is accessible

## Possible Causes

1. **Backend scanner not running**
   - Maybe only runs under certain conditions
   - Or is disabled/failing silently

2. **Device check required first**
   - Maybe backend scanner checks device availability BEFORE loading
   - If no devices found, skips loading CUDA backend
   - But device count functions aren't being called, so count is 0

3. **Constructor not completing**
   - Constructor starts but doesn't complete device count calls
   - Maybe cuInit() not found, so subsequent calls don't happen
   - This would mean device count is never set to 1

4. **Ollama version/configuration change**
   - Behavior different from previous work
   - Maybe backend scanner logic changed

## Next Steps

1. **Verify constructor completes**
   - Check if cuInit() is found via RTLD_DEFAULT
   - Ensure device count functions are called
   - Verify they return count=1

2. **Force device count early**
   - Ensure device count is 1 BEFORE backend scanner runs
   - Maybe need to call device count functions in a different way

3. **Check backend scanner conditions**
   - Understand what triggers backend scanner
   - Verify it's actually running
   - Check if there are conditions that prevent it from loading CUDA backend

4. **Compare with previous working state**
   - Review what was different when it was working
   - Check if there were additional steps or configuration

## Conclusion

**The backend scanner is not opening `libggml-cuda.so`, even though:**
- File exists and is accessible
- All symlinks are correct
- Shims are loaded
- Constructor is called

**This suggests the backend scanner either:**
- Requires device count to be 1 BEFORE it runs (but device count functions aren't being called)
- Has different behavior than previous work
- Is not running at all

**The key issue is that device count functions are not being called, so the backend scanner sees 0 devices and skips the CUDA backend.**
