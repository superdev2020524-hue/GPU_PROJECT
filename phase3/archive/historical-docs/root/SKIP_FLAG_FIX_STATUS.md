# Skip Flag Fix Status

## Date: 2026-02-25

## Summary

Multiple fixes have been applied to ensure the skip flag is set before `find_vgpu_device()` reads PCI device files, but the issue persists - logs still show `skip_flag=0` and files return `vendor=0x0000`.

## Fixes Applied

### 1. Process-Global Skip Flag
- Changed from `__thread` (thread-local) to process-global with mutex protection
- Location: `phase3/guest-shim/libvgpu_cuda.c`
- Status: ✓ Implemented

### 2. Skip Flag in cuda_transport_init()
- Added `libvgpu_set_skip_interception(1)` at start of `cuda_transport_init()`
- Location: `phase3/guest-shim/cuda_transport.c` line ~464
- Status: ✓ Implemented

### 3. Skip Flag in find_vgpu_device()
- Added `libvgpu_set_skip_interception(1)` at the VERY START of `find_vgpu_device()`
- Added FORCE debug messages using `write()` syscall
- Location: `phase3/guest-shim/cuda_transport.c` line ~177
- Status: ✓ Implemented and verified in deployed library

### 4. fgets() Skip Flag Handling
- Modified `fgets()` to ALWAYS use syscall read when `skip_flag=1`
- Removed dependency on `g_real_fgets_global`
- Location: `phase3/guest-shim/libvgpu_cuda.c` line ~1640
- Status: ✓ Implemented

## Current Status

### What's Working
- ✓ Test program device discovery works
- ✓ Real system files contain correct values (0x10de, 0x2331, 0x030200)
- ✓ Skip flag setting code is in source and deployed library
- ✓ Library strings confirm new code is present

### What's Not Working
- ❌ Skip flag is still 0 in all `fopen()` calls
- ❌ FORCE debug messages don't appear in logs
- ❌ Files return `vendor=0x0000 device=0x0000 class=0x000000`
- ❌ Device discovery fails: "VGPU-STUB not found"

## Mystery

The code is correct and deployed, but:
1. FORCE messages don't appear (even though strings are in library)
2. Skip flag is 0 (even though it should be set to 1)
3. Files return zeros (even though real files have correct values)

## Possible Causes

1. **Timing Issue**: Skip flag is set but immediately reset before `fopen()` is called
2. **Race Condition**: Multiple threads accessing skip flag simultaneously
3. **Code Path**: `find_vgpu_device()` is called from a different path that bypasses skip flag setting
4. **Library Loading**: Old library is still loaded in memory
5. **Message Suppression**: FORCE messages are being suppressed or redirected

## Next Steps

1. Verify library is actually loaded (check `/proc/PID/maps`)
2. Add more aggressive debugging (maybe use `syslog()` instead of `write()`)
3. Check if there are multiple instances of the library loaded
4. Verify `libvgpu_set_skip_interception()` is actually being called (add breakpoint or more logging)
5. Consider alternative approach: Use `is_caller_from_our_code()` to bypass interception instead of skip flag

## Files Modified

1. `phase3/guest-shim/libvgpu_cuda.c`:
   - Skip flag made process-global
   - `fgets()` skip flag handling improved
   - `libvgpu_set_skip_interception()` with FORCE messages

2. `phase3/guest-shim/cuda_transport.c`:
   - Skip flag setting in `cuda_transport_init()`
   - Skip flag setting in `find_vgpu_device()`
   - FORCE debug messages added

## Verification Commands

```bash
# Check if library has new code
sudo strings /usr/lib64/libvgpu-cuda.so | grep "setting skip flag"

# Check library timestamp
ls -lh /usr/lib64/libvgpu-cuda.so

# Check if library is loaded
sudo cat /proc/$(pgrep -f "ollama serve" | head -1)/maps | grep libvgpu-cuda

# Check logs
sudo journalctl -u ollama --since "1 minute ago" | grep -E "FORCE|skip_flag"
```
