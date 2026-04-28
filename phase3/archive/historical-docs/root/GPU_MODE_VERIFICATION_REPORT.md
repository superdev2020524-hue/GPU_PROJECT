# GPU Mode Verification Report

## Date: 2026-02-25

## Executive Summary

This report documents the verification and fix attempts for GPU mode activation in Ollama with vGPU support. The primary issue identified is that device discovery in Ollama returns 0x0000 values for vendor/device/class, preventing GPU detection.

## Current Status

### ✅ What's Working

1. **Test Program Device Discovery**: ✓ WORKING
   - Test program successfully finds VGPU-STUB at 0000:00:05.0
   - Returns correct values: vendor=0x10de, device=0x2331, class=0x030200
   - Command: `LD_PRELOAD=/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-cudart.so:/usr/lib64/libvgpu-nvml.so ./test_discover`

2. **Real System Files**: ✓ CORRECT VALUES
   - `/sys/bus/pci/devices/0000:00:05.0/vendor` = 0x10de
   - `/sys/bus/pci/devices/0000:00:05.0/device` = 0x2331
   - `/sys/bus/pci/devices/0000:00:05.0/class` = 0x030200

3. **Shim Libraries**: ✓ DEPLOYED
   - All shim libraries built and deployed to `/usr/lib64/`
   - Libraries loaded via LD_PRELOAD in Ollama service

### ❌ What's Not Working

1. **Ollama Device Discovery**: ❌ FAILING
   - Ollama logs show: `Found 0000:00:05.0: vendor=0x0000 device=0x0000 class=0x000000`
   - Expected: `vendor=0x10de device=0x2331 class=0x030200`
   - Result: `VGPU-STUB not found in /sys/bus/pci/devices`

2. **Skip Flag Not Working**: ❌ ISSUE
   - Skip flag debug messages don't appear in logs
   - `fopen()` logs show `skip_flag=0` for all calls
   - Skip flag should be set to 1 at start of `cuda_transport_discover()`

3. **File Interception**: ❌ RETURNING ZEROS
   - Files are being intercepted but returning 0x0000 instead of real values
   - This prevents device discovery from finding the vGPU

## Root Cause Analysis

### Primary Issue

The skip flag mechanism is not working correctly in Ollama's context:

1. **Skip Flag Not Set**: Debug messages from `libvgpu_set_skip_interception()` don't appear, suggesting the function isn't being called or the messages are suppressed.

2. **Files Return Zeros**: Even though real files contain correct values, intercepted files return 0x0000, preventing device discovery.

3. **Thread-Local vs Process-Global**: The skip flag was originally `__thread` (thread-local), which was changed to process-global with mutex protection. However, the issue persists.

### Technical Details

1. **Skip Flag Implementation**:
   - Location: `phase3/guest-shim/libvgpu_cuda.c`
   - Variable: `static int g_skip_pci_interception = 0;` (process-global)
   - Mutex: `pthread_mutex_t g_skip_flag_mutex`
   - Setter: `libvgpu_set_skip_interception(int skip)`
   - Called from: `cuda_transport_discover()` at line 962

2. **File Interception**:
   - `fopen()` interceptor checks skip flag and uses syscall when set
   - `fgets()` interceptor checks skip flag and uses syscall read when set
   - Both should bypass interception when skip flag is 1

3. **Expected Flow**:
   ```
   cuda_transport_discover() called
   → libvgpu_set_skip_interception(1)  [should set flag]
   → find_vgpu_device() called
   → fopen() called for PCI files
   → fopen() checks skip_flag=1, uses syscall
   → fgets() called, checks skip_flag=1, uses syscall read
   → Real values read (0x10de, 0x2331, 0x030200)
   → Device found
   ```

4. **Actual Flow** (based on logs):
   ```
   cuda_transport_discover() called
   → [skip flag setting - no debug message appears]
   → find_vgpu_device() called
   → fopen() called, skip_flag=0 [flag not set!]
   → Files intercepted, return 0x0000
   → Device not found
   ```

## Fixes Applied

### 1. Process-Global Skip Flag

**Problem**: Skip flag was `__thread` (thread-local), making it invisible across threads.

**Solution**: Changed to process-global with mutex protection:
```c
static int g_skip_pci_interception = 0;
static pthread_mutex_t g_skip_flag_mutex = PTHREAD_MUTEX_INITIALIZER;
```

**Status**: ✓ Implemented, but issue persists

### 2. Application Process Path Fix

**Problem**: Application processes (Ollama) showed "ERROR: Cannot resolve real fopen()".

**Solution**: Added syscall fallback for application processes when `g_real_fopen_global` is NULL.

**Status**: ✓ Implemented

### 3. Skip Flag Mutex Protection

**Problem**: Skip flag access needed thread safety.

**Solution**: All skip flag accesses now use mutex:
```c
pthread_mutex_lock(&g_skip_flag_mutex);
int skip_flag = g_skip_pci_interception;
pthread_mutex_unlock(&g_skip_flag_mutex);
```

**Status**: ✓ Implemented

## Remaining Issues

### Issue 1: Skip Flag Not Being Set

**Symptoms**:
- No debug messages from `libvgpu_set_skip_interception()`
- No debug messages from `cuda_transport_discover()` skip flag setting
- `fopen()` logs show `skip_flag=0` for all calls

**Possible Causes**:
1. `cuda_transport_discover()` is not being called (unlikely - we see "VGPU-STUB not found")
2. Skip flag setting code is not executed
3. Debug messages are suppressed
4. Code on VM is different from deployed code

**Next Steps**:
1. Verify deployed library has latest code
2. Add more debug logging to trace execution
3. Check if `libvgpu_set_skip_interception()` is actually being called
4. Verify function is exported correctly (✓ confirmed exported)

### Issue 2: Files Returning Zeros

**Symptoms**:
- Real files contain correct values (0x10de, 0x2331, 0x030200)
- Intercepted files return 0x0000
- Device discovery fails because values don't match

**Possible Causes**:
1. Skip flag not set, so files are intercepted
2. Interception returns 0 instead of real values
3. `fgets()` interception not working correctly

**Next Steps**:
1. Ensure skip flag is set before file operations
2. Verify `fgets()` interception logic
3. Check if files are opened before skip flag is set

## Recommendations

### Immediate Actions

1. **Add More Debug Logging**:
   - Add debug messages at every step of `cuda_transport_discover()`
   - Log when skip flag is checked in `fopen()` and `fgets()`
   - Log file values read (before and after interception)

2. **Verify Code Deployment**:
   - Ensure latest code is actually deployed
   - Check library timestamps
   - Verify function symbols in deployed library

3. **Test Skip Flag Directly**:
   - Create a test program that calls `libvgpu_set_skip_interception(1)`
   - Verify skip flag is actually set
   - Test file reading with skip flag set

### Long-Term Solutions

1. **Alternative Approach**: Instead of skip flag, use a different mechanism:
   - Check caller stack in `fopen()`/`fgets()` to detect `cuda_transport.c`
   - Use `is_caller_from_our_code()` to bypass interception
   - This is already implemented but may need refinement

2. **Direct Syscall Approach**: Always use syscall for `cuda_transport.c`:
   - Detect calls from `cuda_transport.c` using stack inspection
   - Bypass interception entirely for these calls
   - This avoids skip flag mechanism entirely

3. **File Path-Based Detection**: Check file path in interceptors:
   - If path is `/sys/bus/pci/devices/*/vendor|device|class`
   - And caller is from `cuda_transport.c`
   - Use syscall directly

## Files Modified

1. **`phase3/guest-shim/libvgpu_cuda.c`**:
   - Changed skip flag from `__thread` to process-global
   - Added mutex protection
   - Added syscall fallback for application processes
   - Updated all skip flag accesses to use mutex

2. **`phase3/guest-shim/cuda_transport.c`**:
   - Skip flag setting code already present (line 962)
   - Debug messages already present

## Verification Commands

### Check Device Discovery
```bash
cd /tmp
LD_PRELOAD=/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-cudart.so:/usr/lib64/libvgpu-nvml.so ./test_discover
```

### Check Real File Values
```bash
cat /sys/bus/pci/devices/0000:00:05.0/vendor
cat /sys/bus/pci/devices/0000:00:05.0/device
cat /sys/bus/pci/devices/0000:00:05.0/class
```

### Check Ollama Logs
```bash
sudo journalctl -u ollama --since "10 minutes ago" | grep -E "VGPU-STUB|device found|skip_flag"
```

### Check Skip Flag Function Export
```bash
nm -D /usr/lib64/libvgpu-cuda.so | grep skip
```

## Conclusion

The GPU mode verification identified that device discovery works in isolation (test program) but fails in Ollama's context. The root cause is that files are being intercepted and returning 0x0000 values instead of real values, preventing device discovery. The skip flag mechanism should prevent this, but it's not working correctly - skip flag debug messages don't appear, and `fopen()` logs show `skip_flag=0` for all calls.

The fixes applied (process-global skip flag, mutex protection, syscall fallback) are correct in principle but haven't resolved the issue. Further investigation is needed to determine why the skip flag isn't being set or why it's not visible to the interceptors.

## Next Steps

1. Add comprehensive debug logging to trace execution flow
2. Verify deployed code matches latest source
3. Test skip flag mechanism in isolation
4. Consider alternative approaches (stack inspection, direct syscall)
5. Document working solution once issue is resolved
