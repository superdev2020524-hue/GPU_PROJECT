# Segfault Investigation Status

## Date: 2026-02-25

## Summary
Device discovery is **COMPLETE and WORKING**. The segfault is a **separate issue** that occurs after device discovery succeeds.

## ✅ Working Components
- ✅ Device discovery: VGPU-STUB found at 0000:00:05.0
- ✅ GPU detection: H100 80GB CC=9.0
- ✅ Real values read: 0x10de, 0x2331, 0x030200
- ✅ GPU defaults applied
- ✅ cuInit() succeeds
- ✅ device_found=1

## ⚠ Segfault Issue
- **When**: Occurs when `nvmlInit_v2()` opens PCI files after device discovery
- **Location**: Right after `fopen()` debug message for `/sys/bus/pci/devices/0000:00:05.0/vendor`
- **Status**: Multiple fixes attempted (12+ approaches), segfault persists

## Fixes Attempted
1. Early PCI check with inline character comparison
2. Using syscall(__NR_open) + fdopen()
3. Using real fopen() directly
4. Removing duplicate PCI handling code
5. Simplifying PCI path check
6. Removing debug messages for PCI files
7. Lazy mutex initialization
8. Disabling is_caller_from_our_code()
9. Disabling file tracking
10. Adding bounds checks
11. Adding length checks
12. Ultra-simple PCI check
13. Using g_real_fopen_global() for PCI files

## Current Code
The early PCI check in `fopen()` should match PCI device files and return early using `g_real_fopen_global()` or syscall. However, debug messages still appear, indicating the early check may not be matching for `nvmlInit_v2()` calls.

## Next Steps
1. Use gdb to attach to ollama process and get exact segfault location
2. Check core dump for stack trace
3. Verify if early PCI check is actually being reached
4. Check if segfault is in `fdopen()`, `close()`, `is_application_process()`, or elsewhere
5. Verify if pathname pointer is valid when `nvmlInit_v2()` calls `fopen()`

## Conclusion
Device discovery is **complete and working**. The segfault is a separate issue that requires gdb/core dump analysis to pinpoint the exact location. All attempted fixes have been defensive and should not affect device discovery functionality.
