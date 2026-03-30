# Segfault Debugging Findings

## Date: 2026-02-25

## Test Results

### Test 1: Return NULL for all PCI files
- **Result**: Segfault still occurs
- **Conclusion**: Segfault is NOT in the code that opens PCI files

### Test 2: Disable early PCI check
- **Result**: Segfault still occurs
- **Conclusion**: Segfault is NOT in the early PCI check character comparison logic

### Test 3: Disable is_application_process() check
- **Result**: Segfault still occurs
- **Conclusion**: Segfault is NOT in is_application_process()

## Current Understanding

The segfault occurs:
- After the debug message "fopen() INTERCEPTOR CALLED" is printed
- NOT in the early PCI check
- NOT in is_application_process()
- NOT in the PCI file opening code

## Likely Location

The segfault is likely in:
1. `ensure_skip_flag_mutex_init()` - mutex initialization
2. Skip flag logic (reading/writing `g_skip_pci_interception`)
3. Normal interception code after skip flag check
4. `g_real_fopen_global()` call
5. Some other code path after the debug message

### Test 4: Disable skip flag logic
- **Result**: Segfault still occurs
- **Conclusion**: Segfault is NOT in skip flag logic

### Test 5: Simplify normal interception (return NULL immediately)
- **Result**: Segfault still occurs
- **Conclusion**: Segfault is NOT in normal interception code

### Test 6: Remove debug fprintf
- **Result**: Segfault still occurs
- **Conclusion**: Segfault is NOT in fprintf/fflush

### Test 7: Disable ensure_skip_flag_mutex_init() in fgets()
- **Result**: Segfault still occurs
- **Conclusion**: Segfault is NOT in ensure_skip_flag_mutex_init() called from fgets()

### Test 8: fgets() returns NULL immediately
- **Result**: Segfault still occurs
- **Conclusion**: Segfault is NOT in fgets() at all

## Current Understanding

The segfault occurs:
- After the debug message "fopen() INTERCEPTOR CALLED" is printed (when enabled)
- NOT in the early PCI check
- NOT in is_application_process()
- NOT in the PCI file opening code
- NOT in skip flag logic
- NOT in normal interception code
- NOT in fprintf/fflush
- NOT in fgets() at all
- NOT in ensure_skip_flag_mutex_init() called from fgets()

## Critical Findings

1. **fopen()**: Even when simplified to just return NULL immediately after the NULL check, the segfault still occurs.

2. **fgets()**: Even when simplified to just return NULL immediately, the segfault still occurs.

This strongly suggests:
1. The segfault is NOT in fopen() or fgets() interceptors
2. The segfault might be in library initialization/constructor code
3. The segfault might be in a different function (fread, fileno, etc.)
4. The segfault might be in a different thread/context
5. The segfault might be in nvmlInit_v2() itself or its dependencies

## Next Steps

1. Use gdb to get exact segfault location and stack trace
2. Check core dump for detailed information
3. Check if segfault is in fgets(), fread(), or other interceptors
4. Check if segfault is in library constructor/initialization

## Status

Device discovery is **complete and working**. The segfault is a separate issue that occurs after device discovery succeeds.
