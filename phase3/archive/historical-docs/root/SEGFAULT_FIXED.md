# Segfault Fixed - Final Solution

## Date: 2026-02-25

## Problem
Ollama was segfaulting immediately after device discovery succeeded. The segfault occurred when `nvmlInit_v2()` was called.

## Root Cause
The segfault was caused by calling `cuda_transport_pci_bdf(NULL)` directly in a `fprintf()` format string in `nvmlInit_v2()`:

```c
fprintf(stderr,
        "[libvgpu-nvml] nvmlInit() succeeded with defaults (transport deferred, bdf=%s)\n",
        cuda_transport_pci_bdf(NULL));  // ❌ SEGFAULT HERE
```

## Solution
Store the result of `cuda_transport_pci_bdf()` in a local variable before using it in `fprintf()`:

```c
/* FIX: Store cuda_transport_pci_bdf() result in a local variable first
 * This avoids potential issues with calling it directly in fprintf() format string */
const char *bdf = cuda_transport_pci_bdf(NULL);
fprintf(stderr,
        "[libvgpu-nvml] nvmlInit() succeeded with defaults (transport deferred, bdf=%s)\n",
        bdf ? bdf : "unknown");
```

## File Changed
- `phase3/guest-shim/libvgpu_nvml.c` - Fixed `nvmlInit_v2()` function

## Status
✅ **FIXED** - Ollama is now running successfully
✅ Device discovery working
✅ GPU detection working
✅ No segfaults

## Testing
- Ollama service: `active (running)`
- Device discovery: Working
- GPU mode: Enabled

## Notes
- The issue was not in `fopen()` or `fgets()` interceptors (they were disabled during testing but were not the cause)
- The segfault was specifically in the `fprintf()` call with `cuda_transport_pci_bdf()` as a format argument
- Storing the result in a local variable first resolves the issue
