# PCI Auto-Detection Fix

## Problem

The VM client had a hardcoded PCI address (`0000:00:06.0`) for the vGPU device. This caused failures in VMs where the device appears at a different address:

- **Test-2**: Device at `0000:00:06.0` → Works ✓
- **Test-1**: Device at `0000:00:08.0` → Fails ✗

## Root Cause

Different VMs may have the vGPU device assigned to different PCI slots depending on:
- VM boot order
- Other PCI devices present
- Xen/QEMU device assignment order

## Solution

Implemented automatic PCI device detection by scanning `/sys/bus/pci/devices` for the vGPU stub device based on:
- **Vendor ID**: `0x1af4` (Red Hat, Inc.)
- **Device ID**: `0x1111` (Custom vGPU stub)
- **Class Code**: `0x120000` (Processing Accelerator, General Purpose)

## Changes Made

### 1. Added Auto-Detection Function

**New Function:** `find_vgpu_device()`
- Scans `/sys/bus/pci/devices` directory
- Checks each device's vendor, device, and class IDs
- Returns path to `resource0` file when found
- Returns `NULL` if device not found

### 2. Updated Includes

Added:
```c
#include <dirent.h>  // For directory scanning
```

### 3. Replaced Hardcoded PCI Address

**Before:**
```c
#define PCI_RESOURCE "/sys/bus/pci/devices/0000:00:06.0/resource0"
...
fd = open(PCI_RESOURCE, O_RDONLY);
```

**After:**
```c
// Auto-detect vGPU device
pci_resource = find_vgpu_device();
if (!pci_resource) {
    // Error handling
    return -1;
}
fd = open(pci_resource, O_RDONLY);
```

### 4. Added Device Identification Constants

```c
#define VGPU_VENDOR_ID 0x1af4  // Red Hat, Inc.
#define VGPU_DEVICE_ID 0x1111  // Custom vGPU stub
#define VGPU_CLASS_MASK 0xffff00
#define VGPU_CLASS 0x120000    // Processing Accelerator
```

## Implementation Details

The `find_vgpu_device()` function:
1. Opens `/sys/bus/pci/devices` directory
2. Iterates through all PCI devices
3. For each device:
   - Reads `vendor` file → checks for `0x1af4`
   - Reads `device` file → checks for `0x1111`
   - Reads `class` file → checks for `0x120000` (masked)
4. When match found, constructs path to `resource0`
5. Verifies `resource0` exists and is readable
6. Returns full path to `resource0`

## Benefits

1. **Works across all VMs** regardless of PCI slot assignment
2. **No configuration needed** - automatically finds the device
3. **Better error messages** - clearly indicates if device not found
4. **Future-proof** - adapts to different VM configurations

## Testing

### Test-1 (Device at 00:08.0)
```bash
lspci | grep 'Processing accelerators'
# Output: 00:08.0 Processing accelerators: Red Hat, Inc. Device 1111 (rev 01)

./vm_client_vector 456 789
# Should now work: [SCAN] Found vGPU stub at 0000:00:08.0
```

### Test-2 (Device at 00:06.0)
```bash
lspci | grep 'Processing accelerators'
# Output: 00:06.0 Processing accelerators: Red Hat, Inc. Device 1111 (rev 01)

./vm_client_vector 100 200
# Should still work: [SCAN] Found vGPU stub at 0000:00:06.0
```

## Error Handling

If device not found, the program will:
1. Print scan failure message
2. Show expected device identifiers
3. Suggest troubleshooting steps:
   - Check if running as root
   - Verify vgpu-stub device is attached
   - Check with `lspci | grep 'Processing accelerators'`

## Files Modified

- `/home/david/Downloads/gpu/step2_test/vm_client_vector.c`
  - Added `find_vgpu_device()` function
  - Updated `read_vgpu_properties()` to use auto-detection
  - Added `#include <dirent.h>`
  - Removed hardcoded `PCI_RESOURCE` define
  - Added device identification constants

## Next Steps

1. Rebuild VM client:
   ```bash
   cd /home/david/Downloads/gpu/step2_test
   make vm
   ```

2. Test on Test-1:
   ```bash
   cd build-vm
   sudo ./vm_client_vector 456 789
   ```

3. Verify output shows:
   ```
   [SCAN] Searching for vGPU stub device...
   [SCAN] Found vGPU stub at 0000:00:08.0
   [MMIO] Read vGPU properties:
     Pool ID: A
     Priority: 2 (high)
     VM ID: 1
   ```
