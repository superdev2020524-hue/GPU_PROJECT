# PCI Auto-Detection Troubleshooting Guide

## Quick Fix

If auto-detection fails, the most common causes are:

1. **Not running as root**: Use `sudo ./vm_client_vector <num1> <num2>`
2. **Device not attached**: Verify with `lspci | grep 'Processing accelerators'`
3. **Permission issues**: Check `/sys/bus/pci/devices/` is readable

## How Auto-Detection Works

The code scans `/sys/bus/pci/devices` and looks for devices matching:
- **Vendor ID**: `0x1af4` (Red Hat, Inc.)
- **Device ID**: `0x1111` (vGPU stub)
- **Class**: `0x120000` (Processing Accelerator) - *optional check*

**Note**: Vendor + Device match is sufficient. Class check is optional and won't cause rejection.

## Verification Steps

### Step 1: Check if device is visible
```bash
lspci | grep 'Processing accelerators'
# Should show: XX:XX.X Processing accelerators: Red Hat, Inc. Device 1111 (rev 01)
```

### Step 2: Check sysfs access
```bash
# Find the device path (replace XX:XX.X with actual address from lspci)
ls -la /sys/bus/pci/devices/0000:XX:XX.X/

# Check if resource0 exists
ls -la /sys/bus/pci/devices/0000:XX:XX.X/resource0

# Check vendor/device files
cat /sys/bus/pci/devices/0000:XX:XX.X/vendor
# Should show: 0x1af4

cat /sys/bus/pci/devices/0000:XX:XX.X/device
# Should show: 0x1111
```

### Step 3: Test with sudo
```bash
sudo ./vm_client_vector 100 200
```

## Common Issues

### Issue 1: "vGPU stub device not found"

**Symptoms:**
```
[SCAN] Searching for vGPU stub device...
[SCAN] vGPU stub device not found
```

**Possible Causes:**
1. Device not attached to VM
2. Running without sudo (can't read sysfs files)
3. Device at unexpected PCI address

**Solutions:**
1. Verify device exists: `lspci | grep 'Processing accelerators'`
2. Run with sudo: `sudo ./vm_client_vector <num1> <num2>`
3. Check sysfs: `ls /sys/bus/pci/devices/ | grep 0000`

### Issue 2: "Found matching device(s) but resource0 not accessible"

**Symptoms:**
```
[SCAN] Found 1 matching device(s) but resource0 not accessible
```

**Cause:** Running without root privileges

**Solution:** Use `sudo ./vm_client_vector <num1> <num2>`

### Issue 3: Device found but "Failed to open vGPU device"

**Symptoms:**
```
[SCAN] Found vGPU stub at 0000:00:05.0
Failed to open vGPU device: Permission denied
```

**Cause:** Need root to access MMIO resource

**Solution:** Use `sudo ./vm_client_vector <num1> <num2>`

## Manual Verification

If auto-detection consistently fails, you can manually verify the device:

```bash
# 1. Find device address
lspci -d 1af4:1111
# Output: 00:05.0 Processing accelerators: Red Hat, Inc. Device 1111 (rev 01)

# 2. Check sysfs path
ls /sys/bus/pci/devices/0000:00:05.0/

# 3. Verify vendor
cat /sys/bus/pci/devices/0000:00:05.0/vendor
# Should be: 0x1af4

# 4. Verify device
cat /sys/bus/pci/devices/0000:00:05.0/device
# Should be: 0x1111

# 5. Check resource0 exists
ls -l /sys/bus/pci/devices/0000:00:05.0/resource0
# Should show: -r--r--r-- ... resource0
```

## Debug Mode

The current implementation includes debug output:
- `[SCAN] Searching for vGPU stub device...` - Scan started
- `[SCAN] Found vGPU stub at XXXX` - Device found
- `[SCAN] Device XXXX: class=0xXXXXXX` - Class mismatch (non-fatal)
- `[SCAN] vGPU stub device not found` - No device found

## Expected Behavior

**Success:**
```
[SCAN] Searching for vGPU stub device...
[SCAN] Found vGPU stub at 0000:00:05.0
[MMIO] Read vGPU properties:
  Pool ID: A
  Priority: 2 (high)
  VM ID: 3
```

**Failure (no device):**
```
[SCAN] Searching for vGPU stub device...
[SCAN] vGPU stub device not found
       Expected: Vendor=0x1af4, Device=0x1111, Class=0x120000
```

**Failure (permission):**
```
[SCAN] Searching for vGPU stub device...
[SCAN] Found 1 matching device(s) but resource0 not accessible
       Make sure you're running as root (sudo)
```

## Why This Should Work on All VMs

The auto-detection:
1. **Scans all PCI devices** - Works regardless of PCI slot assignment
2. **Matches by vendor+device** - Unique combination, class is optional
3. **No hardcoded paths** - Adapts to any PCI address
4. **Clear error messages** - Helps diagnose issues

## If It Still Doesn't Work

1. **Check VM configuration**: Ensure vgpu-stub device is attached
2. **Check host logs**: Verify device is properly configured on host
3. **Try different VM**: Test on Test-1 or Test-2 to compare
4. **Check kernel messages**: `dmesg | grep -i pci | grep -i vgpu`
