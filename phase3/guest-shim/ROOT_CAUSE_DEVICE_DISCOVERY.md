# Root Cause: Device Discovery Failing Despite Device Existing

## Investigation Results

### ✅ What's Confirmed Working:

1. **Device EXISTS**: 
   - PCI device `0000:00:05.0` exists in `/sys/bus/pci/devices/`
   - Verified with `ls -la /sys/bus/pci/devices/0000:00:05.0/`

2. **Device MATCHES perfectly**:
   - Vendor: `0x10de` (matches `VGPU_VENDOR_ID 0x10DE`)
   - Device: `0x2331` (matches `VGPU_DEVICE_ID 0x2331`)
   - Class: `0x030200` (matches `VGPU_CLASS 0x030200`)

3. **Matching logic WORKS**:
   - Test program shows: `exact match: 1` (true)
   - All comparisons work correctly
   - No case sensitivity issues (C handles hex correctly)

4. **Code is correct**:
   - Debug code exists in source (line 201-204)
   - Matching logic is correct (lines 206-223)
   - Library was rebuilt (newer than source)

### ❌ What's NOT Working:

1. **Discovery code says device NOT found**:
   - Error: `VGPU-STUB not found in /sys/bus/pci/devices`
   - Scanned 8 devices but didn't find it

2. **Debug message NOT appearing**:
   - Debug code exists but message never appears in logs
   - This suggests either:
     - Device isn't being scanned
     - Code path is different
     - File access fails silently

## Possible Root Causes

### 1. File Access Issue
- Discovery code might not be able to read `/sys/bus/pci/devices/0000:00:05.0/vendor`
- Systemd sandbox might block access
- Permissions might be wrong

### 2. Code Not Being Called
- `cuda_transport_discover()` might not be called at all
- Or called in a different context where `/sys` isn't accessible

### 3. Early Return
- Code might return before scanning all devices
- Or might skip device 0000:00:05.0 for some reason

### 4. Different Code Path
- The running code might be different from source
- Or a different version of the function is being called

## Next Steps

1. **Add more debug logging** to see:
   - If `cuda_transport_discover()` is called
   - How many devices are scanned
   - What values are read for each device
   - Why device 0000:00:05.0 doesn't match

2. **Test file access** from the actual process:
   - Check if Ollama process can read `/sys/bus/pci/devices/0000:00:05.0/vendor`
   - Check systemd sandbox restrictions

3. **Compare with working state**:
   - What was different when it worked?
   - Was the code different?
   - Was the environment different?

## Critical Question

**Why does the matching logic work in a test program, but fail in the actual discovery code?**

This suggests the issue is NOT with the matching logic itself, but with:
- How the values are read from `/sys`
- When/where the discovery code is called
- What environment it runs in
