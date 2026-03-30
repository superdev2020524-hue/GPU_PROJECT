# OLLAMA_NUM_GPU=999 Fix Applied

## Date: 2026-02-26

## Critical Fix Found

According to `command.txt` (lines 84-94), the complete solution requires:

**FIX: cuDeviceGetPCIBusId + real NVML PCI bus ID + OLLAMA_NUM_GPU=999**

### Root Cause

Ollama's GPU discovery matched NVML devices with CUDA devices by PCI bus ID, but:
- Our NVML shim returned a fake "00000000:00:00.0" address
- cuDeviceGetPCIBusId was missing from the CUDA shim entirely
- Result: library=cpu

### Solution Components

1. **cuDeviceGetPCIBusId()** - Implemented and exported ✓
2. **nvmlDeviceGetPciInfo_v3()** - Returns real BDF (0000:00:05.0) ✓
3. **OLLAMA_NUM_GPU=999** - Set in vgpu.conf ✓

### Files Changed

- `guest-shim/cuda_transport.c` - pci_bdf field + find_vgpu_device() BDF output
- `guest-shim/cuda_transport.h` - cuda_transport_pci_bdf() declaration
- `guest-shim/libvgpu_cuda.c` - cuDeviceGetPCIBusId / cuDeviceGetPCIBusId_v2
- `guest-shim/libvgpu_nvml.c` - nvmlDeviceGetPciInfo_v3 uses real BDF
- `/etc/systemd/system/ollama.service.d/vgpu.conf` - OLLAMA_NUM_GPU=999

### Status

✅ **OLLAMA_NUM_GPU=999** - Set with correct syntax in vgpu.conf
✅ **cuDeviceGetPCIBusId()** - Implemented and exported
✅ **nvmlDeviceGetPciInfo_v3()** - Returns correct PCI bus ID
✅ **All symlinks** - Correct
✅ **All dependencies** - Resolved

### Expected Result

With OLLAMA_NUM_GPU=999 set:
- Ollama will call cuDeviceGetPCIBusId() for PCI bus ID matching
- PCI bus IDs from CUDA and NVML will match (0000:00:05.0)
- Matching will succeed
- Result: library=cuda with pci_id="0000:00:05.0"

### Next Steps

1. Verify GPU mode is active (library=cuda)
2. Verify PCI bus ID is set (pci_id="0000:00:05.0")
3. Verify cuDeviceGetPCIBusId() is being called
