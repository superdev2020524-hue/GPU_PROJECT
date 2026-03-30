# SHIM Extension Summary: Making Ollama Discovery Work

## Critical Discovery from VM Investigation

**Ollama scans PCI devices directly** by reading:
- `/sys/bus/pci/devices/*/vendor`
- `/sys/bus/pci/devices/*/device`
- `/sys/bus/pci/devices/*/class`

Our vGPU device exists at `0000:00:05.0` with:
- Vendor: `0x10de` (NVIDIA) ✓
- Device: `0x2331` (H100 PCIe) ✓
- Class: `0x030200` (3D controller) ✓

**But Ollama reports `pci_id=""` and `library=cpu`**, indicating discovery isn't finding the device.

## Research Findings

From web search and codebase analysis:
1. **Ollama stops searching after first library load failure** - doesn't try alternative paths
2. **Ollama uses `dlsym()` to load NVML functions** - if symbols fail, discovery fails
3. **Ollama matches PCI devices with NVML devices by PCI bus ID** - format must match exactly
4. **Discovery may happen in subprocess** - `LD_PRELOAD` might not be inherited
5. **Ollama may use `fread()`/`fgets()` instead of `read()`** for PCI device files

## SHIM Extensions Implemented

### Phase 1: Enhanced Filesystem Interception ✓
- Intercepts `/proc/driver/nvidia/version`, `/proc/driver/nvidia/params`
- Intercepts `/sys/class/drm/card*` and `/dev/nvidia*` paths
- Added glibc internal function interception (`__xstat`, `__xstat64`, `__lxstat`, `__lxstat64`, `__fxstatat`, `__fxstatat64`)
- All filesystem interception tests pass

### Phase 2: Complete NVML Function Export ✓
- Added logging to `nvmlDeviceGetCudaComputeCapability`
- Added logging to `nvmlSystemGetDriverVersion`
- All required NVML functions are exported and functional

### Phase 3: Discovery Trigger Mechanism ✓
- Added discovery trigger in constructor that calls all discovery functions
- All discovery functions return correct values:
  - Device count: 1
  - Memory: 81004 MB
  - PCI bus ID: 00000000:00:05.0
  - Compute capability: 9.0
  - Driver version: 12.3.00

### Phase 4: Library Path Optimization ✓
- Fixed symlinks in `/usr/lib/x86_64-linux-gnu/` to point to our shims
- Updated `ldconfig`

### Phase 5: PCI Device File Read Interception ✓
- Added `read()` interception for PCI device files
- Added `pread()` interception for PCI device files
- Returns correct values: `0x10de`, `0x2331`, `0x030200`

## Current Status

✓ All discovery functions work correctly
✓ All filesystem interception works correctly
✓ All library paths are correct
✓ PCI device file read interception works (tested)
✗ Ollama still reports `library=cpu` with `pci_id=""`

## Root Cause Analysis

The fact that Ollama reports `pci_id=""` suggests:
1. **Ollama uses `fread()`/`fgets()` instead of `read()`** - Our interception doesn't catch these
2. **Discovery happens in subprocess** - `LD_PRELOAD` not inherited
3. **Ollama has validation check that fails** - Before reaching NVML
4. **Ollama uses cached discovery result** - From previous run

## Next Steps to Extend SHIM

### Priority 1: Intercept FILE* Operations
- Intercept `fread()`, `fgets()`, `fscanf()` for PCI device files
- Intercept `fopen()` to track which files are opened
- Ensure PCI device file reads go through our interception

### Priority 2: Subprocess Detection
- Check if discovery happens in subprocess
- If so, inject shim via subprocess environment
- Ensure `LD_PRELOAD` is inherited

### Priority 3: Higher-Level Interception
- Consider intercepting at Go CGO layer
- Intercept Go's C library calls
- May require different approach than C-level interception

### Priority 4: Comprehensive Tracing
- Add `strace`-like logging for all system calls
- Trace exact discovery path Ollama takes
- Identify where discovery fails

## Implementation Files

- `phase3/guest-shim/libvgpu_cuda.c` - CUDA shim with filesystem and PCI interception
- `phase3/guest-shim/libvgpu_nvml.c` - NVML shim with discovery trigger
- `phase3/guest-shim/SHIM_EXTENSION_PLAN.md` - Detailed implementation plan

## Testing

All components work correctly when tested directly:
- Filesystem interception: ✓
- PCI device file reads: ✓
- NVML functions: ✓
- Discovery trigger: ✓

The issue is that Ollama's discovery mechanism doesn't call our intercepted functions, suggesting it uses a different code path or happens in a different process.
