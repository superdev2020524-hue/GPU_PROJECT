# Makefile Updates Applied

## Summary

Updated the Makefile to match current host settings and include all required shim libraries.

## Changes Made

### 1. Added Missing Shim Libraries

**Added build targets for:**
- `libvgpu-cudart.so` — CUDA Runtime API shim (was missing)
- `libvgpu-cublas.so.12` — CUBLAS shim (was missing)
- `libvgpu-cublasLt.so.12` — CUBLAS LT shim (was missing)

**Updated:**
- `libvgpu-cuda.so` → `libvgpu-cuda.so.1` (correct version suffix)

### 2. Updated Guest Build Target

**Before:**
```makefile
guest: $(SHIM_CUDA_LIB) $(SHIM_NVML_LIB)
```

**After:**
```makefile
guest: $(SHIM_CUDA_LIB) $(SHIM_CUDART_LIB) $(SHIM_NVML_LIB) $(SHIM_CUBLAS_LIB) $(SHIM_CUBLASLT_LIB)
```

### 3. Added Build Rules for New Libraries

**libvgpu-cudart.so:**
- Depends on: `libvgpu_cudart.c`, headers
- Links with: `-ldl` (for dynamic loading)
- No transport dependency (uses transport via dlopen)

**libvgpu-cublas.so.12:**
- Depends on: `libvgpu_cublas.c`, headers
- Uses version script: `cublas_version.lds` (for symbol versioning)
- Links with: `-ldl`

**libvgpu-cublasLt.so.12:**
- Depends on: `libvgpu_cublasLt.c`, headers
- Links with: `-ldl`

### 4. Updated Help Text

Added documentation for all shim libraries and CUDA path override.

## Verification

### Current Host Settings Match

| Setting | Makefile | Host Reality | Status |
|---------|----------|--------------|--------|
| CUDA_PATH | `/usr/local/cuda` (default) | Unknown | ⚠️ Can override |
| QEMU path | (not in Makefile) | `/usr/lib64/xen/bin/qemu-system-i386` | ✅ Correct |
| Mediator binary | `mediator_phase3` | `mediator_phase3` | ✅ Matches |
| Build location | Current directory | `/root/phase3/` | ✅ Matches |
| Socket path | (in source code) | `/var/xen/qemu/root-XXX/tmp/vgpu-mediator.sock` | ✅ Correct |

### Build Commands

**Host-side (mediator):**
```bash
cd /root/phase3
make host
# Creates: mediator_phase3, vgpu-admin
```

**Guest-side (shim libraries):**
```bash
cd /root/phase3
make guest
# Creates: libvgpu-cuda.so.1, libvgpu-cudart.so, libvgpu-nvml.so,
#          libvgpu-cublas.so.12, libvgpu-cublasLt.so.12
```

## Remaining Considerations

### 1. CUDA Path Detection

The Makefile assumes CUDA is at `/usr/local/cuda`. If it's elsewhere:

```bash
CUDA_PATH=/opt/cuda make host
```

**Recommendation:** Add auto-detection in future (see MAKEFILE_HOST_SETTINGS_CHECK.md)

### 2. Version Script for CUBLAS

The Makefile references `cublas_version.lds` which should exist in `guest-shim/` directory.

**If missing, create it:**
```lds
CUBLAS_12.0 {
  global:
    cublas*;
  local:
    *;
};
```

### 3. RPM Build Directory

Default is `$(HOME)/vgpu-build/rpmbuild`. Verify this matches your setup:

```bash
# Check if directory exists
ls -ld ~/vgpu-build/rpmbuild

# If not, create it:
mkdir -p ~/vgpu-build/rpmbuild/{SOURCES,SPECS,BUILD,RPMS,SRPMS}
```

## Testing the Updated Makefile

### Test Host Build:
```bash
cd /root/phase3
make clean
make host
# Should build: mediator_phase3, vgpu-admin
```

### Test Guest Build:
```bash
cd /root/phase3
make guest
# Should build all 5 shim libraries
ls -lh guest-shim/libvgpu-*.so*
```

### Verify CUDA Path (if build fails):
```bash
# Check if CUDA exists at default location
ls -l /usr/local/cuda/include/cuda.h

# If not, find CUDA:
find /usr -name "cuda.h" 2>/dev/null | head -1

# Then override:
CUDA_PATH=/found/path make host
```

## Files Modified

- `phase3/Makefile`: Added missing shim library build targets and updated guest target

## Next Steps

1. ✅ **Test the updated Makefile** on the host
2. ⚠️ **Verify CUDA path** matches or override if needed
3. ⚠️ **Check for cublas_version.lds** in guest-shim directory
4. ✅ **Update documentation** to reflect all shim libraries
