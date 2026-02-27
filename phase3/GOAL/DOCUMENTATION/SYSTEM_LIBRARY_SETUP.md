# System Library Setup: Technical Details

## Overview

The vGPU shim libraries are installed as system libraries, allowing them to work automatically with any application that uses CUDA or NVML APIs.

## Installation Details

### Library Locations

```
/usr/lib64/libvgpu-cuda.so      # Driver API shim
/usr/lib64/libvgpu-nvml.so      # NVML API shim
/usr/lib64/libvgpu-cudart.so    # Runtime API shim
```

### System Symlinks

```
/usr/lib64/libcuda.so.1         → /usr/lib64/libvgpu-cuda.so
/usr/lib64/libnvidia-ml.so.1    → /usr/lib64/libvgpu-nvml.so
```

### SONAME Configuration

The shim libraries are built with correct SONAMEs:

- `libvgpu-cuda.so` has SONAME `libcuda.so.1`
- `libvgpu-nvml.so` has SONAME `libnvidia-ml.so.1`

This ensures the dynamic linker resolves `libcuda.so.1` to our shim.

### ldconfig Registration

After installation, libraries are registered with `ldconfig`:

```bash
sudo ldconfig
```

This updates the system library cache so applications can find the libraries.

## How It Works

### Dynamic Library Resolution

When an application requests `libcuda.so.1`:

1. Dynamic linker checks `ldconfig` cache
2. Finds `/usr/lib64/libcuda.so.1` (symlink)
3. Follows symlink to `/usr/lib64/libvgpu-cuda.so`
4. Loads our shim library
5. Application uses vGPU shim transparently

### No LD_PRELOAD Needed

Because the shims are installed as system libraries with correct SONAMEs:
- Applications automatically load our shims
- No `LD_PRELOAD` required
- No special configuration needed
- Works for all applications using CUDA/NVML

## Verification

### Check Installation

```bash
# Libraries exist
ls -la /usr/lib64/libvgpu-*.so

# Symlinks correct
readlink -f /usr/lib64/libcuda.so.1
readlink -f /usr/lib64/libnvidia-ml.so.1

# SONAMEs correct
readelf -d /usr/lib64/libvgpu-cuda.so | grep SONAME
readelf -d /usr/lib64/libvgpu-nvml.so | grep SONAME

# Registered with ldconfig
ldconfig -p | grep -E "libcuda.so.1|libnvidia-ml.so.1"
```

### Test Library Loading

```bash
# Test without LD_PRELOAD
unset LD_PRELOAD
ldd /usr/lib64/libvgpu-cuda.so

# Test application loading
LD_DEBUG=libs ./test_program 2>&1 | grep libcuda
```

## Troubleshooting

### Libraries Not Found

If applications can't find libraries:

1. **Check symlinks exist:**
   ```bash
   ls -la /usr/lib64/libcuda.so.1
   ```

2. **Rebuild ldconfig cache:**
   ```bash
   sudo ldconfig
   ```

3. **Check library path:**
   ```bash
   echo $LD_LIBRARY_PATH
   ldconfig -p | grep libcuda
   ```

### Wrong Library Loaded

If wrong library is loaded:

1. **Check SONAME:**
   ```bash
   readelf -d /usr/lib64/libvgpu-cuda.so | grep SONAME
   ```

2. **Check symlink target:**
   ```bash
   readlink -f /usr/lib64/libcuda.so.1
   ```

3. **Check library search order:**
   ```bash
   LD_DEBUG=libs ./test_program 2>&1 | grep libcuda
   ```

## Best Practices

1. **Always use system library installation** - Don't use LD_PRELOAD for general applications
2. **Verify SONAMEs** - Ensure shims have correct SONAMEs
3. **Register with ldconfig** - Always run `ldconfig` after installation
4. **Test without LD_PRELOAD** - Verify system library resolution works

## Technical Notes

### Why System Libraries Work

- Dynamic linker resolves libraries by SONAME
- Applications request `libcuda.so.1` by SONAME
- Our shim provides `libcuda.so.1` SONAME
- Dynamic linker finds our shim automatically

### Why LD_PRELOAD Not Needed

- System library resolution happens first
- Our shims are in system library path
- Applications load our shims automatically
- No need for LD_PRELOAD override

### Compatibility

- Works with all applications using CUDA Driver API
- Works with all applications using CUDA Runtime API
- Works with all applications using NVML API
- Compatible with PyTorch, TensorFlow, etc.
