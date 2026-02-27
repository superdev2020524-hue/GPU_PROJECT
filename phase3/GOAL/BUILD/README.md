# Build Directory

This directory contains the build script for compiling and installing the vGPU shim libraries.

## Files

- **install.sh** - Main build and installation script

## Usage

```bash
cd phase3/GOAL/BUILD
sudo bash install.sh
```

This will:
1. Build all shim libraries from source files in `../SOURCE/`
2. Install them to `/usr/lib64/`
3. Create system symlinks
4. Register with `ldconfig`

## Prerequisites

- GCC compiler (`gcc`)
- Standard build tools
- Root/sudo access
- VGPU-STUB PCI device present in VM

## Source Files Location

The build script expects source files in:
- `../SOURCE/` - C source files (.c) and headers (.h)
- `../INCLUDE/` - Include directory with protocol headers

## Output

After successful build:
- `/usr/lib64/libvgpu-cuda.so` - Driver API shim
- `/usr/lib64/libvgpu-nvml.so` - NVML API shim
- `/usr/lib64/libvgpu-cudart.so` - Runtime API shim
- `/usr/lib64/libcuda.so.1` → symlink to libvgpu-cuda.so
- `/usr/lib64/libnvidia-ml.so.1` → symlink to libvgpu-nvml.so
