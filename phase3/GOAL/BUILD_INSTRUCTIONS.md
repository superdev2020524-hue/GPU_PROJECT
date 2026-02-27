# Complete Build Instructions

## Overview

The GOAL register is self-contained and includes all files needed to build and install the vGPU shim libraries from scratch.

## Directory Structure

```
phase3/GOAL/
├── SOURCE/              # All C source files and headers
│   ├── libvgpu_cuda.c
│   ├── libvgpu_nvml.c
│   ├── libvgpu_cudart.c
│   ├── cuda_transport.c
│   ├── gpu_properties.h
│   ├── cuda_transport.h
│   └── libcudart.so.12.versionscript
├── INCLUDE/             # Protocol headers
│   ├── cuda_protocol.h
│   ├── vgpu_protocol.h
│   └── ... (other headers)
├── BUILD/               # Build script
│   └── install.sh
└── TEST_SCRIPTS/        # Test programs
    └── ...
```

## Build Process

### Step 1: Verify Prerequisites

```bash
# Check GCC is installed
gcc --version

# Check VGPU-STUB device exists
lspci | grep -i nvidia

# Should show: 00:05.0 VGA compatible controller: NVIDIA Corporation Device 2331
```

### Step 2: Build and Install

```bash
cd phase3/GOAL/BUILD
sudo bash install.sh
```

The build script will:
1. Locate source files in `../SOURCE/`
2. Locate include files in `../INCLUDE/`
3. Compile all shim libraries
4. Install to `/usr/lib64/`
5. Create system symlinks
6. Register with `ldconfig`

### Step 3: Verify Installation

```bash
# Check libraries installed
ls -la /usr/lib64/libvgpu-*.so
ls -la /usr/lib64/libcuda.so.1 /usr/lib64/libnvidia-ml.so.1

# Run tests
cd phase3/GOAL/TEST_SCRIPTS
./test_cuda_detection.sh
python3 test_python_cuda.py
```

## Build Script Details

The `install.sh` script:
- Automatically finds source files in `../SOURCE/`
- Automatically finds include files in `../INCLUDE/`
- Compiles with correct flags and SONAMEs
- Installs as system libraries
- Creates necessary symlinks
- Registers with ldconfig

## Customization

If you need to modify the build:
1. Edit source files in `SOURCE/`
2. Edit headers in `INCLUDE/`
3. Rebuild with `sudo bash BUILD/install.sh`

## Troubleshooting

### Build Fails

```bash
# Check source files exist
ls -la SOURCE/*.c SOURCE/*.h

# Check include files exist
ls -la INCLUDE/*.h

# Check GCC
gcc --version
```

### Libraries Not Found After Install

```bash
# Rebuild ldconfig cache
sudo ldconfig

# Verify symlinks
readlink -f /usr/lib64/libcuda.so.1
```

## Complete Self-Contained Package

The GOAL register contains:
- ✅ All source files (.c, .h)
- ✅ All include files
- ✅ Build script
- ✅ Test scripts
- ✅ Documentation

**Everything needed to build and deploy vGPU shims on a new VM!**
