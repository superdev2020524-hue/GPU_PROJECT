# Source Directory

This directory contains all source files needed to build the vGPU shim libraries.

## Files

### C Source Files (.c)
- **libvgpu_cuda.c** - CUDA Driver API shim implementation
- **libvgpu_nvml.c** - NVML API shim implementation
- **libvgpu_cudart.c** - CUDA Runtime API shim implementation
- **cuda_transport.c** - CUDA transport layer for vGPU communication

### Header Files (.h)
- **gpu_properties.h** - GPU properties and defaults
- **cuda_transport.h** - CUDA transport layer headers

### Build Files
- **libcudart.so.12.versionscript** - Version script for libcudart.so.12 symbol exports

## Dependencies

These source files depend on headers in:
- `../INCLUDE/` - Protocol headers (cuda_protocol.h, etc.)

## Building

Source files are compiled by the build script in `../BUILD/install.sh`.

## Important Notes

### Source File Versions

**CRITICAL**: The source files in this directory must match the working versions in `phase3/guest-shim/`.

If you encounter compilation errors:
1. Verify files match working versions:
   ```bash
   diff phase3/guest-shim/libvgpu_cuda.c phase3/GOAL/SOURCE/libvgpu_cuda.c
   diff phase3/guest-shim/cuda_transport.c phase3/GOAL/SOURCE/cuda_transport.c
   ```

2. If files differ, copy corrected versions:
   ```bash
   cp phase3/guest-shim/libvgpu_cuda.c phase3/GOAL/SOURCE/libvgpu_cuda.c
   cp phase3/guest-shim/cuda_transport.c phase3/GOAL/SOURCE/cuda_transport.c
   ```

### Known Issues

See `../BUILD_ERRORS_FOUND.md` for details on compilation errors that were found during testing.

## Key Components

### libvgpu_cuda.c
- Implements CUDA Driver API functions
- Handles device discovery and initialization
- Provides PCI bus ID matching
- Intercepts dlopen/dlsym for symbol resolution
- **Size**: ~211KB, ~5275 lines

### libvgpu_nvml.c
- Implements NVML API functions
- Provides device information and monitoring
- Handles PCI bus ID matching for device pairing
- **Size**: ~46KB, ~1241 lines

### libvgpu_cudart.c
- Implements CUDA Runtime API functions
- Handles runtime initialization
- Provides version compatibility
- **Size**: ~34KB, ~897 lines

### cuda_transport.c
- Communicates with vGPU-STUB device
- Handles MMIO operations
- Manages device discovery and connection
- **Size**: ~42KB
