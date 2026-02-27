# Include Directory

This directory contains header files used by the vGPU shim libraries.

## Files

- **cuda_protocol.h** - CUDA protocol definitions for vGPU communication
- **vgpu_protocol.h** - vGPU protocol definitions
- **cuda_executor.h** - CUDA executor definitions
- **scheduler_wfq.h** - Weighted fair queuing scheduler
- **rate_limiter.h** - Rate limiting definitions
- **watchdog.h** - Watchdog definitions
- **metrics.h** - Metrics collection
- **nvml_monitor.h** - NVML monitoring
- **vgpu_config.h** - vGPU configuration
- **cuda_vector_add.h** - CUDA vector operations

## Usage

These headers are included by source files in `../SOURCE/` during compilation.

The build script (`../BUILD/install.sh`) automatically finds this directory and passes it to the compiler with `-I` flag.

## Key Headers

### cuda_protocol.h
Defines the communication protocol between the shim and the vGPU-STUB device, including:
- MMIO register definitions
- Command structures
- Response formats

### vgpu_protocol.h
Defines the vGPU protocol for device management and communication.

## Dependencies

These headers are used by:
- `../SOURCE/libvgpu_cuda.c`
- `../SOURCE/libvgpu_nvml.c`
- `../SOURCE/libvgpu_cudart.c`
- `../SOURCE/cuda_transport.c`
