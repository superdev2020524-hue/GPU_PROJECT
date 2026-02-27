# Complete File List: GOAL Register

## Self-Contained Package for vGPU Detection

This register contains **ALL** files needed to build and deploy vGPU shims on a new VM.

---

## Source Files (SOURCE/)

### C Source Files
- `libvgpu_cuda.c` - CUDA Driver API shim (215KB)
- `libvgpu_nvml.c` - NVML API shim (46KB)
- `libvgpu_cudart.c` - CUDA Runtime API shim (34KB)
- `cuda_transport.c` - Transport layer (42KB)

### Header Files
- `gpu_properties.h` - GPU properties and defaults
- `cuda_transport.h` - Transport layer headers

### Build Files
- `libcudart.so.12.versionscript` - Version script for symbol exports

---

## Include Files (INCLUDE/)

### Protocol Headers
- `cuda_protocol.h` - CUDA protocol definitions (14KB)
- `vgpu_protocol.h` - vGPU protocol definitions (17KB)
- `cuda_executor.h` - CUDA executor definitions
- `scheduler_wfq.h` - Weighted fair queuing scheduler
- `rate_limiter.h` - Rate limiting definitions
- `watchdog.h` - Watchdog definitions
- `metrics.h` - Metrics collection
- `nvml_monitor.h` - NVML monitoring
- `vgpu_config.h` - vGPU configuration
- `cuda_vector_add.h` - CUDA vector operations

---

## Build Scripts (BUILD/)

### Main Build Script
- `install.sh` - Complete build and installation script (58KB)
  - Compiles all shim libraries
  - Installs to `/usr/lib64/`
  - Creates system symlinks
  - Registers with `ldconfig`

---

## Test Scripts (TEST_SCRIPTS/)

### C/C++ Tests
- `test_cuda_detection.c` - C test program source
- `test_cuda_detection.sh` - C test script (compiles and runs)
- `test_vgpu_system.c` - System library test

### Python Tests
- `test_python_cuda.py` - Python CUDA test

---

## Documentation

### Main Documentation
- `README.md` - Overview and quick start
- `SUMMARY.md` - Executive summary
- `INDEX.md` - Navigation index
- `INSTALLATION.md` - Installation guide
- `VERIFICATION.md` - Verification guide
- `BUILD_INSTRUCTIONS.md` - Complete build instructions
- `COMPLETE_FILE_LIST.md` - This file

### Technical Documentation (DOCUMENTATION/)
- `SYSTEM_LIBRARY_SETUP.md` - Technical implementation details
- `TESTING_RESULTS.md` - Test results and compatibility

### Directory READMEs
- `SOURCE/README.md` - Source files documentation
- `INCLUDE/README.md` - Include files documentation
- `BUILD/README.md` - Build script documentation

---

## File Count Summary

- **Source files (.c)**: 4 files
- **Header files (.h)**: 2 files in SOURCE/, 10 files in INCLUDE/
- **Build scripts (.sh)**: 1 main script
- **Version scripts**: 1 file
- **Test scripts**: 4 files
- **Documentation**: 10+ files

**Total: Complete self-contained package**

---

## Usage on New VM

1. **Copy entire GOAL directory** to new VM
2. **Build shims:**
   ```bash
   cd GOAL/BUILD
   sudo bash install.sh
   ```
3. **Verify:**
   ```bash
   cd GOAL/TEST_SCRIPTS
   ./test_cuda_detection.sh
   python3 test_python_cuda.py
   ```
4. **Use in applications** - No special config needed!

---

## What Gets Built

After running `install.sh`:

- `/usr/lib64/libvgpu-cuda.so` - Driver API shim
- `/usr/lib64/libvgpu-nvml.so` - NVML API shim
- `/usr/lib64/libvgpu-cudart.so` - Runtime API shim
- `/usr/lib64/libcuda.so.1` → symlink to libvgpu-cuda.so
- `/usr/lib64/libnvidia-ml.so.1` → symlink to libvgpu-nvml.so

---

## Dependencies

### Build-Time
- GCC compiler
- Standard build tools
- Root/sudo access

### Runtime
- VGPU-STUB PCI device
- Linux kernel with PCI support

### No External Libraries Required
- All code is self-contained
- No external CUDA/NVML libraries needed
- Works as drop-in replacement

---

## Status

✅ **Complete** - All source files included
✅ **Self-Contained** - No external dependencies
✅ **Ready to Deploy** - Can be used on any new VM
✅ **Tested** - All test scripts verified working

---

**Last Updated:** 2026-02-27
