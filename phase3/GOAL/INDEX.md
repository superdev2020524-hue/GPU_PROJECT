# GOAL Register Index

## Quick Navigation

### Getting Started
- **[README.md](README.md)** - Overview and quick start
- **[QUICK_START.md](QUICK_START.md)** - Quick deployment guide
- **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Complete deployment instructions

### Installation
- **[INSTALLATION.md](INSTALLATION.md)** - Detailed installation guide
- **[BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md)** - Complete build instructions
- **[BUILD/README.md](BUILD/README.md)** - Build script documentation

### Verification
- **[VERIFICATION.md](VERIFICATION.md)** - How to verify vGPU detection
- **[TEST_SCRIPTS/](TEST_SCRIPTS/)** - Test programs

### Documentation
- **[SUMMARY.md](SUMMARY.md)** - Executive summary
- **[COMPLETE_FILE_LIST.md](COMPLETE_FILE_LIST.md)** - Complete file list
- **[FINAL_STATUS.md](FINAL_STATUS.md)** - Final test status
- **[SUCCESS.md](SUCCESS.md)** - Success confirmation

### Technical Details
- **[SOURCE/README.md](SOURCE/README.md)** - Source files documentation
- **[INCLUDE/README.md](INCLUDE/README.md)** - Include files documentation
- **[BUILD_ERRORS_FOUND.md](BUILD_ERRORS_FOUND.md)** - Build error analysis
- **[NEW_VM_TEST_RESULTS.md](NEW_VM_TEST_RESULTS.md)** - New VM test results

### Status
- **[TESTING_STATUS.md](TESTING_STATUS.md)** - Current testing status
- **[DEPLOYMENT_READY.md](DEPLOYMENT_READY.md)** - Deployment readiness
- **[FINAL_STATUS.md](FINAL_STATUS.md)** - Final status

## Directory Structure

```
GOAL/
├── SOURCE/              # All C source files and headers
├── INCLUDE/             # Protocol headers
├── BUILD/               # Build script
├── TEST_SCRIPTS/        # Test programs
└── DOCUMENTATION/       # Additional documentation
```

## Key Files

### Source Files
- `SOURCE/libvgpu_cuda.c` - CUDA Driver API shim
- `SOURCE/libvgpu_nvml.c` - NVML API shim
- `SOURCE/libvgpu_cudart.c` - CUDA Runtime API shim
- `SOURCE/cuda_transport.c` - Transport layer

### Build Script
- `BUILD/install.sh` - Main build and installation script

### Test Scripts
- `TEST_SCRIPTS/test_cuda_detection.sh` - C test script
- `TEST_SCRIPTS/test_python_cuda.py` - Python test

## Status

✅ **Complete** - All files included
✅ **Tested** - Verified on new VM
✅ **Ready** - Ready for deployment

---

**Last Updated**: 2026-02-27
