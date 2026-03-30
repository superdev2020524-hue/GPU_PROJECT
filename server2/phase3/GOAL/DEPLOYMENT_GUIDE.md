# GOAL Register Deployment Guide

## Complete Step-by-Step Instructions

### Prerequisites

- Linux VM with vGPU-STUB device visible (`lspci | grep -i nvidia`)
- GCC compiler installed (`gcc --version`)
- Root/sudo access
- Standard build tools

### Step 1: Transfer Archive

```bash
scp goal_register_COMPLETE.tar.gz user@vm:/tmp/
```

### Step 2: Extract Archive

```bash
ssh user@vm
cd /tmp
tar -xzf goal_register_COMPLETE.tar.gz
mv phase3/GOAL .
rm -rf phase3
```

### Step 3: Build and Install

```bash
cd /tmp/GOAL/BUILD
sudo bash install.sh
```

This will:
- Build all 3 shim libraries
- Install to `/usr/lib64/`
- Create system symlinks
- Register with `ldconfig`

### Step 4: Verify Installation

```bash
# Check libraries
ls -lh /usr/lib64/libvgpu-*.so
# Should show 3 libraries

# Check symlinks
ls -la /usr/lib64/libcuda.so.1 /usr/lib64/libnvidia-ml.so.1

# Run tests
cd /tmp/GOAL/TEST_SCRIPTS
./test_cuda_detection.sh
python3 test_python_cuda.py
```

### Expected Results

- ✅ 3 libraries in `/usr/lib64/`
- ✅ Symlinks point to shim libraries
- ✅ Test programs show GPU detected
- ✅ Applications can use CUDA/NVML automatically

## Troubleshooting

### Build Fails

1. Check GCC is installed: `gcc --version`
2. Check vGPU device exists: `lspci | grep -i nvidia`
3. Check source files: `ls -la /tmp/GOAL/SOURCE/*.c`
4. Review build log: Check `/tmp/goal_*.log` files

### Libraries Not Found

1. Rebuild ldconfig cache: `sudo ldconfig`
2. Check symlinks: `ls -la /usr/lib64/libcuda.so.1`
3. Check library path: `echo $LD_LIBRARY_PATH`

### GPU Not Detected

1. Verify vGPU device: `lspci | grep -i nvidia`
2. Check device files: `ls -la /sys/bus/pci/devices/0000:00:05.0/`
3. Run test programs: `cd /tmp/GOAL/TEST_SCRIPTS && ./test_cuda_detection.sh`

## Files Included

- ✅ All source files (.c, .h)
- ✅ All include files
- ✅ Build script (install.sh)
- ✅ Test scripts
- ✅ Complete documentation

## Status

✅ **Ready for Deployment** - Tested and verified on new VM

