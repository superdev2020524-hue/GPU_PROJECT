# Installation Guide: vGPU Shim for General GPU Applications

## Prerequisites

- Linux VM with vGPU-STUB device
- Root/sudo access
- GCC compiler
- Standard build tools

## Step-by-Step Installation

### Step 1: Build and Install Shim Libraries

```bash
cd phase3/GOAL/BUILD
sudo bash install.sh
```

This will:
1. Build all shim libraries (`libvgpu-cuda.so`, `libvgpu-nvml.so`, `libvgpu-cudart.so`)
2. Install them to `/usr/lib64/`
3. Create system symlinks (`libcuda.so.1`, `libnvidia-ml.so.1`)
4. Register with `ldconfig`

### Step 2: Verify Installation

```bash
# Check libraries are installed
ls -la /usr/lib64/libvgpu-*.so
ls -la /usr/lib64/libcuda.so.1 /usr/lib64/libnvidia-ml.so.1

# Check ldconfig registration
ldconfig -p | grep -E "libcuda.so.1|libnvidia-ml.so.1"

# Verify SONAMEs
readelf -d /usr/lib64/libvgpu-cuda.so | grep SONAME
readelf -d /usr/lib64/libvgpu-nvml.so | grep SONAME
```

Expected output:
- All shim libraries present in `/usr/lib64/`
- Symlinks point to shim libraries
- Libraries registered in ldconfig cache
- SONAMEs match: `libcuda.so.1`, `libnvidia-ml.so.1`

### Step 3: Run Verification Tests

```bash
# Test C/C++ CUDA access
cd phase3/GOAL/TEST_SCRIPTS
./test_cuda_detection.sh

# Test Python CUDA access
python3 test_python_cuda.py
```

Both tests should show:
- ✓ CUDA initialized successfully
- ✓ GPU detected (device count = 1)
- ✓ VGPU-STUB found at 0000:00:05.0

## Installation Complete

Once verification passes, the vGPU shim is ready to use. **No additional configuration needed!**

Any application using CUDA or NVML will automatically use the vGPU shim.

## Troubleshooting

### Libraries Not Found

If applications can't find CUDA libraries:

```bash
# Verify symlinks exist
ls -la /usr/lib64/libcuda.so.1 /usr/lib64/libnvidia-ml.so.1

# Rebuild ldconfig cache
sudo ldconfig

# Check library path
echo $LD_LIBRARY_PATH
```

### GPU Not Detected

If GPU is not detected:

```bash
# Verify vGPU-STUB device exists
lspci | grep -i nvidia
ls -la /sys/bus/pci/devices/0000:00:05.0/

# Check shim logs
dmesg | grep -i vgpu
journalctl | grep -i "libvgpu"
```

### Compilation Errors

If `install.sh` fails:

```bash
# Check dependencies
gcc --version
make --version

# Check include files
ls -la phase3/include/
ls -la phase3/guest-shim/
```

## Next Steps

After installation:
1. Run verification tests (see `VERIFICATION.md`)
2. Test with your applications
3. No special configuration needed - just use CUDA/NVML as normal!
