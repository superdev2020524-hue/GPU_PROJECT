# Verification Guide: vGPU Detection

## Quick Verification

### Test 1: C/C++ CUDA Detection

```bash
cd phase3/GOAL/TEST_SCRIPTS
./test_cuda_detection.sh
```

**Expected Output:**
```
SUCCESS: CUDA initialized, device count = 1
✓✓✓ GPU DETECTED: vGPU shim works as system library!
```

### Test 2: Python CUDA Detection

```bash
cd phase3/GOAL/TEST_SCRIPTS
python3 test_python_cuda.py
```

**Expected Output:**
```
✓ Loaded libcuda.so.1 (vGPU shim)
✓ cuInit() successful
✓ cuDeviceGetCount() returned: 1
✓✓✓ SUCCESS: Python detected vGPU!
✓✓✓ GPU MODE: Python can use vGPU via CUDA!
```

## Detailed Verification

### 1. System Library Installation

```bash
# Check shim libraries
ls -la /usr/lib64/libvgpu-*.so

# Expected:
# -rwxr-xr-x libvgpu-cuda.so
# -rwxr-xr-x libvgpu-nvml.so
# -rwxr-xr-x libvgpu-cudart.so
```

### 2. System Symlinks

```bash
# Check symlinks
ls -la /usr/lib64/libcuda.so.1 /usr/lib64/libnvidia-ml.so.1

# Expected:
# lrwxrwxrwx libcuda.so.1 -> /usr/lib64/libvgpu-cuda.so
# lrwxrwxrwx libnvidia-ml.so.1 -> /usr/lib64/libvgpu-nvml.so
```

### 3. SONAME Verification

```bash
# Check SONAMEs
readelf -d /usr/lib64/libvgpu-cuda.so | grep SONAME
readelf -d /usr/lib64/libvgpu-nvml.so | grep SONAME

# Expected:
# SONAME: libcuda.so.1
# SONAME: libnvidia-ml.so.1
```

### 4. ldconfig Registration

```bash
# Check ldconfig cache
ldconfig -p | grep -E "libcuda.so.1|libnvidia-ml.so.1"

# Expected: Libraries listed in cache
```

### 5. VGPU-STUB Device

```bash
# Check PCI device
lspci | grep -i nvidia

# Expected:
# 00:05.0 VGA compatible controller: NVIDIA Corporation Device 2331 (rev a1)

# Check device files
ls -la /sys/bus/pci/devices/0000:00:05.0/

# Expected: Device directory exists with vendor/device files
```

### 6. Runtime Detection

```bash
# Test without LD_PRELOAD (should work via system library)
unset LD_PRELOAD
./test_cuda_detection.sh

# Should still detect GPU
```

## Verification Checklist

- [ ] Shim libraries installed in `/usr/lib64/`
- [ ] System symlinks created and point to shims
- [ ] SONAMEs correct (`libcuda.so.1`, `libnvidia-ml.so.1`)
- [ ] Libraries registered with `ldconfig`
- [ ] VGPU-STUB device present at `0000:00:05.0`
- [ ] C test program detects GPU (count = 1)
- [ ] Python test program detects GPU (count = 1)
- [ ] Works without `LD_PRELOAD` (system library resolution)

## Success Criteria

✅ **All checks pass** → vGPU shim is working correctly

✅ **GPU detected in tests** → Applications will detect vGPU automatically

✅ **No special config needed** → Works via system library resolution

## Troubleshooting

### GPU Not Detected

1. Check VGPU-STUB device exists:
   ```bash
   lspci | grep -i nvidia
   ```

2. Check shim logs:
   ```bash
   dmesg | grep -i vgpu
   journalctl | grep "libvgpu"
   ```

3. Verify library loading:
   ```bash
   ldd /usr/lib64/libvgpu-cuda.so
   ```

### Libraries Not Found

1. Rebuild ldconfig cache:
   ```bash
   sudo ldconfig
   ```

2. Check library path:
   ```bash
   echo $LD_LIBRARY_PATH
   ```

3. Verify symlinks:
   ```bash
   readlink -f /usr/lib64/libcuda.so.1
   ```

### Test Programs Fail

1. Check compilation:
   ```bash
   gcc --version
   make --version
   ```

2. Check dependencies:
   ```bash
   ldd test_cuda_detection
   ```

3. Run with debug output:
   ```bash
   LD_DEBUG=libs ./test_cuda_detection
   ```
