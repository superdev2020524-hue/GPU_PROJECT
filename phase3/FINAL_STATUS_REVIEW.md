# Final Status Review - Comprehensive Investigation and Fixes

## Investigation Summary

After thorough investigation, we identified and fixed the root causes of why libraries weren't loading:

### Root Causes Identified

1. **Missing SONAME** ✓ FIXED
   - **CUDA shim**: Lacked `SONAME=libcuda.so.1`
   - **NVML shim**: Lacked `SONAME=libnvidia-ml.so.1`
   - **Impact**: `dlopen("libcuda.so.1")` and `dlopen("libnvidia-ml.so.1")` couldn't find libraries
   - **Fix**: Rebuilt both with `-Wl,-soname,libcuda.so.1` and `-Wl,-soname,libnvidia-ml.so.1`
   - **Status**: ✅ Both verified correct

2. **Wrong Symlinks** ⚠ PARTIALLY FIXED
   - **Problem**: Some symlinks in `/usr/lib/x86_64-linux-gnu` and `/lib/x86_64-linux-gnu` point to real NVIDIA libraries
   - **Impact**: Even with correct SONAME, symlink resolution fails
   - **Fix**: Need to ensure ALL symlinks point to shims
   - **Status**: ⚠ Some symlinks still need fixing

3. **ensure_init() Not Aggressive Enough** ✓ FIXED
   - **Problem**: `cuInit()` wasn't being called early enough
   - **Impact**: GPU discovery fails because `cuInit()` must be called before `cuDeviceGetPCIBusId()`
   - **Fix**: Modified `ensure_init()` to call `cuInit()` immediately when safe for application processes
   - **Status**: ✅ Code modified and rebuilt

## Current Status

### ✅ Completed Fixes

1. **CUDA shim SONAME**: `libcuda.so.1` ✓
2. **NVML shim SONAME**: `libnvidia-ml.so.1` ✓
3. **CUDA symlinks in /usr/lib64**: All correct ✓
4. **NVML symlinks in /usr/lib64**: All correct ✓
5. **ensure_init() modification**: Calls `cuInit()` early ✓
6. **Build errors**: All fixed ✓

### ⚠ Remaining Issues

1. **Some symlinks still wrong**:
   - `/usr/lib/x86_64-linux-gnu/libcuda.so.1` → points to real library
   - `/lib/x86_64-linux-gnu/libcuda.so.1` → points to real library
   - `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1` → points to real library
   - `/lib/x86_64-linux-gnu/libnvidia-ml.so.1` → points to real library

2. **Libraries still not loading into process memory**
   - Despite correct SONAME, libraries aren't appearing in `/proc/PID/maps`
   - This suggests `dlopen()` may not be called, or something else is preventing loading

3. **GPU mode still CPU**
   - No `library=cuda` in logs
   - No initialization messages

## Why Libraries Still Aren't Loading

Even with correct SONAME and most symlinks fixed, libraries aren't loading. Possible reasons:

### 1. Symlinks in Standard Paths Still Wrong
- `/usr/lib/x86_64-linux-gnu` and `/lib/x86_64-linux-gnu` are standard library paths
- If symlinks there point to real library, `dlopen()` may find the wrong one first
- **Solution**: Fix ALL symlinks in these paths

### 2. Ollama May Not Be Calling dlopen()
- Ollama uses `dlopen()` at runtime to load CUDA libraries
- If GPU discovery fails early (before dlopen), libraries never load
- **Solution**: Verify if `dlopen()` is being called using `strace`

### 3. Go Runtime May Clear LD_LIBRARY_PATH
- Go runtime is known to clear `LD_PRELOAD`
- May also affect `LD_LIBRARY_PATH`
- **Solution**: Check process environment, consider `LD_PRELOAD` as backup

### 4. Library Search Order
- Even with correct SONAME, if real library is found first in search path, it may be loaded instead
- **Solution**: Ensure shim libraries are found before real libraries

## Next Steps

### Immediate Actions

1. **Fix remaining symlinks**:
   ```bash
   sudo rm -f /usr/lib/x86_64-linux-gnu/libcuda.so.1
   sudo ln -sf /usr/lib64/libvgpu-cuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1
   sudo rm -f /lib/x86_64-linux-gnu/libcuda.so.1
   sudo ln -sf /usr/lib64/libvgpu-cuda.so /lib/x86_64-linux-gnu/libcuda.so.1
   # Same for NVML
   ```

2. **Verify dlopen() is being called**:
   ```bash
   strace -p $(pgrep -f "ollama serve") -e trace=openat,open 2>&1 | grep -E "libcuda|libnvidia"
   ```

3. **Test library loading manually**:
   ```bash
   LD_LIBRARY_PATH=/usr/lib64:/usr/lib/x86_64-linux-gnu python3 -c "import ctypes.util; print(ctypes.util.find_library('cuda'))"
   ```

### Alternative Approaches

1. **Use LD_PRELOAD as backup**:
   - Add to systemd environment
   - Even if Go clears it, may help with initial loading

2. **Check if real NVIDIA libraries need to be removed**:
   - If real libraries exist, they may be found first
   - Consider moving/renaming them

3. **Verify NVML discovery works**:
   - NVML must succeed before CUDA loads
   - Test NVML shim independently

## Files Modified

1. `libvgpu_cuda.c`:
   - Modified `ensure_init()` to call `cuInit()` early
   - Fixed `write()` return value handling
   - Fixed `strncpy()` truncation warning

2. `libvgpu_nvml.c`:
   - Fixed unused variable/function warnings

3. `cuda_transport.c`:
   - Fixed `snprintf()` truncation warnings
   - Fixed `strncpy()` truncation warning

4. Build commands:
   - Added `-Wl,-soname,libcuda.so.1` for CUDA shim
   - Added `-Wl,-soname,libnvidia-ml.so.1` for NVML shim

## Verification Commands

```bash
# Check SONAMEs
readelf -d /usr/lib64/libvgpu-cuda.so | grep SONAME
readelf -d /usr/lib64/libvgpu-nvml.so | grep SONAME

# Check symlinks
readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1
readlink -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1

# Check libraries in memory
sudo cat /proc/$(pgrep -f "ollama serve" | head -1)/maps | grep -E "libcuda|libvgpu|libnvidia-ml"

# Check GPU mode
journalctl -u ollama --since "5 minutes ago" | grep -i "library="
```

## Conclusion

We've fixed the critical root causes (SONAME and most symlinks), but libraries still aren't loading. The remaining issue is likely:
1. Some symlinks still pointing to wrong library
2. Ollama not calling `dlopen()`, or
3. Something else preventing library loading

The next step is to fix the remaining symlinks and verify if `dlopen()` is actually being called.
