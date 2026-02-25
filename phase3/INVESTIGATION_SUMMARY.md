# Investigation Summary: Library Loading Issue

## Root Causes Identified and Fixed

### 1. Missing SONAME ✓ FIXED
- **Problem**: Library lacked `SONAME=libcuda.so.1`
- **Impact**: `dlopen("libcuda.so.1")` couldn't find the library
- **Fix**: Rebuilt with `-Wl,-soname,libcuda.so.1`
- **Status**: ✅ Verified with `readelf -d /usr/lib64/libvgpu-cuda.so | grep SONAME`
- **Result**: `SONAME: [libcuda.so.1]`

### 2. Wrong Symlinks ✓ FIXED
- **Problem**: 
  - `/usr/lib/x86_64-linux-gnu/libcuda.so.1` pointed to real NVIDIA library
  - `/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1` pointed to real NVIDIA library
  - `/lib/x86_64-linux-gnu/libnvidia-ml.so.1` pointed to real NVIDIA library
- **Impact**: Even with correct SONAME, symlink resolution failed
- **Fix**: Removed and recreated all symlinks pointing to shim libraries
- **Status**: ✅ All symlinks now point to shim:
  - CUDA: `/usr/lib64/libvgpu-cuda.so`
  - NVML: `/usr/lib64/libvgpu-nvml.so`

### 3. Modified ensure_init() ✓ COMPLETED
- **Change**: Made `ensure_init()` more aggressive about calling `cuInit()` early
- **Status**: ✅ Code modified and library rebuilt

## Current Status

### What's Working ✓
- ✅ SONAME is correct: `libcuda.so.1`
- ✅ All CUDA symlinks point to shim
- ✅ All NVML symlinks point to shim
- ✅ Service is active and running
- ✅ No systemd errors
- ✅ Process is NOT using `force_load_shim` wrapper

### Remaining Issue ⚠
- ⚠ Libraries still not appearing in process memory maps
- ⚠ GPU mode still showing `library=cpu`
- ⚠ No initialization messages in logs

## Why Libraries Still Aren't Loading

Despite fixing SONAME and symlinks, libraries aren't loading. Possible reasons:

### 1. Ollama May Not Be Calling dlopen()
- Ollama uses `dlopen()` at runtime to load CUDA libraries
- If GPU discovery fails early (before dlopen), libraries never load
- Need to verify if `dlopen("libcuda.so.1")` is actually being called

### 2. Go Runtime May Clear LD_LIBRARY_PATH
- Go runtime is known to clear `LD_PRELOAD` and may affect `LD_LIBRARY_PATH`
- Even though systemd sets `LD_LIBRARY_PATH`, Go might clear it
- Need to verify environment in process

### 3. GPU Discovery May Fail Before dlopen()
- If NVML discovery fails, Ollama may not proceed to CUDA loading
- Need to verify NVML shim is working correctly
- Need to check if `nvmlInit_v2()` is being called

## Next Steps

### Option 1: Verify dlopen() is Being Called
1. Use `strace` to trace `dlopen()` calls
2. Check if Ollama is actually calling `dlopen("libcuda.so.1")`
3. If not, investigate why GPU discovery isn't triggering it

### Option 2: Use LD_PRELOAD as Backup
1. Add `LD_PRELOAD` to systemd environment
2. Even if Go clears it, it might help with initial loading
3. Test if libraries load with `LD_PRELOAD` set

### Option 3: Test Library Loading Manually
1. Create a test program that calls `dlopen("libcuda.so.1")`
2. Verify our library can be loaded via dlopen
3. Check if there are any errors during loading

### Option 4: Check NVML Shim
1. Verify NVML shim has correct SONAME (`libnvidia-ml.so.1`)
2. Test if `nvmlInit_v2()` works correctly
3. Ensure NVML discovery succeeds before CUDA loading

## Files Modified

1. `libvgpu_cuda.c`:
   - Modified `ensure_init()` to call `cuInit()` early
   - Fixed `write()` return value handling
   - Fixed `strncpy()` truncation warning

2. `cuda_transport.c`:
   - Fixed `snprintf()` truncation warnings
   - Fixed `strncpy()` truncation warning

3. Build command updated:
   - Added `-Wl,-soname,libcuda.so.1` flag

## Verification Commands

```bash
# Check SONAME
readelf -d /usr/lib64/libvgpu-cuda.so | grep SONAME

# Check symlinks
readlink -f /usr/lib/x86_64-linux-gnu/libcuda.so.1
readlink -f /usr/lib/x86_64-linux-gnu/libnvidia-ml.so.1

# Check libraries in memory
sudo cat /proc/$(pgrep -f "ollama serve" | head -1)/maps | grep -E "libcuda|libvgpu|libnvidia-ml"

# Check GPU mode
journalctl -u ollama --since "5 minutes ago" | grep -i "library="
```

## Conclusion

We've fixed the root causes (SONAME and symlinks), but libraries still aren't loading. This suggests the issue is at a higher level - either Ollama isn't calling `dlopen()`, or something is preventing the libraries from being found even with correct SONAME and symlinks.

The next investigation should focus on:
1. Verifying `dlopen()` is being called
2. Testing library loading manually
3. Checking if Go runtime is interfering
4. Verifying NVML shim is working correctly
