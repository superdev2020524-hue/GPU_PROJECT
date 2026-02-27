# Crash Issue Timeline and Investigation

## Date: 2026-02-26

## Timeline: What Happened

### ✅ Feb 25 - Working State (BREAKTHROUGH_SUMMARY.md)
- **GPU Detection**: WORKING
  - Discovery: 302ms
  - GPU detected: NVIDIA H100 80GB HBM3
  - `library=/usr/local/lib/ollama/cuda_v12`
  - `libggml-cuda.so` loading successfully
  - All versioned symbols resolved

### ❌ Crashes Started
**Root Causes:**
1. **`libvgpu-syscall.so` in LD_PRELOAD but file doesn't exist**
   - File `/usr/lib64/libvgpu-syscall.so` doesn't exist
   - But it's listed in `LD_PRELOAD`
   - Result: Crash (SEGV) when systemd tries to load it

2. **`force_load_shim` wrapper conflicts**
   - Wrapper only loads 2 shims (cuda, nvml)
   - Missing: `libvgpu-exec.so` and `libvgpu-cudart.so`
   - LD_PRELOAD tries to load all 4 shims
   - Result: Conflict/double-loading → Crash (SEGV)

**Impact:**
- Ollama crashed immediately on startup
- Discovery never ran (crashes prevented it)
- GPU detection couldn't happen

### ✅ Crashes Fixed
**Fixes Applied:**
1. Removed `libvgpu-syscall.so` from `LD_PRELOAD`
2. Removed `force_load_shim` wrapper from `ExecStart`
3. Added `OLLAMA_LIBRARY_PATH` to `vgpu.conf`
4. Fixed `LD_PRELOAD` order: `exec:cuda:nvml:cudart`

**Result:**
- ✅ Ollama starts without crashing
- ✅ Service runs stable (20+ minutes uptime)
- ✅ All shim libraries loaded

### ❌ Current Issue: Discovery Not Detecting GPU
**Status:**
- ✅ Ollama running stable (crashes fixed)
- ✅ Configuration correct (all fixes applied)
- ✅ Symlink exists: `/usr/local/lib/ollama/libggml-cuda.so` → `cuda_v12/libggml-cuda.so`
- ✅ `OLLAMA_LIBRARY_PATH` set: `/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12`
- ❌ **No discovery logs found**
- ❌ **No `initial_count=1`**
- ❌ **No `library=cuda`**
- ❌ **GPU not detected**

## The Key Question

**Why is discovery not detecting GPU now that crashes are fixed?**

### Possible Reasons

1. **Discovery is not running**
   - Maybe discovery only runs under certain conditions
   - Maybe discovery was disabled/failing silently
   - Maybe discovery needs to be triggered somehow

2. **Discovery is running but failing silently**
   - Maybe discovery runs but doesn't find GPU
   - Maybe discovery runs but doesn't log
   - Maybe discovery fails before reaching GPU detection

3. **Something changed from Feb 25 working state**
   - Maybe a configuration is missing
   - Maybe a file/symlink is wrong
   - Maybe environment variables are different

4. **Discovery needs to be triggered**
   - Maybe discovery only runs when a model is executed
   - Maybe discovery needs an API call to trigger it
   - Maybe discovery runs in a subprocess we're not seeing

## What Needs Investigation

1. **Check if discovery is actually running**
   - Look for any discovery-related logs (even if not GPU-specific)
   - Check if discovery runs at startup or on demand
   - Check if discovery runs in main process or subprocess

2. **Compare with Feb 25 working state**
   - What was the exact configuration when it worked?
   - What environment variables were set?
   - What files/symlinks existed?
   - What was the exact LD_PRELOAD order?

3. **Check if discovery can be triggered**
   - Try making an API call that might trigger discovery
   - Check if discovery runs when a model is loaded
   - Check if there's a way to force discovery

4. **Verify all prerequisites are correct**
   - Check if `libggml-cuda.so` can be loaded manually
   - Check if shim functions are working
   - Check if PCI device discovery is working

## Conclusion

**The crashes were PREVENTING discovery from running.**
- Before: Crashes → Discovery never runs → GPU not detected
- After fixes: No crashes → Discovery should run → But GPU still not detected

**The real question is: Why isn't discovery detecting the GPU now that it can run?**

This is what needs to be investigated next.
