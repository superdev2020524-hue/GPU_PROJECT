# Breakthrough: NVML Shim Is Now Loading!

## Success

**NVML shim is now loading via LD_PRELOAD!**

### Evidence

1. **Main process**: `libvgpu-nvml.so` is in memory maps ✓
2. **Runner process**: `libvgpu-nvml.so` is in memory maps (5 references) ✓
3. **LD_PRELOAD**: Set to `/usr/lib64/libvgpu-nvml.so` ✓

### What Changed

Added `LD_PRELOAD=/usr/lib64/libvgpu-nvml.so` to systemd configuration.

Even though the config previously said not to use LD_PRELOAD (because Go runtime clears it), it actually works for the main process and runner subprocesses inherit it.

## Current Status

- ✅ NVML shim is loading
- ⚠ GPU mode is still CPU
- ⚠ No shim messages in logs (might go to stderr or be suppressed)

## Next Steps

1. **Check if shim is being called**
   - Shim prints to stderr
   - Need to check if messages appear
   - Or if they're suppressed

2. **Check if CUDA libraries load**
   - If NVML discovery succeeds, CUDA should load
   - Need to verify if `libggml-cuda.so` loads
   - Need to verify if `libcuda.so.1` (our shim) loads

3. **Check discovery process**
   - Now that NVML is loaded, does discovery succeed?
   - Does it proceed to CUDA loading?
   - What's the actual discovery result?

## Key Insight

**LD_PRELOAD works!** Even though we thought Go runtime would clear it, it actually works for loading libraries into the process.

Now that NVML is loading, we need to:
1. Verify it's being called
2. Verify discovery succeeds
3. Verify CUDA loads as a result
