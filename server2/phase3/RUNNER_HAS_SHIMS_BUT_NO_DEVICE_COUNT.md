# Runner Has Shims But Device Count Functions Not Called

## Critical Finding

**Runner subprocess has shim libraries loaded, but device count functions are NOT being called!**

### Current Status

✅ **Runner subprocess exists** - PID 63100
✅ **Runner has shim libraries loaded** - libvgpu-cuda.so and libvgpu-nvml.so in memory maps
✅ **cuInit() is called in runner** - Device discovery works
✅ **Device found** - "device found at 0000:00:05.0"
❌ **Device count functions NOT called** - nvmlDeviceGetCount_v2() and cuDeviceGetCount() never called
❌ **GPU mode still CPU** - library=cpu

## What This Means

Ollama's discovery process:
1. Spawns runner subprocess ✓
2. Runner has shims loaded ✓ (exec interception working!)
3. Runner calls cuInit() ✓
4. Device discovery works ✓
5. **But runner doesn't call device count functions** ✗
6. **So libggml-cuda.so never loads** ✗
7. **GPU mode remains CPU** ✗

## The Problem

**Ollama's discovery doesn't use standard device count functions!**

Even though:
- Libraries are loaded
- Initialization works
- Device discovery works

Ollama still doesn't call:
- `nvmlDeviceGetCount_v2()`
- `cuDeviceGetCount()`

This suggests Ollama uses a different discovery mechanism that doesn't rely on device count functions.

## Next Steps

1. **Check if Ollama uses different API** - Maybe uses device handles directly
2. **Check if discovery uses PCI matching** - Maybe matches PCI devices without calling count functions
3. **Check Ollama source code** - Understand exactly how discovery determines GPU availability
4. **Check if there's a different prerequisite** - Maybe checks something else we're not handling
