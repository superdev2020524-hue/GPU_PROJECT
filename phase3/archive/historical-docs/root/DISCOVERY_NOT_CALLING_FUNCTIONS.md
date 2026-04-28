# Discovery Not Calling Device Count Functions

## Current Status

✅ **Symbols exported**: `nvmlInit_v2`, `nvmlDeviceGetCount_v2`, `cuInit`, `cuDeviceGetCount` all exported
✅ **Functions implemented**: All device query functions implemented and return correct values
✅ **Initialization working**: `cuInit()` and `nvmlInit_v2()` are called from constructors
❌ **Device count functions NOT called**: `cuDeviceGetCount()` and `nvmlDeviceGetCount_v2()` never called
❌ **libggml-cuda.so NOT loaded**: Ollama never loads the CUDA backend library
❌ **GPU mode is CPU**: Result is `library=cpu`

## The Problem

Ollama's discovery process:
1. Loads our shim libraries ✓ (via LD_PRELOAD)
2. Calls `nvmlInit_v2()` ✓ (from constructor)
3. Calls `cuInit()` ✓ (from constructor)
4. **Should call `nvmlDeviceGetCount_v2()`** ❌ (NOT HAPPENING)
5. **Should call `cuDeviceGetCount()`** ❌ (NOT HAPPENING)
6. **Should load `libggml-cuda.so` if count > 0** ❌ (NEVER HAPPENS)
7. **Should use GPU mode** ❌ (FALLS BACK TO CPU)

## Why Functions Aren't Called

Possible reasons:
1. **Discovery fails before calling functions** - Maybe checks something else first that fails
2. **Discovery uses different mechanism** - Maybe doesn't use dlsym() or calls functions differently
3. **Discovery happens in subprocess** - Maybe in a process we're not intercepting
4. **Symbol versioning issue** - Maybe Ollama expects specific symbol versions we don't have
5. **Discovery checks library loading** - Maybe just checks if library can be loaded, doesn't call functions

## Next Steps

1. **Add comprehensive logging** to ALL NVML/CUDA functions to see what Ollama actually calls
2. **Check if discovery happens in subprocess** - Verify all Ollama processes have LD_PRELOAD
3. **Test dlsym() directly** - Verify Ollama can actually find our functions
4. **Check Ollama source code** - Understand exactly how discovery works
5. **Add early logging** - Log when library loads, when functions are resolved, etc.

## Key Insight

Since `cuInit()` and `nvmlInit_v2()` ARE called (from constructors), but device count functions are NOT called, this suggests:
- Ollama's discovery might check initialization success before calling device count
- Or discovery uses a different code path that doesn't call these functions
- Or there's an error/check that prevents discovery from proceeding
