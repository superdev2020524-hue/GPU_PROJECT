# Root Cause: nvmlDeviceGetCount_v2() Not Being Called

## Date: 2026-02-26

## Critical Finding

**Discovery uses NVML device count, but `nvmlDeviceGetCount_v2()` is NOT being called during discovery!**

### Evidence

1. **CUDA Device Count Works:**
   ```
   [libvgpu-cuda] cuDeviceGetCount() CALLED (pid=148445)
   [libvgpu-cuda] cuDeviceGetCount() SUCCESS: returning count=1
   ```
   ✅ Main process gets count=1

2. **NVML Device Count NOT Called:**
   ```
   No logs showing: [libvgpu-nvml] nvmlDeviceGetCount_v2() CALLED
   ```
   ❌ `nvmlDeviceGetCount_v2()` is never called

3. **Discovery Reports initial_count=0:**
   ```
   initial_count=0
   library=cpu
   ```
   ❌ Discovery reports no GPU

### Why This Matters

**Ollama's discovery uses NVML device count, not CUDA device count!**

Even though:
- ✅ `cuDeviceGetCount()` returns 1
- ✅ `nvmlDeviceGetCount_v2()` is implemented and returns 1
- ✅ NVML shim symlink is correct

Discovery still reports `initial_count=0` because `nvmlDeviceGetCount_v2()` is **not being called**.

### Possible Reasons

1. **Discovery doesn't load NVML library**
   - Maybe discovery skips NVML when `OLLAMA_LLM_LIBRARY=cuda_v12` is set
   - Maybe discovery uses a different mechanism

2. **Discovery loads NVML but doesn't call device count**
   - Maybe discovery checks something else first
   - Maybe discovery fails before reaching device count

3. **Discovery uses a different NVML function**
   - Maybe uses `nvmlDeviceGetCount()` (without _v2)
   - Maybe uses a different API

4. **Runner subprocess doesn't have NVML shim loaded**
   - Even though symlink is correct
   - Maybe runner loads from a different path

## Next Steps

1. **Check if `nvmlDeviceGetCount()` (without _v2) is being called**
   - Maybe discovery uses the non-versioned function

2. **Check if discovery loads NVML library at all**
   - Check if `libnvidia-ml.so.1` is opened during discovery

3. **Check if `OLLAMA_LLM_LIBRARY=cuda_v12` affects discovery**
   - Maybe this setting skips NVML discovery

4. **Add logging to all NVML functions**
   - See what NVML functions discovery actually calls

## Conclusion

**The root cause is that `nvmlDeviceGetCount_v2()` is not being called during discovery, even though it's implemented and should return 1.**

This is why `initial_count=0` and GPU mode is not active.
