# Loader Resolution Findings

## Date: 2026-02-27

## Key Findings

### 1. Ollama Binary
- ✅ **NO RPATH or RUNPATH** - Good! No hardcoded paths interfering
- ✅ Statically linked (no libcuda dependency at link time)

### 2. System libcuda
- ✅ `/usr/lib64/libcuda.so.1` → symlink to `/usr/lib64/libvgpu-cuda.so` (our shim)
- ✅ Already pointing to our shim - this is good!

### 3. Constructor Logs
- ✅ Main process loads shim (constructor called)
- ⚠️ Need to check runner subprocess constructor logs

### 4. LD_DEBUG
- ⚠️ `ollama list` doesn't trigger libcuda loading (expected - no CUDA backend used)
- Need to test with actual model execution to trigger CUDA loading

## Hypothesis

Since system libcuda.so.1 already points to our shim, the issue might be:

1. **Runner subprocess doesn't load libcuda at all** (discovery fails before dlopen)
2. **GGML's dlopen() uses different search** (maybe relative path?)
3. **Runner uses different binary** (need to check if runner is separate executable)

## Next Steps

1. Check if runner is separate binary
2. Test with model execution to trigger CUDA loading
3. Check if GGML's dlopen() uses absolute path or relative
4. Verify runner subprocess actually calls dlopen("libcuda.so.1")
