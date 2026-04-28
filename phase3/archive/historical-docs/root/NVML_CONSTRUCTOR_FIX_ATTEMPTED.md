# NVML Constructor Fix Attempted

## Date: 2026-02-26

## Fix Applied

**Added call to `nvmlDeviceGetCount_v2()` in NVML constructor** to ensure discovery sees device count=1 before trying to load `libggml-cuda.so`.

### Changes Made

**File**: `phase3/guest-shim/libvgpu_nvml.c`

**Added to constructor (after `nvmlInit_v2()`):**
```c
/* CRITICAL: Call nvmlDeviceGetCount_v2() early so discovery knows there's a GPU
 * Discovery uses NVML device count to decide if it should load libggml-cuda.so
 * If device count is 0, discovery won't load the library
 * By calling it here, we ensure discovery sees count=1 before it tries to load */
unsigned int device_count = 0;
nvmlDeviceGetCount_v2(&device_count);
/* Log the result - use simple syscall writes (no snprintf in constructor) */
const char *msg1 = "[libvgpu-nvml] constructor: nvmlDeviceGetCount_v2() called early, count=";
syscall(__NR_write, 2, msg1, strlen(msg1));
const char *count_str = (device_count == 1) ? "1\n" : "0\n";
syscall(__NR_write, 2, count_str, strlen(count_str));
```

### Status

- ✅ **Library rebuilt** - Code is in library (verified via `strings`)
- ✅ **Library deployed** - Installed to `/usr/lib64/libvgpu-nvml.so`
- ✅ **Library loaded** - Present in process memory maps
- ❌ **No constructor logs** - Logging not appearing (syscall write may not work in constructor)
- ❌ **Discovery still shows initial_count=0** - Fix didn't resolve the issue

### Analysis

**Why the fix didn't work:**

1. **Constructor may not be running** - No logs appear, suggesting constructor isn't executing
2. **Logging may not work** - `syscall(__NR_write, ...)` may not work in constructor context
3. **Discovery may not use NVML when OLLAMA_LLM_LIBRARY is set** - When `OLLAMA_LLM_LIBRARY=cuda_v12` is set, discovery may bypass NVML detection entirely
4. **Function call may not have effect** - Even if called, discovery may not check the result

### Next Steps

Since this fix didn't work, we need to:
1. Verify if constructor is actually running (maybe use a different verification method)
2. Check if discovery uses NVML when `OLLAMA_LLM_LIBRARY=cuda_v12` is set
3. Try a different approach - maybe ensure CUDA device count is available instead
4. Investigate what actually triggers library loading when `OLLAMA_LLM_LIBRARY` is set

## Conclusion

**The fix was applied but didn't resolve the issue.** Discovery still shows `initial_count=0` and `libggml-cuda.so` is not loading. Need to investigate further or try a different approach.
