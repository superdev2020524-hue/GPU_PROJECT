# Critical Fix: libcudart.so.12.8.90 Symlink

## Issue Found

The user correctly identified that this error was already solved before. The issue was:

**`libcudart.so.12.8.90` was NOT symlinked to our shim!**

This is documented in `ROOT_CAUSE_AND_SOLUTION.md` as Solution Option 1, step 3:
- Backup original: `/usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90`
- Create symlink: `libcudart.so.12.8.90 -> /usr/lib64/libvgpu-cudart.so`

## Why This Matters

`libggml-cuda.so` loads `libcudart.so.12.8.90` specifically (the versioned file), not just `libcudart.so.12`. Without this symlink:
- `libggml-cuda.so` loads the original `libcudart.so.12.8.90` (not our shim)
- `cudaRuntimeGetVersion()` is never called (because it's not in our shim)
- Runtime API functions don't work

## Fix Applied

✅ **Symlink created**: `/usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90 -> /usr/lib64/libvgpu-cudart.so`

**Verification**:
```
lrwxrwxrwx 1 root root 28 Feb 25 14:26 /usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90 -> /usr/lib64/libvgpu-cudart.so
```

## Expected Result

Now that `libcudart.so.12.8.90` points to our shim:
1. ✅ `libggml-cuda.so` will load our Runtime API shim
2. ✅ `cudaRuntimeGetVersion()` should be called
3. ✅ Runtime API functions should work
4. ✅ `ggml_backend_cuda_init` should proceed further

## Next Steps

1. Restart Ollama (if not already done)
2. Check logs for `cudaRuntimeGetVersion() CALLED`
3. Verify that Runtime API functions are being called
4. Check if device query functions are now being called

## Lesson Learned

**Always check previous solutions!** The user was right - this issue was already identified and solved before. The symlink was either:
- Never created in the first place
- Reverted/removed at some point
- Lost during a system update or reinstall

This is why it's important to verify that all documented fixes are actually in place.
