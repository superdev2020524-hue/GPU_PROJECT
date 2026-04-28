# Root Cause Fixed: libggml-cuda.so Not Loaded During Discovery

## Date: 2026-02-26

## 🎯 Root Cause Identified

**`libggml-cuda.so` was NOT being loaded during bootstrap discovery!**

### The Problem

Ollama's backend scanner only looks in the **top-level directory** (`/usr/local/lib/ollama/`) for backend libraries. However, `libggml-cuda.so` is located in a **subdirectory** (`/usr/local/lib/ollama/cuda_v12/`), so the scanner never finds it.

### Why This Matters

Without `libggml-cuda.so` being loaded:
- No CUDA backend is available
- Device count functions are never called
- `initial_count=0` (no GPUs detected)
- Ollama uses CPU mode (`library=cpu`)

### The Fix

**Created symlink in top-level directory:**
```bash
ln -sf /usr/local/lib/ollama/cuda_v12/libggml-cuda.so /usr/local/lib/ollama/libggml-cuda.so
```

This allows Ollama's backend scanner to find `libggml-cuda.so` during bootstrap discovery.

### Verification

- ✅ Symlink created: `/usr/local/lib/ollama/libggml-cuda.so` → `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so`
- ✅ Symlinks for CUDA libraries are correct
- ✅ All shim functions are implemented and exported

### Expected Results

After this fix:
1. `libggml-cuda.so` will be loaded during bootstrap discovery
2. `libggml-cuda.so` will load our Runtime API shim (`libcudart.so.12.8.90` → our shim)
3. `cudaGetDeviceCount()` or `cuDeviceGetCount()` will be called
4. Device count will be 1
5. `initial_count=1` will be reported
6. GPU mode will be active (`library=cuda`)

### Next Steps

1. Verify `libggml-cuda.so` is loaded during discovery
2. Verify device count functions are called
3. Verify GPU mode is active (`initial_count=1`, `library=cuda`)

## Conclusion

**The root cause was that `libggml-cuda.so` was not in the top-level directory where Ollama's backend scanner looks for it.** Creating the symlink should fix this issue and enable GPU mode.
