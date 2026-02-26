# Missing Dependencies Symlinks Created

## Date: 2026-02-26

## Issue Found

`libggml-cuda.so` has missing dependencies:
- `libggml-base.so.0 => not found`
- `libcublas.so.12 => not found`
- `libcublasLt.so.12 => not found`

## Solution

Created symlinks in `/usr/local/lib/ollama/` (where Ollama's backend scanner looks):
- `/usr/local/lib/ollama/libcublas.so.12` → `/usr/local/lib/ollama/cuda_v12/libcublas.so.12`
- `/usr/local/lib/ollama/libcublasLt.so.12` → `/usr/local/lib/ollama/cuda_v12/libcublasLt.so.12`
- `/usr/local/lib/ollama/libggml-base.so.0` → `/usr/local/lib/ollama/libggml-base.so.0.0.0`

## Why This Matters

Ollama's discovery:
1. Tries to load `libggml-cuda.so` directly
2. If dependencies are missing, loading fails
3. If loading fails, `initial_count=0` and CPU mode is used

With symlinks in place, `libggml-cuda.so` should be able to load successfully when Ollama runs with `OLLAMA_LIBRARY_PATH` set to include `/usr/local/lib/ollama`.

## Note

`ldd` shows "not found" because it doesn't use `LD_LIBRARY_PATH`, but Ollama sets `OLLAMA_LIBRARY_PATH` which includes `/usr/local/lib/ollama`, so the dependencies should be found at runtime.

## Next Steps

1. Verify `libggml-cuda.so` can load successfully
2. Verify `ggml_backend_cuda_init()` succeeds
3. Verify GPU mode is active (`initial_count=1`, `library=cuda`)
