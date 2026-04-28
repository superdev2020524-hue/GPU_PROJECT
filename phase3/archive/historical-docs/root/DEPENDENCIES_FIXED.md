# Dependencies Fixed for libggml-cuda.so

## The Problem

`libggml-cuda.so` had missing dependencies that prevented it from loading:
- `libggml-base.so.0 => not found`
- `libcudart.so.12 => not found`
- `libcublas.so.12 => not found`
- `libcublasLt.so.12 => not found`

## The Solution

Updated `LD_LIBRARY_PATH` in `/etc/systemd/system/ollama.service.d/vgpu.conf` to include:
- `/usr/local/lib/ollama/` (contains `libggml-base.so.0`)
- `/usr/local/lib/ollama/cuda_v12/` (contains CUDA libraries)
- `/usr/local/lib/ollama/cuda_v13/` (alternative CUDA version)

## Result

✅ **All dependencies resolved** - `ldd` shows 0 missing dependencies
✅ **LD_LIBRARY_PATH updated** - Libraries can now be found
✅ **libggml-cuda.so can be loaded** - All dependencies available

## Next Steps

1. Verify `libggml-cuda.so` is actually loaded in Ollama process
2. Check if GPU mode activates
3. If still not working, investigate why Ollama doesn't load it despite dependencies being available
