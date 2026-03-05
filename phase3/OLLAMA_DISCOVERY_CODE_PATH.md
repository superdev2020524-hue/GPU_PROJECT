# Ollama GPU discovery code path (from upstream source)

## Summary

- **Device list:** Comes from `ml.Backend.BackendDevices()` (see `ml/backend.go`). Each backend (e.g. ggml) implements this and returns `[]DeviceInfo`.
- **Logging:** `discover.LogDetails(devices)` in `discover/types.go` logs "inference compute" for each device; if `devices` is empty it logs the CPU fallback (id=cpu, library=cpu).
- **Runner:** With `--ollama-engine`, the runner is `ollamarunner`; it uses `ml` and backends. Device discovery is done when the backend is created/initialized and `BackendDevices()` is called.
- **C APIs:** The GGML backend (llama.cpp/ggml) ultimately uses CUDA/NVML in C/CGo. Typical flow: load libggml-cuda → it depends on libcuda/libcudart → code calls `cudaGetDeviceCount` (Runtime) or `cuDeviceGetCount` (Driver) or NVML device count. Our strace showed the **child** (runner) opens `libggml-cuda.so` and `cuda_v12/libcuda.so.1` (our shim), but **neither `cuDeviceGetCount` nor `nvmlDeviceGetCount_v2` was ever called** (no debug files created). So either:
  - Discovery in this build does not call those APIs (e.g. uses a different path or cached result), or
  - The process that would call them exits or is not the one we traced.

## Next steps

1. **SEGV with full LD_PRELOAD:** Still occurs; likely from **open/openat/fopen** interception running `is_application_process()` too early. Add an early-call pass-through counter to those wrappers (like dlopen) so the main process does not run process checks until after hundreds of calls.
2. **Force discovery to use shims:** Once the exact C entry point is known (e.g. from Ollama/llama.cpp source or symbol trace), ensure that code path runs in a process that has our libs and that our implementations return device count 1.
