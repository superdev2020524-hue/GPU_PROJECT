# Discovery Path Analysis

## Date: 2026-02-27

## Findings

### 1. Ollama Uses NVML
- ✅ `strings ollama` shows `nvmlInit_v2`
- ✅ NVML shim is installed (`libnvidia-ml.so.1` → `libvgpu-nvml.so`)

### 2. Runtime API Shim Exists
- ✅ `cudaGetDeviceCount()` is implemented in `libvgpu-cudart.so`
- ✅ Returns count=1 immediately
- ✅ Logs show it's called during model execution (pid=212717)

### 3. Key Question
**Are NVML or Runtime API functions called during bootstrap discovery (13:52:26)?**

If NOT called → discovery uses a different mechanism
If called but returns 0 → shim has an issue
If called and returns 1 → discovery has validation that rejects it

## Next Steps

1. Check logs around bootstrap discovery time (13:52:26)
2. Look for NVML constructor/init calls
3. Look for Runtime API calls
4. If none found, discovery might use file system checks (`/dev/nvidia*`, `/proc/driver/nvidia/`)

## Hypothesis

Discovery might be checking:
- `/dev/nvidia0` (device node)
- `/proc/driver/nvidia/version` (driver info)
- Or using a different validation path

If these don't exist, discovery reports `initial_count=0` even if shims work.
