# Filesystem Validation Findings

## Date: 2026-02-27

## ChatGPT's Hypothesis

Discovery checks filesystem paths (`/dev/nvidia*`, `/proc/driver/nvidia/`) BEFORE calling CUDA APIs.

## Our Findings

### ✅ Device Nodes Exist
```
/dev/nvidia0
/dev/nvidiactl
/dev/nvidia-uvm
/dev/nvidia-uvm-tools
```

### ❌ /proc/driver/nvidia/ Does NOT Exist
- Cannot create (procfs is read-only)
- This is expected in a VM without real NVIDIA driver

### ✅ Filesystem Interception Code Exists
- `libvgpu_cuda.c` intercepts `stat()`, `openat()`, `access()` for `/proc/driver/nvidia/*`
- Code is present and should work

### ❌ BUT: No Interception Logs
- No `stat.*nvidia` logs in stderr
- No `openat.*nvidia` logs in stderr
- No interception happening

### ❌ strace Shows NO nvidia Checks
- `strace ollama list` shows NO stat/openat calls to nvidia paths
- Discovery is NOT checking these paths!

## Conclusion

**Discovery is NOT using filesystem validation!**

This means:
1. Discovery uses a different mechanism
2. OR discovery happens in a subprocess without shim loaded
3. OR discovery uses a different validation path

## Next Steps

Need to find what discovery actually checks:
- Environment variables?
- Different library loading mechanism?
- Different validation logic?
