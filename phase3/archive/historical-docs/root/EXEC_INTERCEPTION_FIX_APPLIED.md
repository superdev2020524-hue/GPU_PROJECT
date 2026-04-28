# Exec Interception Fix Applied

## Date: 2026-02-26 07:51

## Fix Applied

**Removed `libvgpu-syscall.so` from SHIM_LIBS in exec interception code**

### What Was Fixed

The exec interception code (`libvgpu_exec.c`) was including `libvgpu-syscall.so` in the `SHIM_LIBS` constant, but that library was removed because it was causing issues. This might have prevented LD_PRELOAD injection.

**Before:**
```c
static const char *SHIM_LIBS = "libvgpu-exec.so:libvgpu-syscall.so:libvgpu-cuda.so:libvgpu-nvml.so:libvgpu-cudart.so";
```

**After:**
```c
static const char *SHIM_LIBS = "libvgpu-exec.so:libvgpu-cuda.so:libvgpu-nvml.so:libvgpu-cudart.so";
```

### Actions Taken

1. ✅ Fixed `libvgpu_exec.c` to remove `libvgpu-syscall.so` from SHIM_LIBS
2. ✅ Rebuilt `libvgpu-exec.so` on VM
3. ✅ Restarted Ollama service
4. ✅ Verified fix is applied

### Current Status

- ✅ Fix applied and library rebuilt (timestamp: Feb 26 07:51)
- ❌ Exec interception logs still not appearing
- ❌ LD_PRELOAD still not in runner environment (from earlier logs)

### Why Exec Interception May Not Be Working

Possible reasons:
1. **Ollama uses different mechanism** - Go runtime may use direct syscalls (clone, fork+exec) that bypass exec interception
2. **Ollama clears environment** - May explicitly clear LD_PRELOAD for security
3. **Exec functions not called** - Ollama may not use execve/execv/execvp to spawn runner
4. **Logs not captured** - Exec interception may happen but logs go elsewhere

### Next Steps

Since exec interception doesn't appear to be working:
1. **Ensure shims work via symlinks** - Test if shims work when loaded via symlinks (not LD_PRELOAD)
2. **Investigate Ollama subprocess mechanism** - Find how Ollama actually spawns runner subprocesses
3. **Alternative approach** - Find another way to ensure runner gets shims

## Conclusion

The fix was applied, but exec interception still doesn't appear to be working. This suggests Ollama uses a different mechanism to spawn subprocesses that bypasses exec interception. Need to investigate alternative approaches.
