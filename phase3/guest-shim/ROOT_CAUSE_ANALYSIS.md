# Root Cause Analysis: Ollama Discovery Failure

## Critical Discovery

After deep analysis of Ollama's discovery logic, we identified the **root cause**:

### The Problem

**Ollama's main process does NOT have our shims loaded**, despite systemd configuration showing `LD_PRELOAD` is set.

### Evidence

1. **Systemd Configuration**: ✓ Correct
   - `LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-syscall.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so`
   - Environment variables are set in `/etc/systemd/system/ollama.service.d/vgpu.conf`

2. **Shim Installation**: ✓ Correct
   - All shim libraries exist in `/usr/lib64/`
   - Manual loading works: `LD_PRELOAD=... python3 -c "import ctypes; nvml=ctypes.CDLL('libnvidia-ml.so.1')"` succeeds

3. **Main Process (PID 57290)**: ✗ **FAILING**
   - **No `LD_PRELOAD` in process environment** (checked `/proc/57290/environ`)
   - **No shims loaded** (checked `/proc/57290/maps` - no `libvgpu` libraries)
   - **No CUDA backend loaded** (no `libggml-cuda` in maps)

### Root Cause

The systemd drop-in was created, but:
1. **Ollama was started BEFORE the drop-in was created**, OR
2. **Systemd wasn't reloaded** after the drop-in was created, OR
3. **Ollama wasn't restarted** after systemd configuration changed

Result: Ollama process doesn't have `LD_PRELOAD` in its environment, so our shims are never loaded, and discovery fails.

### What We Know Works

1. **PCI Bus ID Matching**: ✓ Perfect
   - Filesystem: `0000:00:05.0`
   - NVML: `0000:00:05.0`
   - They match exactly!

2. **NVML Functions**: ✓ All working
   - `nvmlInit_v2()` ✓
   - `nvmlDeviceGetCount_v2()` ✓
   - `nvmlDeviceGetPciInfo_v3()` ✓
   - All return correct values

3. **PCI Device File Interception**: ✓ Working
   - `read()`, `pread()`, `fopen()`, `fread()`, `fgets()` all intercepted
   - Returns correct values: `0x10de`, `0x2331`, `0x030200`

4. **Syscall Interception**: ✓ Working
   - `openat()`, `read()`, `readv()` intercepted
   - PCI device files return correct values

### The Solution

**Simple fix**: Reload systemd and restart Ollama to pick up the environment variables.

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Then verify shims are loaded:
```bash
PID=$(pgrep -f "ollama serve" | head -1)
cat /proc/$PID/environ | tr '\0' '\n' | grep LD_PRELOAD
cat /proc/$PID/maps | grep libvgpu
```

### Expected Result After Fix

Once shims are loaded:
1. Ollama's main process will have `LD_PRELOAD` in environment
2. Shims will be loaded into the process
3. Discovery will use our shims
4. PCI bus IDs will match (already confirmed)
5. Ollama should report `library=cuda` with `pci_id="0000:00:05.0"`

### Next Steps

1. **Fix the immediate issue**: Reload systemd and restart Ollama
2. **Verify shims are loaded**: Check process environment and maps
3. **Verify discovery works**: Check Ollama logs for `library=cuda`
4. **If still failing**: Investigate additional validation checks

## Summary

**Root Cause**: Ollama process doesn't have shims loaded because it was started before the systemd drop-in was created or systemd wasn't reloaded.

**Solution**: Reload systemd and restart Ollama.

**Status**: All shim functionality is correct and working. The only issue is that Ollama isn't using the shims because they're not loaded into the process.
