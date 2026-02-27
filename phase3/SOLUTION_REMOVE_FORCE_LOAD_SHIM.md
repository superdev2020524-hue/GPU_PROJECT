# Solution: Remove force_load_shim Wrapper

## Date: 2026-02-26

## Critical Discovery

**`force_load_shim` wrapper is causing conflicts with LD_PRELOAD!**

### The Problem

1. **`force_load_shim` only loads 2 shims:**
   - `libvgpu-cuda.so` ✓
   - `libvgpu-nvml.so` ✓
   - **Missing:** `libvgpu-exec.so` ✗
   - **Missing:** `libvgpu-cudart.so` ✗

2. **LD_PRELOAD tries to load ALL 4 shims:**
   - `libvgpu-exec.so`
   - `libvgpu-cuda.so`
   - `libvgpu-nvml.so`
   - `libvgpu-cudart.so`

3. **Conflict:**
   - Wrapper loads some shims via `dlopen()`
   - LD_PRELOAD tries to load all shims
   - This creates a conflict or double-loading issue
   - Result: Crash (SEGV)

### The Solution

**Remove `force_load_shim` wrapper and use LD_PRELOAD directly.**

According to documentation:
- `libvgpu-exec.so` handles subprocess injection
- LD_PRELOAD should work for the main process
- The wrapper is redundant and causing conflicts

### Fix to Apply

**Change ExecStart in `/etc/systemd/system/ollama.service.d/vgpu.conf`:**

**From:**
```
ExecStart=/usr/local/bin/force_load_shim /usr/local/bin/ollama serve
```

**To:**
```
ExecStart=/usr/local/bin/ollama serve
```

Or if ExecStart is in the main service file:
```bash
sudo sed -i 's|ExecStart=.*force_load_shim.*ollama serve|ExecStart=/usr/local/bin/ollama serve|g' /etc/systemd/system/ollama.service
```

### Why This Should Work

1. **LD_PRELOAD will handle all shims** - No conflict
2. **libvgpu-exec.so will handle subprocesses** - Runner will get shims
3. **No double-loading** - Each shim loads once
4. **Matches working configuration** - From previous documentation

### Verification Steps

1. **Remove wrapper from ExecStart**
2. **Ensure LD_PRELOAD is set** (should already be set)
3. **Restart Ollama**
4. **Check if it starts without crashing**

### Expected Result

After removing `force_load_shim`:
- ✅ Ollama starts without crashing
- ✅ All shims load via LD_PRELOAD
- ✅ Subprocesses get shims via `libvgpu-exec.so`
- ✅ Discovery works

## Summary

**Root Cause:** `force_load_shim` wrapper conflicts with LD_PRELOAD by loading some shims while LD_PRELOAD tries to load all shims.

**Solution:** Remove the wrapper and use LD_PRELOAD directly.
