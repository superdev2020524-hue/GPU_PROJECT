# Final Fix: Remove force_load_shim Wrapper

## Date: 2026-02-26

## Root Cause Identified

**`force_load_shim` wrapper is causing conflicts with LD_PRELOAD, resulting in crashes!**

### The Problem

1. **ExecStart currently:**
   ```
   ExecStart=/usr/local/bin/force_load_shim /usr/local/bin/ollama serve
   ```

2. **`force_load_shim` only loads 2 shims:**
   - `libvgpu-cuda.so` ✓
   - `libvgpu-nvml.so` ✓
   - **Missing:** `libvgpu-exec.so` ✗
   - **Missing:** `libvgpu-cudart.so` ✗

3. **LD_PRELOAD tries to load ALL 4 shims:**
   - This creates a conflict/double-loading issue
   - Result: Crash (SEGV)

### The Solution

**Remove `force_load_shim` wrapper and use LD_PRELOAD directly.**

## Fix to Apply

**Run on the VM:**

```bash
# Remove force_load_shim from ExecStart
sudo sed -i 's|ExecStart=.*force_load_shim.*ollama serve|ExecStart=/usr/local/bin/ollama serve|g' /etc/systemd/system/ollama.service

# Verify the change
sudo grep ExecStart /etc/systemd/system/ollama.service

# Restart Ollama
sudo systemctl daemon-reload
sudo systemctl restart ollama

# Check if it starts
sleep 10
systemctl is-active ollama
```

## Expected Result

After removing `force_load_shim`:
- ✅ Ollama starts without crashing
- ✅ All shims load via LD_PRELOAD (no conflict)
- ✅ `libvgpu-exec.so` handles subprocess injection
- ✅ Discovery works

## Why This Will Work

1. **No conflict** - LD_PRELOAD handles all shims uniformly
2. **Complete shim loading** - All 4 shims load (not just 2)
3. **Subprocess support** - `libvgpu-exec.so` ensures runner gets shims
4. **Matches working configuration** - From previous documentation

## Current Status

- ✅ Root cause identified: `force_load_shim` wrapper conflict
- ✅ Fix documented: Remove wrapper, use LD_PRELOAD directly
- ⏳ Fix ready to apply: Command provided above

## Summary

**The `force_load_shim` wrapper was causing the crash by creating conflicts with LD_PRELOAD. Removing it and using LD_PRELOAD directly should fix the issue.**
