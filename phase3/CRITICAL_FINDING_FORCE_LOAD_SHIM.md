# Critical Finding: force_load_shim Wrapper

## Date: 2026-02-26

## Critical Discovery

**Ollama's ExecStart shows:**
```
ExecStart=/usr/local/bin/force_load_shim /usr/local/bin/ollama serve
```

**This is a wrapper program that may be causing issues!**

### What This Means

1. **Ollama is NOT started directly** - It's started via `force_load_shim` wrapper
2. **This wrapper may be interfering** with LD_PRELOAD or shim loading
3. **The crash might be from the wrapper**, not from the shims themselves

### Current Status

- Ollama is crashing with SEGV
- Crashes are happening repeatedly
- `force_load_shim` is in the ExecStart path

### Investigation Needed

1. **What is `force_load_shim`?**
   - Check if it's a custom wrapper
   - Check what it does
   - Check if it's related to shim loading

2. **Is `force_load_shim` causing the crash?**
   - Test starting Ollama directly without the wrapper
   - Check if the wrapper is necessary

3. **Does `force_load_shim` conflict with LD_PRELOAD?**
   - The wrapper might be trying to load shims differently
   - This could conflict with LD_PRELOAD

### Next Steps

1. **Check what `force_load_shim` does:**
   ```bash
   file /usr/local/bin/force_load_shim
   cat /usr/local/bin/force_load_shim  # if it's a script
   strings /usr/local/bin/force_load_shim  # if it's a binary
   ```

2. **Test starting Ollama directly:**
   - Modify ExecStart to: `/usr/local/bin/ollama serve`
   - Remove `force_load_shim` wrapper
   - See if Ollama starts

3. **Check if wrapper is needed:**
   - According to documentation, `libvgpu-exec.so` should handle subprocess injection
   - The wrapper might be redundant or conflicting

### Possible Solution

If `force_load_shim` is causing issues:
1. Remove it from ExecStart
2. Use direct: `ExecStart=/usr/local/bin/ollama serve`
3. Let `libvgpu-exec.so` handle subprocess injection via LD_PRELOAD

### Documentation to Check

- Check if `force_load_shim` was mentioned in previous documentation
- Check if it's supposed to be there
- Check if removing it was tried before

## Summary

**Critical finding:** Ollama is started via `force_load_shim` wrapper, not directly. This wrapper may be causing the crash or conflicting with LD_PRELOAD. Need to investigate what this wrapper does and if it's necessary.
