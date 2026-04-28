# Crash Investigation

## Date: 2026-02-26

## Current Status

**Ollama is still crashing with SEGV even after removing `libvgpu-syscall.so` from `LD_PRELOAD`.**

### What We Know

1. ✅ `libvgpu-syscall.so` removed from `LD_PRELOAD`
2. ✅ `OLLAMA_LIBRARY_PATH` is set correctly
3. ✅ `OLLAMA_LLM_LIBRARY=cuda_v12` is present (required)
4. ✅ `OLLAMA_NUM_GPU=999` is present
5. ✅ All shim libraries exist: `exec`, `cuda`, `nvml`, `cudart`
6. ✅ No systemd syntax errors
7. ❌ **Ollama still crashes with SEGV**

### Current LD_PRELOAD

```
LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so
```

### Possible Causes

1. **One of the shim libraries is causing the crash**
   - `libvgpu-exec.so` - exec interception
   - `libvgpu-cuda.so` - CUDA Driver API
   - `libvgpu-nvml.so` - NVML API
   - `libvgpu-cudart.so` - CUDA Runtime API

2. **Library loading order issue**
   - Maybe the order is wrong
   - Maybe a dependency is missing

3. **Constructor initialization issue**
   - One of the shim constructors might be crashing
   - Maybe initialization order is wrong

4. **Symbol conflict**
   - Maybe there's a symbol conflict between shims
   - Maybe a symbol is being resolved incorrectly

### Next Steps to Investigate

1. **Test without LD_PRELOAD:**
   ```bash
   # Temporarily comment out LD_PRELOAD
   sudo sed -i 's/^Environment="LD_PRELOAD=/##Environment="LD_PRELOAD=/' /etc/systemd/system/ollama.service.d/vgpu.conf
   sudo systemctl daemon-reload
   sudo systemctl restart ollama
   ```
   If Ollama starts without LD_PRELOAD, then one of the shims is causing the crash.

2. **Test with individual shims:**
   - Test with just `libvgpu-exec.so`
   - Test with `libvgpu-exec.so:libvgpu-cuda.so`
   - Test with `libvgpu-exec.so:libvgpu-cuda.so:libvgpu-nvml.so`
   - Add one at a time to find which one causes the crash

3. **Check stderr log:**
   ```bash
   sudo tail -100 /tmp/ollama_stderr.log
   ```
   May contain more details about the crash.

4. **Check core dump:**
   ```bash
   sudo gdb /usr/local/bin/ollama /var/lib/systemd/coredump/core.ollama.*
   ```
   May show where the crash occurs.

### Alternative Approach

If one of the shims is causing the crash, we might need to:
1. Check the shim code for issues
2. Review constructor initialization
3. Check for symbol conflicts
4. Verify library dependencies

### Documentation to Review

- Check if there were previous issues with shim crashes
- Review constructor priorities
- Check if there are known compatibility issues
