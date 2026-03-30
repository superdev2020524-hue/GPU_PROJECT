# Final Status and Findings

## Date: 2026-02-26

## Fixes Applied

### ✅ Completed Fixes

1. **`libvgpu-syscall.so` removed from LD_PRELOAD**
   - File doesn't exist, was causing errors
   - Removed successfully

2. **`OLLAMA_LIBRARY_PATH` added**
   - Set to: `/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12`
   - Tells scanner where to find backend libraries

3. **`force_load_shim` wrapper removed from ExecStart**
   - Service file updated: `ExecStart=/usr/local/bin/ollama serve`
   - No longer using wrapper

### Current LD_PRELOAD

```
LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cudart.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so
```

**Note:** Order is: exec, cuda, nvml, cudart

## Current Issue

**Ollama is still crashing with SEGV even after all fixes.**

### What We Know

1. ✅ Service file is correct (no `force_load_shim`)
2. ✅ `OLLAMA_LIBRARY_PATH` is set
3. ✅ `libvgpu-syscall.so` removed from LD_PRELOAD
4. ✅ All shim libraries exist
5. ❌ **Ollama still crashes**

### Possible Causes

1. **One of the shim libraries is causing the crash**
   - `libvgpu-exec.so` - exec interception
   - `libvgpu-cuda.so` - CUDA Driver API
   - `libvgpu-nvml.so` - NVML API
   - `libvgpu-cudart.so` - CUDA Runtime API

2. **Constructor initialization issue**
   - One of the shim constructors might be crashing
   - Initialization order might be wrong

3. **Symbol conflict**
   - Symbol conflict between shims
   - Incorrect symbol resolution

4. **Library dependency issue**
   - Missing dependency
   - Incorrect library version

## Next Steps to Diagnose

### Step 1: Test Without LD_PRELOAD

```bash
# Temporarily disable LD_PRELOAD
sudo sed -i 's/^Environment="LD_PRELOAD=/##Environment="LD_PRELOAD=/' /etc/systemd/system/ollama.service.d/vgpu.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 10
systemctl is-active ollama
```

**If Ollama starts without LD_PRELOAD:**
- One of the shims is causing the crash
- Proceed to Step 2

**If Ollama still crashes:**
- Issue is NOT from shims
- Need to investigate other causes

### Step 2: Test With Individual Shims (if Step 1 shows shim issue)

Test with one shim at a time:

```bash
# Test 1: Just exec
sudo sed -i 's|LD_PRELOAD=.*|LD_PRELOAD=/usr/lib64/libvgpu-exec.so"|g' /etc/systemd/system/ollama.service.d/vgpu.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
sleep 10 && systemctl is-active ollama

# Test 2: exec + cuda
sudo sed -i 's|LD_PRELOAD=.*|LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cuda.so"|g' /etc/systemd/system/ollama.service.d/vgpu.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
sleep 10 && systemctl is-active ollama

# Test 3: exec + cuda + nvml
sudo sed -i 's|LD_PRELOAD=.*|LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so"|g' /etc/systemd/system/ollama.service.d/vgpu.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
sleep 10 && systemctl is-active ollama

# Test 4: All shims
sudo sed -i 's|LD_PRELOAD=.*|LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"|g' /etc/systemd/system/ollama.service.d/vgpu.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
sleep 10 && systemctl is-active ollama
```

This will identify which shim causes the crash.

### Step 3: Check Stderr Log

```bash
sudo tail -100 /tmp/ollama_stderr.log
```

May contain more details about the crash.

## Summary

**All identified fixes have been applied:**
- ✅ `libvgpu-syscall.so` removed
- ✅ `OLLAMA_LIBRARY_PATH` added
- ✅ `force_load_shim` wrapper removed

**But Ollama still crashes.**

**Next step:** Test if LD_PRELOAD is causing the crash (disable temporarily) to determine if one of the shims is the problem.
