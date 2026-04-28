# Crash Diagnosis and Next Steps

## Date: 2026-02-26

## Current Status

**Ollama is still crashing with SEGV even after removing `libvgpu-syscall.so` from `LD_PRELOAD`.**

### What We've Fixed

1. ✅ `libvgpu-syscall.so` removed from `LD_PRELOAD` (file doesn't exist)
2. ✅ `OLLAMA_LIBRARY_PATH` added correctly
3. ✅ `OLLAMA_LLM_LIBRARY=cuda_v12` is present (required)
4. ✅ `OLLAMA_NUM_GPU=999` is present
5. ✅ All shim libraries exist: `exec`, `cuda`, `nvml`, `cudart`
6. ✅ No systemd syntax errors

### Current LD_PRELOAD

```
LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so
```

## Diagnosis Needed

**One of the shim libraries is likely causing the crash.**

### Test Plan

#### Step 1: Test Without LD_PRELOAD

```bash
# Temporarily disable LD_PRELOAD
sudo sed -i 's/^Environment="LD_PRELOAD=/##Environment="LD_PRELOAD=/' /etc/systemd/system/ollama.service.d/vgpu.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
sleep 8
systemctl is-active ollama
```

**Expected Result:**
- If Ollama starts → **One of the shims is causing the crash**
- If Ollama still crashes → **Issue is NOT from the shims**

#### Step 2: Test With Individual Shims (if Step 1 shows shim issue)

Test with one shim at a time to identify which one causes the crash:

```bash
# Test 1: Just exec
sudo sed -i 's|LD_PRELOAD=.*|LD_PRELOAD=/usr/lib64/libvgpu-exec.so"|g' /etc/systemd/system/ollama.service.d/vgpu.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
sleep 8 && systemctl is-active ollama

# Test 2: exec + cuda
sudo sed -i 's|LD_PRELOAD=.*|LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cuda.so"|g' /etc/systemd/system/ollama.service.d/vgpu.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
sleep 8 && systemctl is-active ollama

# Test 3: exec + cuda + nvml
sudo sed -i 's|LD_PRELOAD=.*|LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so"|g' /etc/systemd/system/ollama.service.d/vgpu.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
sleep 8 && systemctl is-active ollama

# Test 4: All shims
sudo sed -i 's|LD_PRELOAD=.*|LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"|g' /etc/systemd/system/ollama.service.d/vgpu.conf
sudo systemctl daemon-reload && sudo systemctl restart ollama
sleep 8 && systemctl is-active ollama
```

**This will identify which shim causes the crash.**

## Possible Causes

1. **Constructor initialization issue**
   - One of the shim constructors might be crashing
   - Check constructor priorities and initialization order

2. **Symbol conflict**
   - Symbol conflict between shims
   - Incorrect symbol resolution

3. **Library dependency issue**
   - Missing dependency
   - Incorrect library version

4. **Order issue**
   - Wrong loading order
   - Dependency not available when needed

## Documentation to Review

Based on previous documentation:
- `CONSTRUCTOR_FIX_ATTEMPT.md` - Constructor issues were resolved
- `CONSTRUCTOR_WORKING.md` - Constructors were working before
- `LD_PRELOAD_ORDER_FIXED.md` - Order was fixed before

**Question:** Did something change in the shim code that could cause crashes?

## Next Steps

1. **Test without LD_PRELOAD** to confirm if shims are the issue
2. **If shims are the issue**, test with individual shims to find the culprit
3. **If not shims**, investigate other causes (OLLAMA_LIBRARY_PATH format, etc.)
4. **Check stderr log** for more details: `sudo tail -100 /tmp/ollama_stderr.log`
5. **Check core dump** if available: `sudo ls -la /var/lib/systemd/coredump/`

## Summary

- ✅ Fixes applied: `libvgpu-syscall.so` removed, `OLLAMA_LIBRARY_PATH` added
- ❌ Ollama still crashing
- ⏳ Need to test if shims are causing the crash
- ⏳ Need to identify which shim (if any) is the problem
