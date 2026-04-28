# Manual Fix Commands - Apply These on the VM

## Date: 2026-02-26

## Critical Issues Found

1. **`libvgpu-syscall.so` is in `LD_PRELOAD` but file doesn't exist** → causing crashes
2. **`OLLAMA_LIBRARY_PATH` is missing** → scanner can't find `cuda_v12/`

## Complete Fix Commands

Run these commands **on the VM** (test-10@10.25.33.110):

```bash
# Connect to VM first
ssh test-10@10.25.33.110
# Password: Calvin@123

# Fix 1: Remove libvgpu-syscall.so from LD_PRELOAD
sudo sed -i 's|:/usr/lib64/libvgpu-syscall.so||g' /etc/systemd/system/ollama.service.d/vgpu.conf
sudo sed -i 's|libvgpu-syscall.so:||g' /etc/systemd/system/ollama.service.d/vgpu.conf

# Fix 2: Fix LD_PRELOAD order (should be: exec, cuda, nvml, cudart)
sudo sed -i 's|LD_PRELOAD=.*|LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"|g' /etc/systemd/system/ollama.service.d/vgpu.conf

# Fix 3: Add OLLAMA_LIBRARY_PATH if not present
if ! grep -q "OLLAMA_LIBRARY_PATH" /etc/systemd/system/ollama.service.d/vgpu.conf; then
    sudo bash -c 'echo "" >> /etc/systemd/system/ollama.service.d/vgpu.conf'
    sudo bash -c 'echo "# OLLAMA_LIBRARY_PATH" >> /etc/systemd/system/ollama.service.d/vgpu.conf'
    sudo bash -c 'echo "Environment=\"OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12\"" >> /etc/systemd/system/ollama.service.d/vgpu.conf'
fi

# Verify fixes
echo "=== LD_PRELOAD ==="
sudo grep LD_PRELOAD /etc/systemd/system/ollama.service.d/vgpu.conf
echo ""
echo "=== OLLAMA_LIBRARY_PATH ==="
sudo grep OLLAMA_LIBRARY_PATH /etc/systemd/system/ollama.service.d/vgpu.conf

# Restart Ollama
sudo systemctl daemon-reload
sudo systemctl restart ollama

# Wait and check status
sleep 8
systemctl is-active ollama

# Wait for discovery and check logs
sleep 12
journalctl -u ollama --since "20 seconds ago" --no-pager | grep -E "verifying|library=|initial_count|OLLAMA_LIBRARY_PATH" | tail -10
```

## Expected Results

After applying fixes:

1. ✅ `LD_PRELOAD` should NOT contain `libvgpu-syscall.so`
2. ✅ `OLLAMA_LIBRARY_PATH` should be set with quotes
3. ✅ Ollama should start without crashing (`systemctl is-active ollama` shows "active")
4. ✅ Discovery logs should show:
   - `OLLAMA_LIBRARY_PATH` in logs
   - "verifying if device is supported"
   - `library=cuda_v12` or `library=cuda`
   - `initial_count=1`

## Verification Checklist

- [ ] `libvgpu-syscall.so` removed from `LD_PRELOAD`
- [ ] `LD_PRELOAD` order is: `exec:cuda:nvml:cudart`
- [ ] `OLLAMA_LIBRARY_PATH` is set (one line with quotes)
- [ ] Ollama starts without crashing
- [ ] Discovery logs show GPU detection

## If Ollama Still Crashes

If Ollama still crashes after these fixes:

1. Check systemd syntax:
   ```bash
   sudo systemd-analyze verify ollama.service
   ```

2. Check for other issues in vgpu.conf:
   ```bash
   sudo cat /etc/systemd/system/ollama.service.d/vgpu.conf
   ```

3. Check if all shim libraries exist:
   ```bash
   ls -la /usr/lib64/libvgpu-*.so
   ```
   Should show: `exec`, `cuda`, `nvml`, `cudart` (NOT `syscall`)

## Summary

**Root Cause:** `libvgpu-syscall.so` in `LD_PRELOAD` but file doesn't exist → crash

**Fix:** Remove it from `LD_PRELOAD` and add `OLLAMA_LIBRARY_PATH`

**Expected:** Ollama starts, scanner finds `cuda_v12/`, GPU detected
