# Critical Fix: libvgpu-syscall.so Causing Crash

## Date: 2026-02-26

## Critical Issue

**Ollama is crashing because `libvgpu-syscall.so` is in `LD_PRELOAD` but the file doesn't exist!**

### Root Cause

The file `/usr/lib64/libvgpu-syscall.so` does not exist, but it's listed in `LD_PRELOAD`:
```
LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-syscall.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so
```

When systemd tries to start Ollama with this `LD_PRELOAD`, it fails because the file cannot be loaded, causing a crash.

### Evidence

1. **File doesn't exist:**
   ```bash
   ls -la /usr/lib64/libvgpu-syscall.so
   # Result: No such file or directory
   ```

2. **Ollama crashes:**
   ```
   Feb 26 05:54:39 ollama.service: Main process exited, code=dumped, status=11/SEGV
   Feb 26 05:54:39 ollama.service: Failed with result 'core-dump'.
   ```

3. **According to documentation:**
   - `ROOT_CAUSE_ANALYSIS.md` states: "libvgpu-syscall.so was listed in LD_PRELOAD but the file doesn't exist"
   - Fix was to remove it from `LD_PRELOAD`

## The Fix

### Step 1: Remove libvgpu-syscall.so from LD_PRELOAD

```bash
sudo sed -i 's|:/usr/lib64/libvgpu-syscall.so||g' /etc/systemd/system/ollama.service.d/vgpu.conf
sudo sed -i 's|libvgpu-syscall.so:||g' /etc/systemd/system/ollama.service.d/vgpu.conf
```

### Step 2: Fix LD_PRELOAD Order

The correct order should be:
```
libvgpu-exec.so:libvgpu-cuda.so:libvgpu-nvml.so:libvgpu-cudart.so
```

```bash
sudo sed -i 's|LD_PRELOAD=.*|LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"|g' /etc/systemd/system/ollama.service.d/vgpu.conf
```

### Step 3: Add OLLAMA_LIBRARY_PATH (if not present)

```bash
if ! grep -q "OLLAMA_LIBRARY_PATH" /etc/systemd/system/ollama.service.d/vgpu.conf; then
    sudo bash -c 'echo "" >> /etc/systemd/system/ollama.service.d/vgpu.conf'
    sudo bash -c 'echo "# OLLAMA_LIBRARY_PATH" >> /etc/systemd/system/ollama.service.d/vgpu.conf'
    sudo bash -c 'echo "Environment=\"OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12\"" >> /etc/systemd/system/ollama.service.d/vgpu.conf'
fi
```

### Step 4: Restart Ollama

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

## Verification

1. **Check LD_PRELOAD doesn't contain libvgpu-syscall.so:**
   ```bash
   sudo grep LD_PRELOAD /etc/systemd/system/ollama.service.d/vgpu.conf
   ```
   Should NOT contain `libvgpu-syscall.so`

2. **Check OLLAMA_LIBRARY_PATH is set:**
   ```bash
   sudo grep OLLAMA_LIBRARY_PATH /etc/systemd/system/ollama.service.d/vgpu.conf
   ```
   Should show one line with quotes

3. **Check Ollama starts:**
   ```bash
   systemctl is-active ollama
   ```
   Should show "active" (not crashing)

4. **Check discovery:**
   ```bash
   sleep 12
   journalctl -u ollama --since "15 seconds ago" | grep -E "verifying|library=|initial_count|OLLAMA_LIBRARY_PATH"
   ```

## Expected Result

After the fix:

1. ✅ Ollama starts without crashing
2. ✅ `LD_PRELOAD` doesn't contain non-existent library
3. ✅ `OLLAMA_LIBRARY_PATH` is set correctly
4. ✅ Scanner can find `cuda_v12/` directory
5. ✅ Library loads and GPU is detected

## Why This Happened

According to `ROOT_CAUSE_ANALYSIS.md`, `libvgpu-syscall.so` was supposed to be removed from `LD_PRELOAD` because it doesn't exist. However, it appears it was added back or never fully removed.

## Current Status

- ⚠ `libvgpu-syscall.so` is still in `LD_PRELOAD` (causing crashes)
- ⚠ `OLLAMA_LIBRARY_PATH` was removed during testing (needs to be added back)
- ⏳ Need to apply both fixes

## Complete Fix Command

```bash
# Remove libvgpu-syscall.so
sudo sed -i 's|:/usr/lib64/libvgpu-syscall.so||g' /etc/systemd/system/ollama.service.d/vgpu.conf
sudo sed -i 's|libvgpu-syscall.so:||g' /etc/systemd/system/ollama.service.d/vgpu.conf

# Fix LD_PRELOAD order
sudo sed -i 's|LD_PRELOAD=.*|LD_PRELOAD=/usr/lib64/libvgpu-exec.so:/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"|g' /etc/systemd/system/ollama.service.d/vgpu.conf

# Add OLLAMA_LIBRARY_PATH
if ! grep -q "OLLAMA_LIBRARY_PATH" /etc/systemd/system/ollama.service.d/vgpu.conf; then
    sudo bash -c 'echo "" >> /etc/systemd/system/ollama.service.d/vgpu.conf'
    sudo bash -c 'echo "# OLLAMA_LIBRARY_PATH" >> /etc/systemd/system/ollama.service.d/vgpu.conf'
    sudo bash -c 'echo "Environment=\"OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12\"" >> /etc/systemd/system/ollama.service.d/vgpu.conf'
fi

# Restart
sudo systemctl daemon-reload
sudo systemctl restart ollama
```
