# Manual Fix Required - Configuration Issues

## Date: 2026-02-26

## Current Issues

### 1. LD_PRELOAD Has Triple Path
Current:
```
Environment="LD_PRELOAD=/usr/lib64//usr/lib64//usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"
```

Should be:
```
Environment="LD_PRELOAD=/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"
```

### 2. Problematic Libraries Still Present
- `libvgpu-exec.so` may still be in LD_PRELOAD (should be removed)
- `libvgpu-syscall.so` may still be in LD_PRELOAD (should be removed)

### 3. OLLAMA_LIBRARY_PATH Missing
The environment variable `OLLAMA_LIBRARY_PATH` needs to be added to `vgpu.conf`.

## Manual Fix Commands

Execute these commands on the VM:

```bash
# 1. Fix triple path in LD_PRELOAD
sudo sed -i "s|/usr/lib64//usr/lib64//usr/lib64/|/usr/lib64/|g" /etc/systemd/system/ollama.service.d/vgpu.conf

# 2. Remove problematic libraries
sudo sed -i "s|libvgpu-exec.so:||g" /etc/systemd/system/ollama.service.d/vgpu.conf
sudo sed -i "s|libvgpu-syscall.so:||g" /etc/systemd/system/ollama.service.d/vgpu.conf

# 3. Add OLLAMA_LIBRARY_PATH if missing
grep -q "OLLAMA_LIBRARY_PATH" /etc/systemd/system/ollama.service.d/vgpu.conf || \
  echo 'Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"' | \
  sudo tee -a /etc/systemd/system/ollama.service.d/vgpu.conf

# 4. Verify configuration
grep -E "LD_PRELOAD|OLLAMA_LIBRARY_PATH" /etc/systemd/system/ollama.service.d/vgpu.conf

# 5. Reload and restart
sudo systemctl daemon-reload
sudo systemctl restart ollama

# 6. Wait for Ollama to start
sleep 5
systemctl is-active ollama

# 7. Verify environment
PID=$(pgrep -f "ollama serve")
sudo cat /proc/$PID/environ | tr "\0" "\n" | grep -E "OLLAMA_LIBRARY_PATH|LD_PRELOAD"

# 8. Test GPU detection
curl -s http://localhost:11434/api/generate -d '{"model":"llama3.2:1b","prompt":"test","stream":false,"options":{"num_predict":5}}' > /dev/null 2>&1 &

# 9. Check discovery logs
sleep 20
journalctl -u ollama --since "25 seconds ago" --no-pager | grep -E "bootstrap discovery|initial_count|library=" | tail -8
```

## Expected Results

After fixing:
1. `LD_PRELOAD` should only contain: `libvgpu-cuda.so:libvgpu-nvml.so:libvgpu-cudart.so`
2. `OLLAMA_LIBRARY_PATH` should be set to: `/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12`
3. Main process environment should have both variables
4. Discovery should show `initial_count=1` and `library=cuda`

## Current Status

- ✅ Shim libraries: Working correctly
- ✅ GPU device: Detected by shim (device count = 1)
- ✅ Constructor fix: Deployed in code
- ⚠️ Configuration: Needs manual fix (SSH automation timing out)
- ❌ Discovery: Still showing `initial_count=0` and `library=cpu` (due to config issues)

## Next Steps

1. Manually execute the fix commands above
2. Verify the configuration is correct
3. Restart Ollama
4. Test GPU detection
5. Check discovery logs for `initial_count=1` and `library=cuda`
