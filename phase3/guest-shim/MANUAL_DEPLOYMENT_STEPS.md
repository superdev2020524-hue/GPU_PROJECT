# Manual Deployment Steps for dlsym Interception

## Quick Deployment Commands

Run these commands on the VM (test-10@10.25.33.110):

```bash
# 1. SSH to VM
ssh test-10@10.25.33.110

# 2. Navigate to shim directory
cd ~/phase3/guest-shim

# 3. Build shims (will prompt for sudo password: Calvin@123)
sudo ./install.sh

# 4. Verify dlsym symbol is present
nm -D libvgpu-cuda.so | grep " dlsym" | head -3

# 5. Restart Ollama
sudo systemctl restart ollama

# 6. Wait for discovery
sleep 8

# 7. Check for dlsym interception messages
sudo journalctl -u ollama -n 300 --no-pager | grep -E "(dlsym|libggml-cuda)" | head -20

# 8. Check compute capability
sudo journalctl -u ollama -n 300 --no-pager | grep -i "compute" | head -10

# 9. Check GPU detection status
sudo journalctl -u ollama -n 300 --no-pager | grep -iE "(gpu|device|didn't fully)" | tail -10
```

## Expected Results

### Success Indicators:

1. **dlsym interception working:**
   ```
   [libvgpu-cuda] dlopen("...libggml-cuda.so") - libggml-cuda.so loading
   [libvgpu-cuda] dlsym(handle=0x..., "cuDeviceGetAttribute") called (pid=...)
   [libvgpu-cuda] dlsym() REDIRECTED "cuDeviceGetAttribute" to shim at 0x... (pid=...)
   ```

2. **Compute capability correct:**
   ```
   compute=9.0
   ```
   (NOT `compute=0.0`)

3. **Device not filtered:**
   - Should NOT see: `"didn't fully initialize"`
   - Should see: GPU detected and used

## Troubleshooting

### If dlsym symbol not found:
```bash
# Check if build succeeded
ls -lh libvgpu-cuda.so

# Check build errors
sudo ./install.sh 2>&1 | grep -i error
```

### If no dlsym messages in logs:
```bash
# Check if libggml-cuda.so is loading
sudo journalctl -u ollama -n 500 | grep "libggml-cuda"

# Check if shim is loaded
sudo cat /proc/$(pgrep ollama)/maps | grep libvgpu-cuda
```

### If compute still 0.0:
```bash
# Check if CUDA functions are being looked up
sudo journalctl -u ollama -n 500 | grep -i "cuDevice\|cudaDevice"

# Check if functions are being redirected
sudo journalctl -u ollama -n 500 | grep "REDIRECTED"

# Check if our shim functions are being called
sudo journalctl -u ollama -n 500 | grep "cuDeviceGetAttribute.*CALLED"
```

## Next Steps After Deployment

1. **If dlsym interception works:**
   - Verify compute=9.0
   - Test GPU mode with a model
   - Document the solution

2. **If dlsym interception doesn't work:**
   - Check bootstrap logic
   - Verify RTLD_NEXT behavior
   - Consider alternative approaches

3. **If compute is still 0.0:**
   - Check which functions are being looked up
   - Verify our shim functions return correct values
   - Check if there are other ways Ollama gets compute capability
