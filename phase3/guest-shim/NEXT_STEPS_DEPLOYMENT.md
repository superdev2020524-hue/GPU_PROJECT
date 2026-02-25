# Next Steps: Deploy and Test dlsym Interception

## Implementation Status
✅ **dlsym interception implemented** in `libvgpu_cuda.c`
✅ **dlopen logging enhanced** for libggml-cuda.so
✅ **Build requirements verified** (-ldl already included)

## Deployment Steps

### 1. Build on VM
```bash
# SSH to VM
ssh test-10@10.25.33.110

# Navigate to shim directory
cd ~/phase3/guest-shim

# Build shims (this will rebuild libvgpu-cuda.so with dlsym interception)
sudo ./install.sh
```

### 2. Verify Build
```bash
# Check if libvgpu-cuda.so was rebuilt
ls -lh libvgpu-cuda.so

# Verify dlsym symbol is present
nm -D libvgpu-cuda.so | grep dlsym
```

### 3. Restart Ollama
```bash
# Restart Ollama service to load new shim
sudo systemctl restart ollama

# Wait a few seconds for discovery
sleep 5

# Check Ollama status
sudo systemctl status ollama
```

### 4. Check Logs for dlsym Interception

#### Check for dlsym interception messages:
```bash
# Check journalctl for dlsym messages
sudo journalctl -u ollama -n 100 | grep -i "dlsym\|libggml-cuda"

# Check stderr log if configured
tail -100 /tmp/ollama_stderr.log | grep -i "dlsym\|libggml-cuda"
```

#### Expected log messages:
1. **libggml-cuda.so loading:**
   ```
   [libvgpu-cuda] dlopen("...libggml-cuda.so") - libggml-cuda.so loading
   [libvgpu-cuda]   This library may use dlsym() to resolve CUDA functions
   [libvgpu-cuda]   Our dlsym() interceptor will catch those lookups
   ```

2. **CUDA function lookups:**
   ```
   [libvgpu-cuda] dlsym(handle=0x..., "cuDeviceGetAttribute") called (pid=...)
   ```

3. **Function redirection:**
   ```
   [libvgpu-cuda] dlsym() REDIRECTED "cuDeviceGetAttribute" to shim at 0x... (pid=...)
   ```

### 5. Verify Compute Capability

#### Check Ollama logs for compute value:
```bash
# Check for compute capability in logs
sudo journalctl -u ollama -n 200 | grep -i "compute"

# Expected: compute=9.0 (instead of compute=0.0)
```

#### Check if device is filtered:
```bash
# Check for "didn't fully initialize" message
sudo journalctl -u ollama -n 200 | grep -i "didn't fully initialize"

# Should NOT appear if compute capability is correct
```

### 6. Test GPU Mode

#### Trigger discovery:
```bash
# Run a simple Ollama command to trigger discovery
ollama list

# Or check GPU info
curl http://localhost:11434/api/ps
```

#### Verify GPU is detected:
```bash
# Check Ollama logs for GPU detection
sudo journalctl -u ollama -n 200 | grep -i "gpu\|cuda\|device"

# Should show GPU detected with compute=9.0
```

## Troubleshooting

### If dlsym interception doesn't work:

1. **Check if dlsym is being called:**
   ```bash
   # Look for dlsym log messages
   sudo journalctl -u ollama -n 500 | grep "dlsym"
   ```

2. **Check if libggml-cuda.so is loading:**
   ```bash
   # Look for libggml-cuda.so load messages
   sudo journalctl -u ollama -n 500 | grep "libggml-cuda"
   ```

3. **Verify shim is loaded:**
   ```bash
   # Check if our shim is in process memory
   sudo cat /proc/$(pgrep ollama)/maps | grep libvgpu-cuda
   ```

4. **Check bootstrap issues:**
   - If dlsym interception fails during bootstrap, it will return NULL
   - This is expected for the first call - subsequent calls should work
   - If all calls fail, there may be a bootstrap issue

### If compute capability is still 0.0:

1. **Check if CUDA functions are being looked up:**
   ```bash
   # Look for cuDeviceGetAttribute or similar
   sudo journalctl -u ollama -n 500 | grep -i "cuDevice\|cudaDevice"
   ```

2. **Check if functions are being redirected:**
   ```bash
   # Look for REDIRECTED messages
   sudo journalctl -u ollama -n 500 | grep "REDIRECTED"
   ```

3. **Verify our shim functions are being called:**
   ```bash
   # Look for cuDeviceGetAttribute CALLED messages
   sudo journalctl -u ollama -n 500 | grep "cuDeviceGetAttribute.*CALLED"
   ```

## Success Criteria

✅ **dlsym interception logs show CUDA function lookups**
✅ **Function redirection works (our shims are called)**
✅ **`compute=9.0` appears in Ollama logs** (instead of `compute=0.0`)
✅ **Device is not filtered** as "didn't fully initialize"
✅ **Ollama uses GPU mode** instead of CPU mode

## Next Actions After Testing

1. **If dlsym interception works but compute is still 0.0:**
   - Check if the functions being looked up are the right ones
   - Verify our shim functions return correct values
   - Check if there are other ways Ollama gets compute capability

2. **If dlsym interception doesn't work:**
   - Check bootstrap logic
   - Verify RTLD_NEXT behavior
   - Consider alternative approaches (LD_AUDIT, etc.)

3. **If everything works:**
   - Document the solution
   - Create a summary of what fixed the issue
   - Update deployment guide
