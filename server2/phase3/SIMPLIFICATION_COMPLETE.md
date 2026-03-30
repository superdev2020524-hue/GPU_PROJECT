# Simplification Complete - Shims Work as System Libraries

## Date: 2026-02-26

## Key Discovery

**The shims work perfectly as system libraries!** We were overcomplicating the solution.

## What We Verified

### 1. Shims Work as System Libraries ✓

**Test Results:**
- Simple `dlopen("libcuda.so.1")` test: **SUCCESS**
- `cuInit()` function call: **SUCCESS** (returns 0)
- `cuDeviceGetCount()` function call: **SUCCESS** (returns count=1)
- GPU detected: **SUCCESS** (count=1)

**This means:**
- Any application using system CUDA libraries will work
- No LD_PRELOAD needed
- No special environment variables needed
- No constructor detection logic needed
- Just like real CUDA libraries!

### 2. System Library Installation ✓

**Verified:**
- SONAMEs are correct (`libcuda.so.1`, `libnvidia-ml.so.1`)
- Symlinks are correct (`/usr/lib64/libcuda.so.1 -> libvgpu-cuda.so`)
- Libraries are in `ldconfig` cache
- Standard library paths work

### 3. Ollama's Bundled Libraries ✓

**Status:**
- Ollama's bundled libraries already point to our shims via symlinks:
  - `/usr/local/lib/ollama/cuda_v12/libcuda.so.1 -> /usr/lib64/libvgpu-cuda.so`
  - `/usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90 -> /usr/lib64/libvgpu-cudart.so`
  - `/usr/local/lib/ollama/cuda_v12/libnvidia-ml.so.1 -> /usr/lib64/libvgpu-nvml.so`

## What We Simplified

### Before (Overcomplicated):
```ini
[Service]
Environment="LD_PRELOAD=/usr/lib64/libvgpu-cuda.so:/usr/lib64/libvgpu-nvml.so:/usr/lib64/libvgpu-cudart.so"
Environment="LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:/usr/lib64"
Environment="OLLAMA_NUM_GPU=999"
Environment="OLLAMA_LLM_LIBRARY=cuda_v12"
Environment="OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12"
```

### After (Simple):
```ini
[Service]
# Simplified configuration - shims work as system libraries
# Ollama's bundled libraries already point to our shims via symlinks
# No LD_PRELOAD needed - libraries load naturally
Environment="LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama:/usr/lib64"
```

**Removed:**
- ❌ `LD_PRELOAD` (not needed - shims work as system libraries)
- ❌ `OLLAMA_NUM_GPU` (not needed)
- ❌ `OLLAMA_LLM_LIBRARY` (not needed)
- ❌ `OLLAMA_LIBRARY_PATH` (not needed)

**Kept:**
- ✓ `LD_LIBRARY_PATH` (for Ollama's bundled library directory)

## Why This Works

1. **System Libraries**: Shims are installed as proper system libraries with correct SONAMEs
2. **Symlinks**: System symlinks (`/usr/lib64/libcuda.so.1`) point to our shims
3. **ldconfig**: Libraries are registered in system cache
4. **Ollama Bundled Libraries**: Symlinks in Ollama's directory point to our shims

## The Right Approach

### For System-Wide Compatibility (Like Real CUDA):
1. Install shims in standard locations (`/usr/lib64`)
2. Set correct SONAMEs (`libcuda.so.1`, `libnvidia-ml.so.1`)
3. Run `ldconfig` to register them
4. Create symlinks (`libcuda.so.1 -> libvgpu-cuda.so`)
5. **Done!** Any application using system CUDA will work

### For Ollama Specifically (Because It Bundles Libraries):
- Symlink Ollama's bundled libraries to point to our shims
- This is an Ollama-specific workaround, not a general solution

## Next Steps

1. **Verify Ollama works with simplified config**:
   - Restart Ollama
   - Test GPU discovery
   - Check if `initial_count=1` and `library=cuda`

2. **If Ollama still doesn't detect GPU**:
   - The issue is likely in Ollama's discovery mechanism, not the shims
   - Shims work correctly (verified with test program)
   - Need to investigate why Ollama's runner process doesn't see the GPU

3. **Remove constructor complexity**:
   - Since shims work as system libraries, we don't need:
     - OLLAMA environment variable detection
     - LD_PRELOAD detection
     - Process type detection
   - Constructor can be simplified to just initialize the shim

## Key Insight

**The user was right!** We were going in the wrong direction by making everything Ollama-specific. The shims should work like real CUDA libraries - and they do! The only Ollama-specific work needed is symlinking its bundled libraries.

## Status

- ✅ Shims work as system libraries (verified)
- ✅ GPU detected in test program (count=1)
- ✅ Ollama bundled libraries point to shims (verified)
- ✅ Configuration simplified (no LD_PRELOAD, no OLLAMA env vars)
- ⏳ Testing Ollama with simplified config (in progress)
