# Root Cause: Runner Subprocesses Not Using Our Shims

## Critical Finding

**`initial_count=0`** - Runner subprocesses are not detecting the GPU because they don't have our shims loaded.

## Why This Happens

1. **Runner subprocesses are spawned separately**:
   - Main Ollama process spawns runner subprocesses for GPU discovery
   - These subprocesses run `ollama runner` commands
   - They have their own environment and library paths

2. **Runner subprocesses use different library paths**:
   - From logs: `LD_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12`
   - They look for libraries in `/usr/local/lib/ollama/cuda_v12/` first
   - This directory contains real CUDA libraries, not our shims

3. **Exec interception is not working**:
   - `libvgpu-exec.so` should inject LD_PRELOAD into subprocesses
   - But no exec interception logs are appearing
   - Runner subprocesses don't inherit our shims

## Solution

We need to ensure runner subprocesses use our shims. Options:

### Option 1: Replace/Symlink Libraries in Ollama Directory (Recommended)

Replace or symlink the CUDA libraries in `/usr/local/lib/ollama/cuda_v12/` to point to our shims:

```bash
# Backup originals
sudo mv /usr/local/lib/ollama/cuda_v12/libcuda.so.1 /usr/local/lib/ollama/cuda_v12/libcuda.so.1.backup
sudo mv /usr/local/lib/ollama/cuda_v12/libcudart.so.12 /usr/local/lib/ollama/cuda_v12/libcudart.so.12.backup

# Create symlinks to our shims
sudo ln -sf /usr/lib64/libvgpu-cuda.so /usr/local/lib/ollama/cuda_v12/libcuda.so.1
sudo ln -sf /usr/lib64/libvgpu-cudart.so /usr/local/lib/ollama/cuda_v12/libcudart.so.12
```

### Option 2: Fix Exec Interception

Ensure `libvgpu-exec.so` is properly intercepting exec calls and injecting LD_PRELOAD:

1. Verify `libvgpu-exec.so` is first in LD_PRELOAD
2. Check if exec interception is working
3. Verify LD_PRELOAD is being injected into subprocesses

### Option 3: Use System-Wide Library Paths

Ensure our shims are in system-wide library paths that runner subprocesses will find:

1. Add `/usr/lib64` to system-wide library paths
2. Ensure symlinks are in place
3. Verify runner subprocesses can find our shims

## Immediate Action

**Replace/symlink libraries in Ollama directory** - This is the fastest and most reliable solution since runner subprocesses look there first.
