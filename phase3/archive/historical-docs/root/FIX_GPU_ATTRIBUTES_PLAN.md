# Fix GPU Attributes Plan - General SHIM Fix

## Problem Identified

According to ChatGPT analysis:
- `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK` is returning `1620000` instead of `1024`
- This causes Ollama (GGML) to reject the GPU as invalid
- Other frameworks may work because they don't validate this strictly
- **This is a general SHIM bug, not Ollama-specific**

## Current Local Code Status

✅ **Local code is CORRECT:**
- `gpu_properties.h` line 44: `#define GPU_DEFAULT_MAX_THREADS_PER_BLOCK   1024`
- `libvgpu_cuda.c` line 2901: `g_gpu_info.max_threads_per_block = GPU_DEFAULT_MAX_THREADS_PER_BLOCK;`
- `libvgpu_cuda.c` line 3611: `*pi = g_gpu_info.max_threads_per_block;`

## Action Plan

### Step 1: Connect to VM and Check Actual Code
```bash
# On VM (test-10@10.25.33.110):
# Find shim code location
find ~ -name "libvgpu_cuda.c" -type f
find /home -name "libvgpu_cuda.c" -type f
find /opt -name "libvgpu_cuda.c" -type f

# Check installed library
ls -la /usr/lib64/libvgpu-cuda.so
```

### Step 2: Compare VM Code with Local Code
- Check if VM has `gpu_properties.h` with correct value (1024)
- Check if VM code matches local code
- **VM is source of truth** - if different, sync local to match VM first

### Step 3: Verify the Problem
```bash
# On VM, check if library contains invalid value
strings /usr/lib64/libvgpu-cuda.so | grep -E "1620000|1024" | head -5

# Check gpu_properties.h on VM
grep "MAX_THREADS_PER_BLOCK" <path_to_gpu_properties.h>
```

### Step 4: Fix in Local Code (if needed)
If VM code is different, sync local to match VM first, then:
1. Ensure `gpu_properties.h` has: `#define GPU_DEFAULT_MAX_THREADS_PER_BLOCK   1024`
2. Verify `libvgpu_cuda.c` uses this constant correctly
3. Check all other GPU attributes are valid

### Step 5: Deploy Fixed Code to VM
```bash
# Copy fixed files to VM
scp phase3/guest-shim/libvgpu_cuda.c test-10@10.25.33.110:~/path/to/shim/
scp phase3/guest-shim/gpu_properties.h test-10@10.25.33.110:~/path/to/shim/
```

### Step 6: Rebuild on VM
```bash
# On VM:
cd ~/path/to/shim/
sudo ./install.sh  # or make, depending on build system
```

### Step 7: Restart Ollama and Test
```bash
# On VM:
sudo systemctl restart ollama
sleep 5
journalctl -u ollama --since "10 seconds ago" | grep -E "initial_count|library=|verifying"
```

## Files to Check/Fix

1. **gpu_properties.h**
   - Line with `GPU_DEFAULT_MAX_THREADS_PER_BLOCK`
   - Must be `1024`, not `1620000`

2. **libvgpu_cuda.c**
   - Line ~2901: `g_gpu_info.max_threads_per_block = GPU_DEFAULT_MAX_THREADS_PER_BLOCK;`
   - Line ~3611: `case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *pi = g_gpu_info.max_threads_per_block;`

3. **libvgpu_cudart.c**
   - Line ~426: `*value = GPU_DEFAULT_MAX_THREADS_PER_BLOCK;`
   - Line ~479: `prop->maxThreadsPerBlock = GPU_DEFAULT_MAX_THREADS_PER_BLOCK;`

## Expected Result

After fix:
- ✅ `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)` returns `1024`
- ✅ Ollama detects GPU: `initial_count=1`, `library=cuda_v12`
- ✅ All other GPU programs continue to work (generic fix)

## Notes

- This is a **general SHIM fix**, not Ollama-specific
- Fixing this will help ALL GPU programs that validate this attribute
- The fix is simple: ensure constant is 1024, not 1620000
- Must rebuild library after fix
