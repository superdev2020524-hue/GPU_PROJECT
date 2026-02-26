# Driver Version 13.0 Upgrade Applied

## Date: 2026-02-26

## Changes Made

### 1. Driver Version Upgraded to 13.0 (13000)

**File**: `phase3/guest-shim/gpu_properties.h`

**Changed**:
```c
#define GPU_DEFAULT_DRIVER_VERSION          13000  /* CUDA 13.0 (increased from 12.9) */
```

**From**: 12090 (12.9)  
**To**: 13000 (13.0)

### 2. Runtime Version Logic Updated

**File**: `phase3/guest-shim/libvgpu_cudart.c`

**Comment updated** to reflect that runtime version 12.8 (12080) is compatible with driver 12.9+ including 13.0:
```c
if (driver_version >= 12090) {
    runtime_version = 12080; /* CUDA 12.8 compatible with 12.9+ driver (including 13.0) */
}
```

## Why This Was Needed

According to `DRIVER_VERSION_INCREASED_TO_13.md`:
- CUDA runtime (libcudart.so.12) checks driver version
- Error: "CUDA driver version is insufficient for CUDA runtime version"
- Driver version 13.0 should satisfy the runtime's version check
- This allows `ggml_backend_cuda_init` to proceed further

## Expected Results

After rebuilding and restarting:
1. ✅ Driver version will be 13000 (13.0)
2. ✅ Runtime version will remain 12080 (12.8)
3. ✅ CUDA runtime version check should pass
4. ✅ `ggml_backend_cuda_init` should proceed further
5. ✅ Device query functions should be called
6. ✅ GPU mode should activate

## Deployment Status

### ✅ Completed
- Driver version updated in `gpu_properties.h` to 13000
- Runtime version logic verified (already handles >= 12090)
- Files updated on VM (verified via grep)

### ⏳ Pending
- Rebuild shim libraries: `cd ~/phase3/guest-shim && sudo ./install.sh`
- Restart Ollama: `sudo systemctl restart ollama`
- Verify driver version 13.0 in logs

## Verification

To verify the upgrade worked:
```bash
# Check driver version in logs
journalctl -u ollama --since "1 minute ago" | grep -iE "13000|13\.0|driver.*version"

# Check discovery status
journalctl -u ollama --since "1 minute ago" | grep -iE "discovery|library=|initial_count"
```

Expected logs:
- Driver version: 13000 or 13.0
- Runtime version: 12080 or 12.8
- Library: cuda_v12 (if GPU mode activates)
- initial_count: 1 (if GPU mode activates)

## References

- `DRIVER_VERSION_INCREASED_TO_13.md` - Original documentation
- `ROOT_CAUSE_FOUND_DRIVER_VERSION.md` - Initial version upgrade (12.8 → 12.9)
- `IMPLEMENTATION_COMPLETE_SUMMARY.md` - Runtime version compatibility logic

## Notes

- Runtime version remains 12080 (12.8) as it's compatible with driver 13.0
- The runtime version logic already handles driver >= 12090, so 13000 is covered
- This fix addresses the "CUDA driver version is insufficient" error
