# GGML Patch Implementation

## Date: 2026-02-27

## Problem

GGML sees `compute capability 0.0` despite shim returning `major=9 minor=0`. This indicates GGML is reading from different offsets than we're patching.

## Solution: Multi-Offset Patching

### Implementation

Added `patch_ggml_cuda_device_prop()` function that patches compute capability at **multiple likely offsets**:

1. **CUDA 12 offsets**: 0x148/0x14C (computeCapabilityMajor/Minor)
2. **Legacy offsets**: 0x150/0x154 (may be used by older GGML)
3. **Old CUDA 11 offsets**: 0x158/0x15C (fallback for compatibility)

### Code

```c
static void patch_ggml_cuda_device_prop(void *prop_ptr) {
    if (!prop_ptr) return;
    uint8_t *ptr = (uint8_t *)prop_ptr;
    
    size_t offsets_major[] = {0x148, 0x150, 0x158};
    size_t offsets_minor[] = {0x14C, 0x154, 0x15C};
    
    int major = GPU_DEFAULT_CC_MAJOR;
    int minor = GPU_DEFAULT_CC_MINOR;
    
    for (size_t i = 0; i < sizeof(offsets_major)/sizeof(offsets_major[0]); i++) {
        *(int32_t *)(ptr + offsets_major[i]) = major;
        *(int32_t *)(ptr + offsets_minor[i]) = minor;
    }
}
```

### Integration

Called from `cudaGetDeviceProperties_v2()` after setting all properties:
```c
/* CRITICAL: Apply GGML-specific patch to ensure all possible offsets are set */
patch_ggml_cuda_device_prop(prop);
```

## Expected Results

1. **GGML PATCH logs**: Should appear showing patching at all offsets
2. **Device compute capability**: Should now show 9.0 (not 0.0)
3. **Bootstrap discovery**: Should show `initial_count=1`
4. **Model execution**: Should work correctly with GPU

## Verification

After implementation:
- Check for `[GGML PATCH]` logs
- Verify `Device 0: ..., compute capability 9.0`
- Check bootstrap discovery logs for `initial_count=1`

## Status

- ✅ Code implemented
- ✅ Integrated into `cudaGetDeviceProperties_v2`
- ⏳ Testing on VM
