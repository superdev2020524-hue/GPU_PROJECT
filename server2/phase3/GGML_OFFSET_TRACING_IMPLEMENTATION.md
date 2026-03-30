# GGML Offset Tracing Implementation

## Date: 2026-02-27

## Problem

GGML sees `compute capability 0.0` despite shim returning `major=9 minor=0`. This suggests GGML is reading from different offsets than we're patching.

## Solution: Enhanced Tracing

### Changes Made

1. **Added Pointer Address Logging**
   - Log the exact pointer address passed to `cudaGetDeviceProperties_v2`
   - This helps trace what GGML is accessing

2. **Enhanced GGML CHECK Logging**
   - Log both `computeCapabilityMajor/Minor` (new CUDA 12 fields)
   - Log both `major/minor` (legacy fields)
   - Log all at their exact offsets

3. **Multiple Offset Checking**
   - Check CUDA 12 offsets: 0x148/0x14C
   - Check legacy offsets: `&prop->major` / `&prop->minor`
   - Check old CUDA 11 offsets: 0x158/0x15C (in case GGML uses those)

4. **Patching Multiple Offsets**
   - Patch CUDA 12 offsets (0x148/0x14C)
   - Also patch old CUDA 11 offsets (0x158/0x15C) as fallback
   - Ensure both new and legacy fields are set

## Code Changes

### 1. Pointer Address Logging
```c
char addr_buf[128];
int addr_len = snprintf(addr_buf, sizeof(addr_buf),
                       "[GGML TRACE] cudaGetDeviceProperties_v2 called with prop=%p device=%d\n",
                       (void*)prop, device);
```

### 2. Enhanced GGML CHECK
```c
[GGML CHECK] prop=%p: computeCapabilityMajor=%d computeCapabilityMinor=%d (at offsets 0x148/0x14C) major=%d minor=%d (legacy) multiProcessorCount=%d totalGlobalMem=%llu warpSize=%d
```

### 3. Offset Verification
```c
[GGML OFFSET CHECK] 0x148=%d 0x14C=%d legacy_major=%d legacy_minor=%d struct_size=%zu
[GGML OLD OFFSET CHECK] 0x158=%d 0x15C=%d
```

### 4. Multiple Offset Patching
```c
// CUDA 12 offsets
int *cc_major_ptr = (int*)((char*)prop + 0x148);
int *cc_minor_ptr = (int*)((char*)prop + 0x14C);
*cc_major_ptr = GPU_DEFAULT_CC_MAJOR;
*cc_minor_ptr = GPU_DEFAULT_CC_MINOR;

// Old CUDA 11 offsets (fallback)
int *old_major_ptr = (int*)((char*)prop + 0x158);
int *old_minor_ptr = (int*)((char*)prop + 0x15C);
*old_major_ptr = GPU_DEFAULT_CC_MAJOR;
*old_minor_ptr = GPU_DEFAULT_CC_MINOR;
```

## Expected Results

After this implementation:
1. We'll see exactly what pointer GGML passes
2. We'll see all possible offset values
3. We'll patch multiple offsets to cover all possibilities
4. We can determine which offset GGML actually reads from

## Next Steps

1. Review logs to see which offsets have correct values
2. If GGML still sees 0.0, check if it's using a different API
3. Consider intercepting at a different level if needed
4. May need to check GGML source code for exact structure layout
