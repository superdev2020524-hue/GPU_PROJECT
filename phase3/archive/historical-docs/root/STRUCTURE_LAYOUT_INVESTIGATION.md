# Structure Layout Investigation

## Date: 2026-02-27

## ChatGPT's Analysis

**GGML sees compute capability 0.0 despite returning 9.0 - likely structure field offset mismatch.**

### The Problem

Even though the shim returns `major=9, minor=0`, GGML reads `compute capability 0.0`. This suggests:

1. **Structure field mismatch** - `cudaDeviceProp` layout doesn't match GGML's expectations
2. **Field offset mismatch** - `major` and `minor` fields are at wrong offsets
3. **Structure size mismatch** - Padding/alignment differences

### Current Structure Definition

```c
typedef struct {
    char name[256];              // offset 0
    size_t totalGlobalMem;       // offset 256 (but size_t is 8 bytes on 64-bit)
    int major;                   // offset 264 (256 + 8)
    int minor;                   // offset 268 (264 + 4)
    // ... more fields
} cudaDeviceProp;
```

### Possible Issues

1. **Padding between fields** - Compiler may add padding for alignment
2. **Size_t alignment** - `size_t` (8 bytes) may require 8-byte alignment
3. **Structure size** - Total size may not match CUDA headers

### Fix Applied

1. Added detailed logging with field offsets
2. Added direct memory write to ensure values are set
3. Added verification logging after direct write

### Next Steps

1. Check actual field offsets in logs
2. Compare structure size with CUDA headers
3. Verify padding/alignment matches
4. Test if direct memory write fixes the issue
