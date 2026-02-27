# CUDA 12 Structure Layout Fix

## Date: 2026-02-27

## Problem

GGML sees `compute capability 0.0` despite shim returning `9.0`. This is due to structure field offset mismatch.

## Root Cause

The `cudaDeviceProp` structure in our shim didn't match CUDA 12's layout. GGML reads `computeCapabilityMajor` at offset `0x148`, but our structure had `major` at a different offset.

## CUDA 12 Structure Layout

Key offsets from CUDA 12 headers:
- `name`: 0x00 (256 bytes)
- `totalGlobalMem`: 0x100 (size_t = 8 bytes)
- `sharedMemPerBlock`: 0x108 (size_t = 8 bytes)
- `regsPerBlock`: 0x110 (int = 4 bytes)
- `warpSize`: 0x114 (int = 4 bytes)
- `memPitch`: 0x118 (int = 4 bytes)
- `maxThreadsPerBlock`: 0x11C (int = 4 bytes)
- `maxThreadsDim[3]`: 0x120 (12 bytes)
- `maxGridSize[3]`: 0x12C (12 bytes)
- `clockRate`: 0x138 (int = 4 bytes)
- `multiProcessorCount`: 0x13C (int = 4 bytes)
- `l2CacheSize`: 0x140 (int = 4 bytes)
- `maxThreadsPerMultiProcessor`: 0x144 (int = 4 bytes)
- **`computeCapabilityMajor`: 0x148 (int = 4 bytes)** ← CRITICAL
- **`computeCapabilityMinor`: 0x14C (int = 4 bytes)** ← CRITICAL

## Fix Applied

1. **Updated structure definition** to match CUDA 12 layout exactly
2. **Added `computeCapabilityMajor` and `computeCapabilityMinor` fields** at correct offsets
3. **Added direct memory patching** at known offsets (0x148/0x14C) as safety measure
4. **Set both new fields and legacy `major`/`minor` fields** for compatibility

## Code Changes

```c
// Now using computeCapabilityMajor/Minor instead of just major/minor
prop->computeCapabilityMajor = GPU_DEFAULT_CC_MAJOR;
prop->computeCapabilityMinor = GPU_DEFAULT_CC_MINOR;

// Direct memory patching at known offsets
int *cc_major_ptr = (int*)((char*)prop + 0x148);
int *cc_minor_ptr = (int*)((char*)prop + 0x14C);
*cc_major_ptr = GPU_DEFAULT_CC_MAJOR;
*cc_minor_ptr = GPU_DEFAULT_CC_MINOR;
```

## Expected Result

GGML should now see:
- `computeCapabilityMajor` = 9
- `computeCapabilityMinor` = 0
- `multiProcessorCount` = 132
- `totalGlobalMem` = 80GB
- `warpSize` = 32

Device validation should pass, fixing `initial_count=0`.
