# Enhanced Tracing Results

## Date: 2026-02-27

## Implementation

Added comprehensive tracing to determine exactly what GGML reads:
1. Pointer address logging
2. Enhanced GGML CHECK with all field values
3. Multiple offset checking (CUDA 12, legacy, old CUDA 11)
4. Patching multiple offsets as fallback

## Status

### Code Status
- ✅ Enhanced logging code added to source
- ✅ Code compiled and installed
- ⏳ Logs not appearing (checking if function is called)

### Findings
- Old logging format still appears: `returning: major=9 minor=0 (compute=9.0)`
- New GGML TRACE/CHECK/OFFSET logs not appearing
- This suggests either:
  1. Function isn't being called with new code path
  2. Old code path still being used
  3. Library wasn't rebuilt correctly

## Next Steps

1. Verify new code is in compiled library
2. Check if function is being called
3. If not appearing, investigate why old code path is used
4. May need to check if there are multiple versions of the function

## Expected vs Actual

**Expected:**
- `[GGML TRACE] cudaGetDeviceProperties_v2 called with prop=...`
- `[GGML CHECK] prop=...: computeCapabilityMajor=9 ...`
- `[GGML OFFSET CHECK] 0x148=9 0x14C=0 ...`

**Actual:**
- Old format: `returning: major=9 minor=0 (compute=9.0)`
- New logs not appearing

**Conclusion:** Need to verify code compilation and function execution path.
