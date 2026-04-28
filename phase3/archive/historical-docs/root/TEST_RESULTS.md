# Test Results - GPU Attributes Fix

## Date: 2026-02-27

## Fix Applied

✅ **Explicit hardcoded return value of 1024** for `CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK` added to `libvgpu_cuda.c`

## Verification

### Source Code
- ✅ Fix present in VM source: `/home/test-10/phase3/guest-shim/libvgpu_cuda.c`
- ✅ Code shows: `*pi = 1024;` for MAX_THREADS_PER_BLOCK case

### Library Build
- ✅ Library rebuilt after fix
- ✅ Library installed: `/usr/lib64/libvgpu-cuda.so`

### Ollama Status
- ✅ Ollama is running (active)
- ✅ VGPU-STUB detected at 0000:00:05.0
- ✅ GPU defaults applied (H100 80GB)
- ✅ Discovery message appears: "discovering available GPUs..."
- ⚠️ Still shows `library=cpu` (discovery may need model execution in runner subprocess)

### Model Test
- ⚠️ Model file appears corrupted: `llama3.2:1b` blob file has invalid magic
- ⚠️ Cannot test with actual model execution due to corrupted model file

## Current Status

**The fix has been applied and library rebuilt.** However:

1. **Discovery still shows CPU mode** - This may be because:
   - Discovery runs in runner subprocess (not main process)
   - Discovery may need actual model execution to complete
   - There may be other issues preventing GPU detection

2. **Model file is corrupted** - Cannot test with actual model execution

## Next Steps

1. **Fix or re-download model** - The model blob file appears corrupted
2. **Check runner subprocess logs** - Discovery may happen in runner, not main process
3. **Verify attribute is returned correctly** - Need to test with a working model or direct test

## Notes

- The fix is a **general SHIM improvement** - ensures all GPU programs get correct value
- Source code fix is correct and library was rebuilt
- Need working model to fully test GPU detection
