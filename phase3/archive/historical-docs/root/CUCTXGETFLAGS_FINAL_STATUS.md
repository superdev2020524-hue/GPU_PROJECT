# cuCtxGetFlags Export Issue - Final Status

## Problem Summary
Ollama fails to start with: `symbol lookup error: /opt/vgpu/lib/libcuda.so.1: undefined symbol: cuCtxGetFlags`

## Root Cause
The function `cuCtxGetFlags` is defined in source code but:
- ❌ Does NOT appear in preprocessed output (libvgpu_cuda.i)
- ❌ Shows as `*UND*` (undefined) in object file
- ❌ Not exported in shared library
- ❌ Compilation errors when moved to different locations

## Attempted Solutions (All Failed)
1. ✅ Fixed function definition syntax
2. ✅ Added `__attribute__((visibility("default")))`
3. ✅ Copied exact pattern from working `cuCtxGetDevice` function
4. ✅ Moved function before `stub_funcs` array
5. ✅ Tried various linker flags (`-Wl,--export-dynamic-symbol`, etc.)
6. ✅ Verified file transfer from local to VM
7. ❌ Function still not being compiled/exported

## Current State
- Function definition exists at line ~3205 (moved from ~5770)
- Compilation error: function reference in `stub_funcs[11].func` cannot find definition
- Object file shows function as undefined
- Shared library does not export the symbol

## Recommended Next Steps

### Option 1: Make Function Optional (Quick Workaround)
If `cuCtxGetFlags` is not critical for Ollama operation, we could:
- Remove it from `stub_funcs` array
- Provide a weak symbol or stub
- Test if Ollama works without it

### Option 2: Separate Compilation Unit
- Move function to separate `.c` file
- Compile separately and link
- This might bypass whatever is preventing compilation

### Option 3: Continue Deep Investigation
- Check for preprocessor macros affecting compilation
- Verify compiler version compatibility
- Check for hidden syntax errors or encoding issues

## Impact on Project
- **Blocking**: Ollama cannot start
- **Priority**: HIGH
- **Workaround Available**: Possibly make function optional
- **Other Components**: All other CUDA functions working correctly

## Recommendation
Given the time spent on this issue, recommend:
1. **Short-term**: Try making function optional and test if Ollama works
2. **Medium-term**: Move to separate compilation unit
3. **Long-term**: Deep dive into why function isn't being compiled
