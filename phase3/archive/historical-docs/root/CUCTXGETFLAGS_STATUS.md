# cuCtxGetFlags Export Issue - Current Status

## Problem
Ollama fails to start with: `symbol lookup error: /opt/vgpu/lib/libcuda.so.1: undefined symbol: cuCtxGetFlags`

## Root Cause
The function `cuCtxGetFlags` is defined in the source file (line ~5770) but is NOT being compiled/exported in the shared library. The preprocessed file shows the function declaration but not the definition.

## Attempted Solutions
1. ✅ Fixed function definition syntax
2. ✅ Added `__attribute__((visibility("default")))`
3. ✅ Verified file transfer from local to VM
4. ✅ Tried `-Wl,--export-dynamic-symbol=cuCtxGetFlags`
5. ✅ Tried `-fvisibility=default`
6. ❌ Function still not exported

## Next Steps Needed
1. Investigate why function definition is not in preprocessed output
2. Check if function needs to be defined before stub_funcs array
3. Consider alternative: Make function optional if Ollama can work without it
4. Or: Provide a simpler stub implementation

## Current Blocking Issue
Ollama cannot start until this symbol is exported.
