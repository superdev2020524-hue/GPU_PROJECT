# Next Steps After cuCtxGetFlags Issue

## Current Status
- ✅ Unified Memory APIs fixed (cuMemCreate, cuMemMap, etc.)
- ✅ All other CUDA functions working
- ❌ cuCtxGetFlags function defined but NOT being compiled/exported
- ❌ Ollama cannot start due to missing symbol

## Root Cause Analysis
The function `cuCtxGetFlags` is defined in source (line ~5770) but:
1. Does NOT appear in preprocessed output (libvgpu_cuda.i)
2. Shows as `*UND*` (undefined) in object file
3. Not exported in shared library

Even copying the exact pattern from working `cuCtxGetDevice` function doesn't help.

## Possible Solutions to Try

### Option 1: Move Function Definition Earlier
Move `cuCtxGetFlags` definition to before line 3205 (before `stub_funcs` array) to match where other functions are defined.

### Option 2: Check for Preprocessor Issues
- Look for `#if` directives that might exclude the function
- Check for macro definitions that might affect compilation
- Verify no syntax errors preventing inclusion

### Option 3: Make Function Optional
- Check if Ollama can work without `cuCtxGetFlags`
- Provide a weak symbol or stub that returns default value
- Use `dlopen`/`dlsym` to make it optional

### Option 4: Alternative Implementation
- Define function in separate file and link it
- Use assembly to force export
- Use linker script to export symbol

## Recommended Next Steps
1. **Move function definition to before stub_funcs** (around line 3200)
2. **Check compilation with verbose flags** to see why it's excluded
3. **If still failing, make function optional** and test if Ollama works without it
4. **Proceed with other verification** (kernel launches, GPU utilization) once Ollama is running

## Impact
- Blocking: Ollama cannot start
- Priority: HIGH - must be resolved before end-to-end testing
- Workaround: May be able to proceed with other components if function is made optional
