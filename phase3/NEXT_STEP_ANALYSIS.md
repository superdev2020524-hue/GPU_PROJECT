# Next Step Analysis

## Current Status

✅ **What Works:**
- Libraries ARE loaded (10 references in process memory)
- Symbols ARE resolvable (dlsym() test passed)
- `/proc/driver/nvidia/version` interception works (test program shows interception)
- `open()` and `openat()` have early interception

❌ **What Doesn't Work:**
- Functions are NOT called (no `cuInit()`, `nvmlInit_v2()` messages)
- GPU mode stays CPU
- No interception messages in Ollama logs

## Key Insight

**Ollama might not check `/proc/driver/nvidia/version` at all!**

Instead, Ollama might:
1. Load libraries via `dlopen()` ✓ (we confirmed libraries load)
2. Check if functions exist via `dlsym()` ✓ (we confirmed symbols resolve)
3. **But then check something ELSE before calling functions** ✗
4. If that check fails, never calls functions

## Possible Prerequisite Checks

1. **`/dev/nvidia*` device files**
   - Ollama might check if `/dev/nvidia0`, `/dev/nvidiactl`, etc. exist
   - Our interception handles these in `is_nvidia_proc_file()`
   - But `stat()` and `access()` check process type FIRST
   - Maybe they need early interception too?

2. **Library version/capabilities**
   - Ollama might check library version before calling functions
   - Or check if library supports certain capabilities

3. **Wrapper function checks**
   - Ollama has `ggml_nvml_init` wrapper
   - Wrapper might check prerequisites before calling functions

4. **Subprocess discovery**
   - Discovery might happen in a subprocess
   - Subprocess might not have `LD_PRELOAD` set
   - Or subprocess doesn't inherit interception

## Next Steps

1. **Add early interception to `stat()` and `access()`**
   - Similar to what we did for `open()` and `openat()`
   - Intercept `/dev/nvidia*` files BEFORE process check
   - This ensures prerequisite checks pass

2. **Verify if discovery happens in subprocess**
   - Check if runner subprocess has `LD_PRELOAD`
   - Verify if subprocess has interception

3. **Check if Ollama uses wrapper functions**
   - Maybe `ggml_nvml_init` wrapper checks prerequisites
   - If wrapper fails, never calls our functions

4. **Consider early initialization**
   - If discovery won't call functions, we might need early init
   - But must be extremely careful to avoid VM crashes

## Recommended Action

**Add early interception to `stat()` and `access()` for `/dev/nvidia*` files.**

This ensures that if Ollama checks device files before calling functions, those checks will pass.
