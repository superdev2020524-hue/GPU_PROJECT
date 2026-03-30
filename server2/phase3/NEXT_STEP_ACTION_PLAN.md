# Next Step Action Plan

## Current Status

✅ **What's Working:**
- Shim libraries load correctly via LD_PRELOAD
- CUDA APIs (`cuInit`, `cuDeviceGetCount`) return correct values
- Backend library (`libggml-cuda.so`) CAN be loaded manually
- When loaded, backend calls our shim functions successfully
- Environment configured: `OLLAMA_LLM_LIBRARY=cuda_v12`

❌ **Current Blocker:**
- Ollama's bootstrap discovery doesn't load CUDA backend
- Reports `library=cpu` instead of `library=cuda`
- Model download in progress (llama3.2:1b)

## Next Steps (Action Items)

### 1. Test Model Execution (Primary Goal)
**Once model download completes:**
- Run a model inference: `ollama run llama3.2:1b 'test'`
- Check if CUDA backend loads during inference (not just discovery)
- Verify if our shims are called during actual model execution
- This is the real test - discovery might fail but inference might work

### 2. Verify Runner Subprocess Environment
**Check if runner inherits LD_PRELOAD:**
- Find runner PID during model execution
- Check `/proc/<runner_pid>/environ` for LD_PRELOAD
- Verify runner has access to our shims

### 3. Force Backend Loading (If Needed)
**If backend still doesn't load during inference:**
- Check if there's a way to bypass discovery check
- Verify if backend loads on-demand when model runs
- Consider if we need additional environment variables

### 4. Verify GPU Usage During Inference
**Once model runs:**
- Check if CUDA functions are called during inference
- Monitor if GPU memory allocation happens
- Verify if compute operations go through our shims

## Key Insight

**Discovery failure doesn't necessarily mean inference failure.**

Ollama might:
1. Fail discovery (reports `library=cpu`)
2. But still load CUDA backend when model actually runs
3. Use GPU for inference even if discovery said "no GPU"

## Immediate Action

**Wait for model download to complete, then test actual inference.**

This will tell us if:
- Backend loads during inference (even if discovery failed)
- Our shims are called during actual GPU operations
- The virtual GPU works end-to-end
