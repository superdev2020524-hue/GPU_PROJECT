# Discovery Mechanism Analysis

## Key Findings

### 1. NVML Shim Works Perfectly
- ✅ All required symbols are exported
- ✅ Library loads successfully when called manually
- ✅ `nvmlInit_v2()` works and returns success
- ✅ `nvmlDeviceGetCount_v2()` works and returns count=1
- ✅ GPU is discovered: `0000:00:05.0`

### 2. Ollama Tries to Load NVML But Fails
- Binary contains: `'%s unable to load libnvidia-ml: %s'`
- Binary contains: `'%s unable to locate required symbols in libnvidia-ml.so'`
- But NO error messages appear in logs
- This means Ollama fails silently

### 3. Discovery Happens in Runner Subprocess
- Log shows: `source=runner.go:67 msg="discovering available GPUs..."`
- This is in the runner subprocess, not main process
- Runner has `LD_LIBRARY_PATH=/usr/local/lib/ollama:/usr/lib64:/usr/lib/x86_64-linux-gnu`
- Runner has `OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama`

### 4. All Symlinks Are Correct
- `/usr/local/lib/ollama/libnvidia-ml.so.1` → our shim ✓
- `/usr/lib64/libnvidia-ml.so.1` → our shim ✓
- All system paths → our shim ✓

## The Mystery

**Why does Ollama fail to load NVML when:**
1. All symbols are available ✓
2. Library can be loaded manually ✓
3. Symlinks are correct ✓
4. Paths are in LD_LIBRARY_PATH ✓

## Possible Reasons

### 1. Timing Issue
- Discovery might happen before libraries are accessible
- Runner subprocess might start before environment is set
- Library paths might not be ready yet

### 2. Path Resolution Issue
- Ollama might use absolute paths
- Might check specific locations first
- Might fail before checking all paths

### 3. Symbol Versioning
- Real NVML might have versioned symbols
- Our shim might not have versioned symbols
- Ollama might require specific symbol versions

### 4. Dependency Issue
- NVML might have dependencies we're missing
- Our shim might be missing required dependencies
- Loading might fail due to missing dependencies

### 5. Discovery Doesn't Use NVML
- Maybe discovery uses a different mechanism
- Maybe NVML is optional
- Maybe it only tries NVML under certain conditions

## Next Steps

1. **Check symbol versioning** - See if real NVML has versioned symbols
2. **Check dependencies** - See if our shim is missing dependencies
3. **Trace actual dlopen() calls** - See what path Ollama tries
4. **Check if discovery is optional** - Maybe it skips NVML if not found
5. **Check Ollama source code** - Understand how discovery actually works

## Critical Question

**Is NVML discovery optional?**
- If NVML fails, does Ollama skip GPU discovery entirely?
- Or does it try alternative methods?
- Or does it proceed to CUDA anyway?

This would explain why there are no errors - if NVML is optional, Ollama might just skip it silently.
