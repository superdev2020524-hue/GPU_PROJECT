# Root Cause: Exec Interception Not Working for Runner

## Date: 2026-02-26

## Critical Finding

**LD_PRELOAD is NOT being injected into runner subprocess environment!**

### Evidence

From discovery logs (05:00:57):
```
subprocess PATH=... LD_LIBRARY_PATH=... OLLAMA_LIBRARY_PATH=...
```

**Missing**: `LD_PRELOAD` is NOT in the subprocess environment!

### What This Means

1. **Exec interception is not working** - `libvgpu-exec.so` is loaded but not injecting LD_PRELOAD
2. **Runner subprocess doesn't get shims via LD_PRELOAD** - Must rely on symlinks
3. **Symlinks are correct** - All point to our shims ✓
4. **But discovery still fails** - initial_count=0

### Why This Is a Problem

**The chicken-and-egg problem:**
- Discovery needs device count > 0 to load `libggml-cuda.so`
- But device count is 0 because runner doesn't have shims (no LD_PRELOAD)
- Even though symlinks are correct, discovery doesn't load libraries when `OLLAMA_LLM_LIBRARY=cuda_v12` is set
- So shims never get used, device count stays 0, library never loads

### Why Exec Interception Isn't Working

Possible reasons:
1. **Ollama uses different mechanism** - Go runtime may use direct syscalls (clone, fork+exec) that bypass exec interception
2. **Ollama clears LD_PRELOAD** - May explicitly clear environment variables for security
3. **Exec interception not being called** - Maybe Ollama doesn't use execve/execv/execvp
4. **Logs not captured** - Maybe exec interception happens but logs go elsewhere

### Current Status

- ✅ `libvgpu-exec.so` loaded in main process
- ✅ Exec interception code is correct
- ❌ No exec interception logs found
- ❌ LD_PRELOAD not in runner environment
- ❌ Runner doesn't get shims via LD_PRELOAD

### Solution Options

1. **Fix exec interception** - Ensure it actually works for runner subprocess
2. **Use symlinks only** - Ensure shims work when loaded via symlinks (not LD_PRELOAD)
3. **Force library loading** - Pre-load library or force discovery to load it
4. **Different approach** - Find another way to inject shims into runner

## Next Steps

1. Verify why exec interception isn't working
2. Check if Ollama uses different mechanism to spawn runner
3. Ensure shims work when loaded via symlinks (not LD_PRELOAD)
4. Or find alternative way to ensure runner gets shims
