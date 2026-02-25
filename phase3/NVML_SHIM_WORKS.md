# NVML Shim Works - But Ollama Doesn't Call It

## Critical Finding

**NVML shim works perfectly when loaded manually:**
- ✅ Library loads successfully
- ✅ `nvmlInit_v2()` works and returns 0 (success)
- ✅ `nvmlDeviceGetCount_v2()` works and returns count=1
- ✅ Our shim messages are printed
- ✅ GPU is discovered: `0000:00:05.0`

**But Ollama is NOT calling it:**
- ❌ No libraries in process memory
- ❌ No shim messages in logs
- ❌ GPU mode is CPU

## The Problem

**Ollama's discovery is either:**
1. Not happening at all
2. Not calling NVML functions
3. Failing before it gets to NVML

## What We Know

1. **NVML shim is functional** - works when loaded manually
2. **NVML shim is accessible** - can be found via `dlopen()`
3. **Symlinks are correct** - all point to our shim
4. **Ollama has correct LD_LIBRARY_PATH** - includes `/usr/local/lib/ollama`
5. **But Ollama never calls NVML** - no shim messages in logs

## The Real Question

**Why isn't Ollama calling NVML during discovery?**

Possible reasons:
1. **Discovery doesn't happen** - Maybe discovery is skipped
2. **Discovery uses different mechanism** - Maybe not using `dlopen()` for NVML
3. **Discovery fails early** - Maybe fails before reaching NVML
4. **Discovery happens but doesn't call NVML** - Maybe uses different API

## Next Steps

1. **Check if discovery actually runs** - Look for "discovering available GPUs..." log
2. **Trace what discovery does** - Use `strace` to see what it actually calls
3. **Check if there's a different discovery mechanism** - Maybe not using standard NVML
4. **Check Ollama source code** - See how discovery actually works

## Key Insight

**The shim works. The problem is that Ollama isn't calling it.**

We need to understand HOW Ollama does discovery, not just assume it uses standard NVML API.
