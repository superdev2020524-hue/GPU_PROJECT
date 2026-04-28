# Solution Research Based on Captured Errors

## Captured Error Summary

**Primary Error:**
```
time=2026-02-25T08:54:34.396-05:00 level=INFO source=runner.go:464 
msg="failure during GPU discovery" 
error="failed to finish discovery before timeout"
```

**Key Finding:** Discovery times out after exactly 30 seconds with no intermediate error messages.

## Discovery Sequence Analysis

### What Works ✅
1. **cuInit()** - Succeeds, device found at 0000:00:05.0
2. **NVML Init** - nvmlInit_v2() succeeds
3. **Device Discovery** - H100 PCIe identified correctly
4. **Runtime API Shim** - Initialized successfully

### What Fails ❌
1. **Discovery Timeout** - Times out after 30 seconds
2. **No libggml-cuda.so Loading** - No evidence of library being loaded
3. **No ggml_backend_cuda_init() Call** - Function never called
4. **Silent Failure** - No error messages explaining why

## Root Cause Hypothesis

Based on the captured logs, Ollama's discovery mechanism:

1. **Starts discovery** - "discovering available GPUs..." message appears
2. **Waits for something** - No activity for 30 seconds
3. **Times out** - "failed to finish discovery before timeout"
4. **Falls back to CPU** - total_vram="0 B"

The 30-second timeout with no errors suggests Ollama is waiting for a specific event that never occurs. This is likely:

- **Waiting for libggml-cuda.so to load** - But it never loads
- **Waiting for ggml_backend_cuda_init() to complete** - But it's never called
- **Waiting for a specific function call** - That never happens

## Research Findings

### From Web Search

1. **Ollama Debug Logging**: Set `OLLAMA_DEBUG=1` for more detailed logs
2. **Library Loading**: Discovery may check for library existence before loading
3. **Timeout Mechanism**: 30-second timeout is hardcoded in discovery logic

### Key Insight

The discovery timeout occurs **before** libggml-cuda.so is loaded. This means:
- Discovery doesn't proceed to library loading
- Something is blocking discovery from reaching that stage
- The blocker is silent (no error messages)

## Potential Solutions

### Solution 1: Force libggml-cuda.so Loading

Ollama may be checking prerequisites before loading the library. We need to ensure all prerequisites are met:

1. **Verify library exists** - Check if libggml-cuda.so is accessible
2. **Check dependencies** - Ensure all dependencies are available
3. **Force loading** - Use LD_PRELOAD or other mechanism to force load

### Solution 2: Investigate Discovery Prerequisites

Discovery may require specific conditions:
1. **NVML device count** - Must return > 0
2. **CUDA device count** - Must return > 0  
3. **Specific function calls** - That must succeed before proceeding

### Solution 3: Trace Discovery Process

Use debugging tools to see what discovery is actually doing:
1. **strace** - Trace all syscalls during discovery
2. **ltrace** - Trace library calls
3. **gdb** - Step through discovery code

## Recommended Next Steps

### Immediate Actions

1. **Enable OLLAMA_DEBUG=1** - Get more detailed logs
   ```bash
   # Add to systemd drop-in
   Environment="OLLAMA_DEBUG=1"
   sudo systemctl daemon-reload
   sudo systemctl restart ollama
   ```

2. **Trace Discovery Process** - See what discovery actually does
   ```bash
   sudo strace -p $(pgrep -f "ollama serve") -s 2000 -e trace=open,openat,dlopen,access,stat 2>&1 | grep -E "(libggml|libcuda|discover)" > /tmp/discovery_trace.log
   ```

3. **Check Library Accessibility** - Verify libggml-cuda.so can be loaded
   ```bash
   LD_LIBRARY_PATH=/usr/local/lib/ollama/cuda_v12 ldd /usr/local/lib/ollama/cuda_v12/libggml-cuda.so
   ```

### Research Actions

1. **Find Ollama Source Code** - Locate runner.go:464 to understand timeout logic
2. **Understand Discovery Flow** - Map out what discovery checks before loading libraries
3. **Identify Prerequisites** - Determine what must succeed for discovery to proceed

## Implementation Status

✅ **Error Capture System**: Complete and deployed
✅ **Error Messages Captured**: Full error messages obtained
✅ **Analysis Complete**: Root cause identified (discovery timeout)
⏳ **Solution Research**: In progress
⏳ **Fix Implementation**: Pending solution research

## Next Action

**Enable OLLAMA_DEBUG=1** to get more detailed logs about what discovery is waiting for.
