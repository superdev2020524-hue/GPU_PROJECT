# dlopen Not Called - Critical Finding

## The Problem

**Ollama is NOT calling dlopen() to load libggml-cuda.so!**

### Evidence

- ✅ `libggml-cuda.so` files exist in `/usr/local/lib/ollama/cuda_v12/` and `cuda_v13/`
- ❌ No `dlopen()` calls logged (our interception should log all calls)
- ❌ `libggml-cuda.so` not loaded in runner process memory maps
- ❌ No device query function calls
- ❌ GPU mode is CPU

## What This Means

Ollama's discovery mechanism:
1. Scans PCI devices directly ✓
2. Calls `cuInit()` ✓ (via LD_PRELOAD library)
3. **But then doesn't proceed to load libggml-cuda.so** ✗

## Possible Explanations

1. **Ollama doesn't use dlopen()** - Maybe uses direct syscalls or different mechanism
2. **Discovery fails before dlopen** - Maybe a prerequisite check fails
3. **Ollama uses LD_PRELOAD libraries directly** - Maybe doesn't need to dlopen if libraries are already loaded
4. **Our dlopen interception isn't working** - Maybe calls happen before interception is enabled

## Key Insight

Since `cuInit()` IS being called, Ollama must be using our library somehow. But:
- It's not calling `dlopen()` to load it
- It's not calling device query functions
- It's not loading `libggml-cuda.so`

This suggests Ollama's discovery uses a completely different mechanism than expected.

## Next Steps

1. **Check if Ollama uses direct syscalls** - Maybe bypasses libc dlopen()
2. **Check if discovery uses symbol checks** - Maybe checks if functions exist via dlsym() without loading
3. **Check Ollama source code** - Understand exactly how discovery works
4. **Check if there's a prerequisite that fails** - Maybe something prevents discovery from proceeding
