# dlsym Interception Too Complex

## The Problem

Attempted to add `dlsym()` interception to see what functions Ollama is looking for, but ran into bootstrap/recursion issues.

## Why It's Complex

When intercepting `dlsym()`, we need to get the "real" `dlsym()` function to call it. But:
- We can't call our own `dlsym()` to get the real one (infinite recursion)
- `__libc_dlsym` may not be available on all glibc builds
- Using `RTLD_NEXT` still requires calling `dlsym()` which we're intercepting

## Current Status

- Removed complex `dlsym()` interception for now
- Functions are properly exported (verified via `nm` and `objdump`)
- `dlsym()` should be able to find our functions

## The Real Issue

Even without `dlsym()` interception, we know:
- ✅ Functions are exported
- ✅ `cuInit()` is called (from constructor)
- ❌ Device query functions are NOT called
- ❌ `libggml-cuda.so` is NOT loaded

This suggests Ollama's discovery doesn't use the standard mechanism we expected.

## Next Steps

1. **Check Ollama source code** - Understand exactly how discovery works
2. **Check if discovery uses different API** - Maybe not using standard NVML/CUDA functions
3. **Check if there are prerequisite checks** - Maybe something fails before function calls
4. **Consider alternative approaches** - Maybe we need to implement discovery differently
