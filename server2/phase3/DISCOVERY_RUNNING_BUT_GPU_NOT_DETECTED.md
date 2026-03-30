# Discovery Running But GPU Not Detected

## Date: 2026-02-26

## Key Finding

**Discovery IS running, but returning `initial_count=0` instead of `initial_count=1`!**

### Evidence from Logs

```
Feb 26 05:00:57 - bootstrap discovery took 239.805475ms
Feb 26 05:00:57 - OLLAMA_LIBRARY_PATH="[/usr/local/lib/ollama /usr/local/lib/ollama/cuda_v12]"
Feb 26 05:00:57 - initial_count=0  ← PROBLEM!
Feb 26 05:00:57 - library=cpu  ← PROBLEM!
```

### What This Means

1. ✅ **Discovery is running** - Bootstrap discovery completes in ~240ms
2. ✅ **OLLAMA_LIBRARY_PATH is set** - Scanner knows where to look
3. ❌ **GPU NOT detected** - `initial_count=0` (should be 1)
4. ❌ **CPU mode active** - `library=cpu` (should be `cuda` or `cuda_v12`)

## The Problem

Discovery is running, but it's not finding the GPU. This means:

1. **Device count functions are returning 0**
   - `cuDeviceGetCount()` returns 0 (should return 1)
   - `nvmlDeviceGetCount_v2()` returns 0 (should return 1)
   - OR these functions are not being called at all

2. **Or discovery is failing before device count**
   - Maybe `libggml-cuda.so` is not loading
   - Maybe initialization is failing
   - Maybe PCI bus ID matching is failing

## Comparison with Feb 25 Working State

**Feb 25 (Working):**
```
bootstrap discovery took 302ms
verifying if device is supported
library=/usr/local/lib/ollama/cuda_v12
description="NVIDIA H100 80GB HBM3"
initial_count=1  ← WORKING!
```

**Now:**
```
bootstrap discovery took 239ms
initial_count=0  ← NOT WORKING!
library=cpu
```

## What Needs Investigation

1. **Why is `initial_count=0`?**
   - Are device count functions being called?
   - Are they returning 0?
   - Or are they not being called at all?

2. **Why is `libggml-cuda.so` not loading?**
   - Is the scanner finding it?
   - Is it loading but failing initialization?
   - Are symbols resolving correctly?

3. **Why is there no "verifying if device is supported" message?**
   - This message appeared in Feb 25 working state
   - It's missing now
   - This suggests discovery is failing earlier

## Next Steps

1. **Check if device count functions are being called**
   - Add logging to see if `cuDeviceGetCount()` is called
   - Add logging to see if `nvmlDeviceGetCount_v2()` is called

2. **Check if `libggml-cuda.so` is loading**
   - Check if library is opened during discovery
   - Check if initialization succeeds

3. **Compare with Feb 25 working state**
   - What was different when it worked?
   - What configuration/files were present?

## Conclusion

**Discovery is running, but GPU detection is failing.**

The problem is that discovery returns `initial_count=0` instead of `initial_count=1`. This is why GPU mode is not active.
