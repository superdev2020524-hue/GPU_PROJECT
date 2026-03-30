# Discovery Timeout Analysis

## The Problem

Ollama's GPU discovery is timing out with error:
- "failure during GPU discovery"
- "failed to finish discovery before timeout"

## What We Know

✅ **cuInit() is called** - Device discovery works
✅ **Device found** - "device found at 0000:00:05.0"
✅ **Dependencies resolved** - All libggml-cuda.so dependencies available
❌ **Discovery times out** - After 30 seconds
❌ **cuDeviceGetCount() NOT called** - No logs showing it's called
❌ **nvmlDeviceGetCount_v2() NOT called** - No logs showing it's called
❌ **libggml-cuda.so NOT loaded** - Never loads
❌ **GPU mode is CPU** - Falls back to CPU

## The Timeout

Discovery starts at time T, then times out at T+30 seconds. During this time:
- cuInit() is called ✓
- Device is found ✓
- But no device query functions are called ✗

## Possible Causes

1. **Discovery happens in subprocess** - Maybe runner subprocess doesn't have shims
2. **Ollama waits for something else** - Maybe checks something we're not intercepting
3. **Blocking operation** - Maybe something hangs and never completes
4. **Discovery mechanism different** - Maybe doesn't use standard device query functions

## Next Steps

1. Verify runner subprocesses have shims loaded (via libvgpu-exec.so)
2. Check if discovery happens in a subprocess that's hanging
3. Ensure all discovery functions return quickly without blocking
4. Check if there are other functions Ollama calls that we're not intercepting
