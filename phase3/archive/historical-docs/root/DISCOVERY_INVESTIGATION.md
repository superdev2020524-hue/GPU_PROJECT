# Discovery Investigation - Why initial_count=0

## Date: 2026-02-27

## Problem

**Ollama logs show:**
```
initial_count=0  ← No GPU detected
library=cpu      ← Falls back to CPU
```

## Discovery Process

From logs:
1. ✅ Ollama starts: `"discovering available GPUs..."`
2. ✅ Runner subprocess starts: `"starting runner" cmd="/usr/local/bin/ollama runner"`
3. ✅ Bootstrap discovery runs: `"bootstrap discovery took 232.754758ms"`
4. ❌ **GPU count = 0**: `"evaluating which, if any, devices to filter out" initial_count=0`
5. ❌ Falls back to CPU: `library=cpu`

## Functions That Should Be Called

### 1. cuDeviceGetCount()
**Implementation:**
- Always returns `count=1` immediately
- No `ensure_init()` call
- Logs: `"[libvgpu-cuda] cuDeviceGetCount() CALLED"`
- Logs: `"[libvgpu-cuda] cuDeviceGetCount() SUCCESS: returning count=1"`

**Status:** ✅ Should work, but need to verify if called

### 2. cuDeviceGet()
**Implementation:**
- Always returns `device=0` immediately
- No `ensure_init()` call
- Logs: `"[libvgpu-cuda] cuDeviceGet() CALLED"`
- Logs: `"[libvgpu-cuda] cuDeviceGet() SUCCESS: device=0"`

**Status:** ✅ Should work, but need to verify if called

### 3. cuMemGetInfo_v2()
**Implementation:**
- Fixed to always return valid values (78GB free, 80GB total)
- Logs: `"[libvgpu-cuda] cuMemGetInfo_v2() returning: free=...MB, total=...MB"`

**Status:** ✅ Fixed, but may not be called if discovery fails earlier

## Hypothesis

**The runner subprocess may not be loading the shim**, or:
1. Runner subprocess has different environment variables
2. Runner subprocess doesn't have LD_PRELOAD set
3. Runner subprocess uses different library path
4. Discovery happens before shim is loaded

## What to Check

1. **Runner subprocess environment:**
   - Does it have `LD_PRELOAD`?
   - Does it have `OLLAMA_LIBRARY_PATH`?
   - Does it have `LD_LIBRARY_PATH`?

2. **Runner subprocess logs:**
   - Are there any `cuDeviceGetCount` calls?
   - Are there any shim initialization messages?
   - Are there any errors?

3. **Service configuration:**
   - Is `LD_PRELOAD` set in systemd service?
   - Is it inherited by subprocess?

4. **Library loading:**
   - Is the shim actually loaded in the runner process?
   - Are symbols being intercepted?

## Next Steps

1. Check runner subprocess environment variables
2. Check if runner subprocess has LD_PRELOAD
3. Look for shim logs from runner subprocess
4. Verify service configuration passes environment to subprocess
