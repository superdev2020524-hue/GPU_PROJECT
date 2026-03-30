# Summary for ChatGPT Discussion

## Date: 2026-02-27

## Problem Statement

Ollama is not detecting the vGPU despite fixes to `cuMemGetInfo_v2` and other GPU attributes. Discovery shows `initial_count=0`.

## Fixes Applied (Based on ChatGPT's Analysis)

### 1. cuMemGetInfo_v2 Fix ✅
- **Problem:** Was returning `CUDA_ERROR_NOT_INITIALIZED` (error code 3)
- **Fix:** Always return valid memory values (78GB free, 80GB total) even if `ensure_init()` fails
- **Status:** Code fixed, library rebuilt

### 2. Unified Addressing Attribute Fix ✅
- **Problem:** May not return 1 consistently
- **Fix:** Always return 1 (required for H100)
- **Status:** Code fixed, library rebuilt

### 3. Virtual Memory API ✅
- **Status:** Symbols verified (`cuMemAddressReserve`, `cuMemCreate`, `cuMemMap` all exist)

## Current Status

### What Works
- ✅ Main Ollama process loads shim (has `LD_PRELOAD`)
- ✅ Main process calls `cuDeviceGetCount()` → returns 1
- ✅ Shim functions work correctly when called

### What Doesn't Work
- ❌ Runner subprocess does NOT have `LD_PRELOAD`
- ❌ Runner subprocess runs discovery without shim
- ❌ Discovery reports `initial_count=0`
- ❌ Ollama falls back to CPU

## Root Cause Identified

**The runner subprocess does NOT inherit `LD_PRELOAD` from the main process.**

### Evidence:
1. Main process (pid=208867) has shim loaded - logs show shim calls
2. Runner subprocess environment shows `LD_LIBRARY_PATH` but NO `LD_PRELOAD`
3. Discovery happens in runner subprocess
4. No shim logs from runner subprocess PID
5. Discovery reports `initial_count=0`

### Discovery Flow:
1. Main process starts → loads shim (LD_PRELOAD set)
2. Main process calls `cuDeviceGetCount()` → returns 1 ✅
3. Main process spawns runner subprocess
4. **Runner subprocess does NOT inherit LD_PRELOAD** ❌
5. Runner subprocess runs discovery
6. Runner subprocess calls `cuDeviceGetCount()` → **NO SHIM** → fails or returns 0
7. Discovery reports `initial_count=0`
8. Ollama falls back to CPU

## Questions for ChatGPT

1. **How to ensure runner subprocess gets LD_PRELOAD?**
   - Should we set it in systemd service environment?
   - Does Ollama pass environment to subprocess?
   - Is there another mechanism?

2. **Alternative approaches:**
   - Can we use `OLLAMA_LIBRARY_PATH` to point to shim?
   - Should we install shim in `/usr/local/lib/ollama/cuda_v12/` as `libcuda.so.1`?
   - Is there a way to force shim loading in runner?

3. **Why doesn't subprocess inherit LD_PRELOAD?**
   - Is this a systemd limitation?
   - Does Ollama explicitly clear environment?
   - Is there a security restriction?

## Files to Review

1. `/etc/systemd/system/ollama.service.d/vgpu.conf` - Service configuration
2. `/tmp/ollama_stderr.log` - Full logs showing discovery process
3. `libvgpu_cuda.c` - Shim implementation (cuDeviceGetCount, cuMemGetInfo, etc.)

## Next Steps

1. Fix systemd service to set `LD_PRELOAD` for subprocess
2. Or find alternative way to load shim in runner subprocess
3. Test discovery again after fix
