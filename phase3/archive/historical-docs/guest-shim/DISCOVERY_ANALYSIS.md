# Discovery Analysis - Root Cause Identified

## Date: 2026-02-25 09:04:08

## Critical Finding

**Discovery now completes quickly (500ms) but still doesn't find GPU**

### Before Fix
- Discovery timeout: 30 seconds
- Error: "failed to finish discovery before timeout"

### After Fix (libvgpu-exec.so updated)
- Discovery completes: 500ms (no timeout!)
- But result: `total_vram="0 B"` (GPU not found)

## Discovery Sequence Analysis

### Main Process (pid=96581) ✅
- ✅ cuInit() called - SUCCESS
- ✅ Device found at 0000:00:05.0
- ✅ NVML init - SUCCESS
- ✅ Runtime API shim - SUCCESS

### Runner Subprocess (Discovery) ⚠️
- ✅ Discovery starts: "discovering available GPUs..."
- ✅ Discovery completes: 500ms (no timeout)
- ❌ **No logs from device count functions** (cuDeviceGetCount, nvmlDeviceGetCount_v2)
- ❌ Result: total_vram="0 B"

## Root Cause Hypothesis

**Discovery completes but device count functions are NOT being called in runner subprocess**

Possible reasons:
1. **Runner doesn't have shims loaded** - Despite libvgpu-exec.so fix
2. **Discovery uses different mechanism** - May not call device count functions
3. **Functions called but not logged** - Logs may not be reaching journalctl
4. **Discovery checks something else** - May check library existence instead of calling functions

## Evidence

### What We See
- Main process logs: ✅ All shim logs present
- Runner process logs: ❌ No shim logs from runner
- Device count calls: ❌ Not seen in logs
- Discovery result: ❌ total_vram="0 B"

### What We Don't See
- No `cuDeviceGetCount() CALLED` logs from runner
- No `nvmlDeviceGetCount_v2() CALLED` logs from runner
- No evidence runner has shims loaded

## Next Steps

### 1. Verify Runner Has Shims
```bash
# Check if runner process has shim libraries loaded
RUNNER_PID=$(pgrep -f "ollama runner" | head -1)
sudo cat /proc/$RUNNER_PID/maps | grep libvgpu
```

### 2. Check Runner Environment
```bash
# Check if runner has LD_PRELOAD
sudo strings /proc/$RUNNER_PID/environ | grep LD_PRELOAD
```

### 3. Add More Logging
- Add logging to verify runner process type
- Add logging to verify shims are loaded in runner
- Add logging to all device query functions

### 4. Research Ollama Discovery
- How does Ollama's discovery actually work?
- What functions does it call?
- What does it check before loading libggml-cuda.so?

## Key Questions

1. **Does runner have shims?** - Need to verify
2. **Are device count functions called?** - No evidence they are
3. **What does discovery actually check?** - May not be what we think
4. **Why does discovery complete but find no GPU?** - Completes but returns 0 devices

## Progress

✅ **Timeout resolved** - Discovery completes in 500ms (was 30s timeout)
❌ **GPU not found** - total_vram="0 B" still
⏳ **Root cause** - Device count functions not called in runner?

## Solution Path

1. Verify runner has shims loaded
2. If not, fix libvgpu-exec.so injection
3. If yes, investigate why device count functions aren't called
4. Research Ollama's actual discovery mechanism
