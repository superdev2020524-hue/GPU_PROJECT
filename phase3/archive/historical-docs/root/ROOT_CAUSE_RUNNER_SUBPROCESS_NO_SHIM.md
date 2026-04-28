# Root Cause: Runner Subprocess Doesn't Have LD_PRELOAD

## Date: 2026-02-27

## Critical Finding

**The runner subprocess does NOT have `LD_PRELOAD` set**, so the shim is not loaded during discovery.

### Evidence

1. **Main process (pid=208867) HAS shim:**
   ```
   [libvgpu-cuda] cuDeviceGetCount() CALLED (pid=208867)
   [libvgpu-cuda] cuDeviceGetCount() SUCCESS: returning count=1 (pid=208867)
   [libvgpu-cuda] constructor: Application process detected (via LD_PRELOAD)
   ```

2. **Runner subprocess environment (from logs):**
   ```
   msg=subprocess PATH=... LD_LIBRARY_PATH=... OLLAMA_DEBUG=1 
   OLLAMA_LLM_LIBRARY=cuda_v12 OLLAMA_NUM_GPU=999 
   OLLAMA_LIBRARY_PATH=...
   ```
   **NO `LD_PRELOAD` mentioned!**

3. **Discovery happens in runner subprocess:**
   ```
   msg="starting runner" cmd="/usr/local/bin/ollama runner --ollama-engine --port 46589"
   msg="bootstrap discovery took" duration=232.754758ms
   msg="evaluating which, if any, devices to filter out" initial_count=0
   ```

4. **No shim logs from runner subprocess:**
   - All shim logs show `pid=208867` (main process)
   - No shim logs from runner subprocess PID
   - This means runner subprocess is NOT loading the shim

### The Problem

1. Main Ollama process loads shim (has LD_PRELOAD)
2. Main process calls `cuDeviceGetCount()` → returns 1 ✅
3. Main process starts runner subprocess
4. **Runner subprocess does NOT inherit LD_PRELOAD**
5. Runner subprocess runs discovery
6. Runner subprocess calls `cuDeviceGetCount()` → **NO SHIM** → returns 0 or fails
7. Discovery reports `initial_count=0`
8. Ollama falls back to CPU

### Why This Happens

Systemd service configuration may not be passing `LD_PRELOAD` to subprocesses, or Ollama's subprocess spawning doesn't inherit it.

### The Fix Needed

**Ensure runner subprocess has `LD_PRELOAD` set:**

1. **Option 1:** Set `LD_PRELOAD` in systemd service environment
2. **Option 2:** Set `LD_PRELOAD` in Ollama's subprocess environment
3. **Option 3:** Use `OLLAMA_LIBRARY_PATH` to point to shim location (if Ollama respects it)

### Current Service Configuration

From `/etc/systemd/system/ollama.service.d/vgpu.conf`:
```
[Service]
(empty - no LD_PRELOAD set)
```

### What to Check

1. Does systemd service have `Environment=LD_PRELOAD=...`?
2. Does Ollama pass environment to subprocess?
3. Can we set `LD_PRELOAD` via `OLLAMA_LIBRARY_PATH` or other mechanism?

### Next Steps

1. Check systemd service configuration
2. Add `LD_PRELOAD` to service environment
3. Verify runner subprocess gets `LD_PRELOAD`
4. Test discovery again
