# Load runner GPULayers fix (Phase 3)

*Mar 18, 2026*

---

## Finding

- **Server** sends non-empty GPULayers (createLayout returns layers; Phase3 fallback never logged).
- **Runner** still did not call `cuMemAlloc` (no 0x0030/0x0032 in call_sequence).
- **Cause:** The **first** load request is `LoadOperationFit`. It is sent in the **ollamaServer** path (Fit → Alloc → Commit). If `gpuLayers` from `createLayout()` is empty at that time, the first request has empty GPULayers; the runner then calls `allocModel` with empty GPULayers → CPU-only load.

---

## Fixes applied on VM

### 1. Early fallback in ollamaServer.Load() (llm/server.go)

Right **after** `gpuLayers, err := s.createLayout(...)` and before `waitUntilRunnerLaunched`, add:

```go
// Phase3: ensure first request (Fit) gets non-empty GPULayers when gpus exist
if len(gpus) > 0 && gpuLayers.Sum() == 0 { gpuLayers = ml.GPULayersList{{DeviceID: gpus[0].DeviceID, Layers: []int{0}}} }
```

So the **first** request (Fit) always has at least one layer on the first GPU when `gpus` is non-empty.

### 2. Runner-side log (runner/ollamarunner/runner.go)

After `slog.Info("load", "request", req)` append to `/tmp/runner_load_gpulayers.txt`:

- `gpulayers=<len> sum=<sum> op=<operation>` for every load request.

Use `os.O_CREATE|os.O_WRONLY|os.O_APPEND` (Go constant is **O_CREATE**, not O_CREAT).

This confirms what the runner receives (Fit with non-empty GPULayers after the fix).

### 3. Install

- **Stop** ollama first: `sudo systemctl stop ollama` (may take a few seconds).
- Then: `sudo cp /home/test-4/ollama/ollama.bin /usr/local/bin/ollama.bin.real`
- Then: `sudo systemctl start ollama`

If `cp` fails with "Text file busy", the old process is still running; wait for stop to complete or kill the process.

---

## Verify

1. Clear: `rm -f /tmp/runner_load_gpulayers.txt /tmp/vgpu_cuMemAlloc_called.log`
2. Trigger generate (e.g. `curl -X POST http://127.0.0.1:11434/api/generate -d '{"model":"tinyllama","prompt":"Hi","stream":false}'`).
3. Check:
   - `cat /tmp/runner_load_gpulayers.txt` — should show lines with `gpulayers=1 sum=1 op=0` (or similar) for Fit.
   - `wc -l /tmp/vgpu_cuMemAlloc_called.log` — should be > 0 if the runner uses GPU for load.
   - `grep -cE "0x0030|0x0032" /tmp/vgpu_call_sequence.log` — should be > 0.

---

## References

- ERROR_TRACKING_STATUS.md §6 (runner-side backend selection).
- STAGE1_SCHED_NUMGPU_FIX.md (scheduler NumGPU fix).
