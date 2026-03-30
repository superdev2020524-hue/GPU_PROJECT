# Work note: WaitUntilRunning and runner start (Mar 19, 2026)

## What was done

1. **Confirmed** the runner start/health instrumentation is already applied: `waitUntilRunnerLaunched` in llm/server.go logs "Phase3 waitUntilRunnerLaunched waiting for runner" and "runner responded" (port, polls).

2. **Located** the source of "timed out waiting for llama runner to start: context canceled": it is **WaitUntilRunning** in llm/server.go (lines 1438 and 1449). The "context canceled" variant is returned when **ctx.Done()** fires (request context canceled), e.g. when the **client** disconnects or times out.

3. **Ran** a generate with a **600s client timeout** (curl -m 600) so the context would not be canceled by the client for 10 minutes. Service already has OLLAMA_LOAD_TIMEOUT=40m and CUDA_TRANSPORT_TIMEOUT_SEC=2700.

4. **Captured** journal for the run (since 15 min ago). No code or config was changed; only observation.

## Result

- **waitUntilRunnerLaunched:** Succeeds. Journal shows "Phase3 waitUntilRunnerLaunched waiting for runner" then "Phase3 waitUntilRunnerLaunched runner responded" port=38331 polls=9. So the runner’s HTTP is up and the health check passes.

- **WaitUntilRunning:** The server then logs "waiting for llama runner to start responding" and "waiting for server to become available" status="llm server loading model". The runner **never** transitions to **ServerStatusReady** within 10 minutes; it stays in "llm server loading model".

- After **600s**, curl times out (exit 28). Journal: "client connection closed before server finished loading, aborting load" and "error loading llama server" error="timed out waiting for llama runner to start: context canceled". So the **context canceled** is the **client** (curl) disconnecting after 10 min, not a server-side timeout.

## Conclusion

- **Health check:** Runner responds to getServerStatus; waitUntilRunnerLaunched is not the blocker.
- **Blocker:** The runner never reports **ServerStatusReady**. It remains in "loading model" for at least 10 minutes. So either:
  1. Load (HtoD, etc.) is slow and has not finished in 10 min, or  
  2. The runner never sends progress / Ready to the server (progress callback or status update path), or  
  3. Something on the host or in the runner blocks before it can report Ready (e.g. module load, or sync after HtoD).

## Next step (recommended)

- **Runner progress / Ready reporting:** Find where the runner sets load progress and sends **Ready** (or equivalent) to the server. Confirm that the runner updates progress (and eventually Ready) when load completes. If the runner blocks before that (e.g. waiting on host/transport), fix that path or the progress callback so the server sees Ready once load is done.

## Change applied (non-destructive)

- In **llm/server.go** line 1464, added **progress** to the existing log: `slog.Info("waiting for server to become available", "status", status, "progress", s.loadProgress)`. So when status changes (and is not Ready), the journal will show load progress. Rebuilt and installed on VM so the next long generate will show whether progress advances (e.g. 0.00 → 0.5 → 1.0).

## Anti-destructive verification

- No code or config was changed. Only journal inspection and one long-timeout generate. GPU mode and existing behavior unchanged.
