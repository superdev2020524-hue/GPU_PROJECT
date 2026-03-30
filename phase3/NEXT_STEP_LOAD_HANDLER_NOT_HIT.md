# Next step: load handler not hit (Mar 18, 2026)

## What was done

1. **Verified** VM/host status and Ollama GPU mode (VERIFICATION_REPORT_MAR18.md).
2. **Rebuilt** ollama on VM from current source (includes Phase3 logging in `runner/ollamarunner/runner.go`).
3. **Installed** new binary as `ollama.bin.new` and restarted ollama.
4. **Added** unconditional entry log: at the very start of the load handler (before status check), write `"entered\n"` to `/tmp/runner_load_entered.txt`.
5. **Re-tested** generate (tinyllama): after a 60s run, **both** `/tmp/runner_load_entered.txt` and `/tmp/runner_load_gpulayers.txt` were **missing**; call_sequence still had **0** alloc/HtoD (0x0030/0x0032).

## Finding

The **ollama-engine runner’s load handler** (`runner/ollamarunner/runner.go` `load()`) is **not** being called when the server sends load requests for tinyllama, or the request never reaches it.

- Journal shows: server starts runner with `--ollama-engine` and later "loading first model" (sched.go:259).
- So the server believes it is using the ollama-engine runner and has begun loading.
- The runner subprocess is `ollama.bin.new runner --ollama-engine --port <port>`; the same binary was rebuilt with the entry log.

So either:

- **A)** The server never sends the first load (e.g. `waitUntilRunnerLaunched` or something before the Fit loop never completes, or times out), or  
- **B)** The load request is sent but the runner process that receives it is not the one we think (e.g. wrong port, or an older runner process without the new code), or  
- **C)** The runner receives the request but the handler returns or panics before the first line (e.g. lock or defer issue).

## Update: server-side log added (Mar 18)

- **Patch:** In `llm/server.go` before `initModel(ctx, s.loadRequest, operation)` added `slog.Info("Phase3 sending load to runner", "operation", operation, "port", s.port)`. Rebuilt and installed; new binary running (06:15).
- **Test:** Triggered generate; journal shows **"loading first model"** (sched.go:259) at 06:16:48 but **no** "Phase3 sending load to runner".
- **Conclusion:** The server never reaches the first `initModel` call. So it is **stuck before** the load loop — most likely in **`waitUntilRunnerLaunched`** (polling `getServerStatus` until the runner responds). So the runner either never responds to the status check, or the server times out / takes a different path. Next: add log inside `waitUntilRunnerLaunched` (e.g. when getServerStatus is called and when it returns) or confirm the runner’s HTTP server is up and responding on the expected port.

## Recommended next steps

1. **Server-side log**  
   In `llm/server.go`, immediately before `initModel(ctx, s.loadRequest, operation)` (e.g. around line 827), add a log (e.g. `slog.Info("Phase3 sending load to runner", "operation", operation, "port", s.port)`). Rebuild, reinstall, trigger generate. If this log never appears, the server is not sending load (e.g. stuck in `waitUntilRunnerLaunched` or earlier). If it appears, the server is sending; then the issue is on the runner side or the connection.

2. **Confirm runner process and port**  
   When a generate is in progress, from the VM list processes and the port:  
   `pgrep -fa "ollama.*runner"` and `ss -tlnp | grep ollama` (or similar). Confirm that the runner process is the new binary and that the server’s `s.port` matches the runner’s listening port.

3. **OLLAMA_DEBUG**  
   With `OLLAMA_DEBUG=1` already set in the service override, check journal for any error or timeout from the server around “loading first model” or “do load request”, and for any runner-side error.

4. **If load is sent but handler not hit**  
   Consider that the binary that handles the request might be a different build (e.g. runner started before the install, or a cached runner). Ensure a single runner process is used for the load and that it is the newly installed binary.

## References

- ERROR_TRACKING_STATUS.md §6–7  
- PHASE3_REVIEW_AND_RESUME.md §7  
- VERIFICATION_REPORT_MAR18.md  
