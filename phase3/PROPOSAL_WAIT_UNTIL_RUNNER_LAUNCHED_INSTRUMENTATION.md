# Proposal: Instrument waitUntilRunnerLaunched (Phase 3)

*Proposal and implementation under granted authority (VM: full). Mar 18, 2026.*

---

## Role and authority

- **VM (test-4):** Full — edit files, rebuild, install, restart, run commands.
- **Host:** Read-only — no edits.

---

## Problem

The server reaches "loading first model" but never logs "Phase3 sending load to runner". So it never calls `initModel` and is stuck **before** the load loop. The only blocking point in that path is **`waitUntilRunnerLaunched`**, which loops on `getServerStatus(ctx)` until the runner’s **GET /health** returns a success status (Launched, Ready, or NoSlotsAvailable). If the runner never responds to /health, or returns another status, the server never proceeds and never sends load.

---

## Proposal

1. **Instrument `waitUntilRunnerLaunched`** in `llm/server.go`:
   - Log once when entering the loop (port).
   - Log every 50th failed poll with the error from `getServerStatus` (so we see "connection refused", "server not responding", or unmarshal/status errors) to avoid log flood.
   - Log when the runner responds (success) and when we exit due to context cancellation (timeout).

2. **Optionally log first failure in `getServerStatus`** when the HTTP request fails (e.g. connection refused), so the first error is visible without waiting for the 50th poll.

3. **Rebuild, install, restart** on the VM and trigger one generate. Inspect journal for Phase3 messages to see:
   - Whether we ever see "runner responded" (then the bug is elsewhere) or only poll/errors (then runner /health is not responding or wrong port).

4. **Document** the result and any follow-up (e.g. fix runner /health for ollama-engine, or port mismatch).

---

## Implementation (applying on VM)

- Patch `llm/server.go`: add a poll counter in `waitUntilRunnerLaunched`; log as above.
- Optionally in `getServerStatus`: log the error on first failure (or every failure with a cap).
- Rebuild `ollama.bin`, install to `ollama.bin.new` (via .tmp move), restart ollama, run one generate, capture journal.

---

## Success criteria

- Journal shows either "Phase3 waitUntilRunnerLaunched runner responded" (then we proceed to investigate why load is still not sent or not handled) or "Phase3 waitUntilRunnerLaunched poll" / "getServerStatus" errors (then we know the runner is not answering /health and can fix that path or the port).

---

## Implementation result (Mar 18, 2026)

- **Done:** Instrumentation added to `waitUntilRunnerLaunched` (poll counter, slog on first poll, every 50th failure, on success, on timeout). File write to `/tmp/phase3_wait_entered.txt` at function entry. Script `patch_wait_until_runner_launched.py` added in phase3.
- **Build/install:** Rebuilt on VM, installed via `.tmp` move; binary contains Phase3 strings and phase3_wait_entered path.
- **Observation:** After generate, `/tmp/phase3_wait_entered.txt` is **not** created; journal shows no "Phase3 waitUntilRunnerLaunched" messages. So **`waitUntilRunnerLaunched` is not being entered** for the run that logs "loading first model".
- **Conclusion:** The code path that logs "loading first model" (sched.go) and then calls `llama.Load()` either (1) does not reach the `Load()` implementation that calls `waitUntilRunnerLaunched` (e.g. different interface implementation or early return), or (2) runs in a context where the file write fails (e.g. different cwd or permission). Next: add entry file write at the very start of **ollamaServer.Load()** and **llamaServer.Load()** to see which, if any, is invoked; then trace why that path does not call or reach `waitUntilRunnerLaunched`.

---

## Load-path entry result (Mar 18, 2026)

- **Done:** Added file writes at start of **sched.load()** (`/tmp/phase3_sched_load_entered.txt`), **ollamaServer.Load()** and **llamaServer.Load()** (`/tmp/phase3_load_path.txt` with `ollama_load` or `llama_load`), and a write in sched **immediately before** `llama.Load()` (`/tmp/phase3_before_llama_load.txt`). Rebuilt and installed.
- **Observation:** On generate, **sched_load** is written, but **before_llama_load** and **load_path** are **not** written.
- **Conclusion:** The scheduler’s **load()** is entered, but execution **never reaches** the line just before `llama.Load()`. So the server is **blocked or returning** between the start of `load()` and the call to `llama.Load()`. The only non-trivial work in that span is **newServerFn()** (create/start runner). So **newServerFn() is likely blocking** (e.g. waiting for the runner to become ready) and never returns, so `llama.Load()` is never called and the load path never proceeds.
- **Next:** Inspect **newServerFn** (and how the runner is started and when it returns). If it waits for the runner to respond (e.g. a “launched” or health check), that wait may be the one that never completes; the fix is to make the runner respond correctly to that check (e.g. `/health`) or to fix the port/process so the server talks to the right runner.
