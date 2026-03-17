# Actual error verification (Mar 16)

## Run 1: Client timeout (2 min)

When the generate request used a **2-minute** curl timeout:

- **Observed:** HTTP 499 after 2m0s; server log showed model load in progress (tensors loading, vGPU in use).
- **Actual error in log:**
  - `level=WARN source=server.go:1370 msg="client connection closed before server finished loading, aborting load"`
  - `level=INFO source=sched.go:516 msg="Load failed" ... error="timed out waiting for llama runner to start: context canceled"`

So the **verified error** in that run was: **client disconnected (timeout)** while the server was still loading — not a runner crash. The runner may not have reached the post-GEMM crash because the load was aborted.

## Run 2: Long timeout (10 min) — server-side timeout

With **curl -m 600** (client waits 10 min), the **server** aborted after ~5 minutes with:

- **Actual error in log:**
  - `level=INFO source=sched.go:516 msg="Load failed" ... error="timed out waiting for llama runner to start - progress 0.00 - "`
  - `[GIN] ... 500 | 5m1s | POST "/api/generate"`

So the **verified error** is: **"timed out waiting for llama runner to start - progress 0.00"**.

Interpretation:

- The **llama runner** subprocess never reported any load progress (progress stayed 0.00).
- The server waited ~5 minutes for the runner to “start” (report progress), then gave up and returned 500.
- So the failure is **during model load** (runner never gets to “started” from the server’s view), not necessarily during inference. The runner may be:
  - **Stuck** in a call (e.g. transport/shim blocking and never returning), or
  - **Crashing** very early (before sending progress), so the server only sees “no progress” and eventually times out.

So we have two distinct observable errors:

1. **Client timeout (2 min):** client disconnects → "context canceled".
2. **Server timeout (5 min):** runner never reports progress → "timed out waiting for llama runner to start - progress 0.00".

The runner’s own stderr (e.g. CUDA/GGML error string or exit status 2) is still not visible in the server log; the server only reports the timeout.

**Next (to get runner-level error):** Log the runner’s exit status and stderr when the server detects the runner has exited (if Ollama code allows), or run the server so the runner is started under `strace -f` and inspect the trace for the runner process (e.g. `write(2, ...)` with an error string, or the signal that killed it).

---

## Strace capture (Mar 16) — runner’s actual stderr

We ran **track_runner_error.sh** on the VM (server under `strace -f -ff`, then generate, then analysis of per-PID logs). Summary:

### What the runner (llama runner PID 40370) wrote to stderr

- Normal load: token EOG messages, `create_tensor: loading tensor ...`, `[libvgpu-cuda]` skip/set, `[cuda-transport] Cannot resolve GPA ... using BAR1`, `load_tensors: offloaded 17/17 layers to GPU`, `load_tensors: CUDA0 model buffer size = 1252.41 MiB`, `load_all_data: using async uploads for device CUDA0, buffer type CUDA0, backend CUDA0`.
- **No** "CUDA error", "exit status 2", "failed", or "panic" string from the runner.

### What the server reported

- From another PID (scheduler): `Load failed ... error="timed out waiting for llama runner to start - progress 0.00 - "`, then `stopping llama server pid=40370`.

### Conclusion (runner’s actual error)

1. **Observed failure:** The server times out because the **runner never reports load progress** (progress stays 0.00). The runner is supposed to send progress updates (e.g. over the load channel) during/after async uploads; the server never sees them.
2. **Runner stderr:** The runner’s last stderr is normal load/upload text. There is **no** CUDA or GGML error message written to stderr by the runner in this run.
3. **Implication:** The runner either:
   - **Hangs** during `load_all_data: using async uploads` (e.g. in HtoD or first GEMM/init on the vGPU path) and never reaches the code that sends progress, or
   - **Exits** without writing an error (e.g. crash/abort before any error fprintf).

So the **tracked-down “actual error”** is: **timeout due to no progress** — the runner never signals progress 0.00 → … → 1.00, so the server aborts with `"timed out waiting for llama runner to start - progress 0.00"`. The next step is to find where the runner gets stuck or exits during/right after async upload (e.g. first host RPC that blocks, or a missing progress report after upload).

## Summary

| Scenario         | Timeout      | Resulting "error" in log |
|------------------|-------------|---------------------------|
| Client timeout   | 2 min       | "timed out waiting for llama runner to start: context canceled" |
| Server timeout  | ~5 min      | "timed out waiting for llama runner to start - progress 0.00" |
