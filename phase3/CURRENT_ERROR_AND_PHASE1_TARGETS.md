# Current error and what we must identify for Phase 1

*Clear summary: what is failing now, and what we need to find/fix to reach the Stage 1 milestone.*

---

## Phase 1 (Stage 1) milestone

**Goal:** Successfully complete GPU-mode inference in Ollama in the VM — i.e. a generate request runs end-to-end: model loads on the vGPU path, inference runs on the host GPU (H100), and the VM receives the generated response.

---

## The error we are currently experiencing

**In one sentence:** The Ollama server times out waiting for the llama runner to report load progress (progress stays 0.00), so the load is aborted and the generate request returns HTTP 500 with *"timed out waiting for llama runner to start - progress 0.00"*.

**What actually happens:**

1. User (or API client) sends a generate request (e.g. `POST /api/generate` for `llama3.2:1b`).
2. The server starts the **llama runner** subprocess to load the model and run inference on the vGPU.
3. The runner **does start**: it loads the model file, uses the vGPU shims, offloads 17/17 layers to GPU, and prints `load_all_data: using async uploads for device CUDA0, buffer type CUDA0, backend CUDA0`.
4. The runner **never sends a load-progress update** to the server. The server waits for progress (e.g. 0.00 → 0.25 → 1.00) on a timeout (about 5 minutes).
5. When the timeout expires, the server reports **Load failed** with  
   `error="timed out waiting for llama runner to start - progress 0.00 - "`  
   and returns **500** to the client.

**What we have already ruled out:**

- The runner does **not** write a CUDA or GGML error message to stderr before the timeout (we captured all runner stderr via strace; only normal load messages appear).
- So the problem is **not** a clear “CUDA error: …” or “ggml … failed” string from the runner; it is that **progress never advances** from the server’s point of view.

**So the current error is:**  
**Timeout due to no progress** — the runner never signals that load has progressed (or completed), so the server aborts the load and fails the generate request.

---

## What we must identify in the next stage (to achieve Phase 1)

We need to answer **why the runner never reports progress**. That means identifying one (or both) of the following:

### 1. Is the runner **stuck** (blocking)?

- **What to identify:** The exact call or code path where the runner blocks and never returns (so it never gets to the code that sends progress).
- **Likely places:**  
  - A **guest** CUDA/shim call that blocks (e.g. a transport RPC that never completes, or waits forever on the host).  
  - The **first** or an early HtoD (host-to-device) transfer, or the **first** GEMM or other GPU op during “async uploads” or right after.  
  - A **host** mediator path that never responds (e.g. a request type that is not handled or is mishandled, so the guest waits indefinitely).
- **How we might find it:**  
  - Add more logging in the guest shims/transport (before/after each RPC or around the first HtoD and first GEMM).  
  - Check host mediator logs for the same time window: do we see the corresponding requests, and do we see responses?  
  - Use strace/gdb to see which syscall or frame the runner is blocked in (e.g. a read on a pipe/socket that never returns).

### 2. Is the runner **exiting** (crash/signal) before it can report progress?

- **What to identify:** Whether the runner process exits (or is killed) before sending any progress, and **why** (exit code, signal, or abort).
- **Likely causes:**  
  - Crash in GGML or in a guest shim (e.g. segfault, assert, or unhandled CUDA error that aborts).  
  - Host closing the connection or returning an error that the guest interprets as fatal.  
  - OOM or other resource limit (e.g. on the host) that kills the runner or the mediator.
- **How we might find it:**  
  - From the strace run: inspect **exit_group(** *code* **)** (and any **kill** or signal) for the runner PID to see exit code or signal.  
  - Ensure the server (or our scripts) logs the runner’s **exit status** when the subprocess exits (e.g. in Ollama’s runner reaper).  
  - Reproduce with the host mediator in verbose/debug mode and with host logs captured to see if the host reports an error or disconnection at the same time.

### 3. Is progress reported on a **different path** we are not using?

- **What to identify:** Where in the Ollama/llama runner code “load progress” is sent to the server (e.g. which channel, after which operations). Confirm that the vGPU path actually reaches that code (and that we are not failing earlier in a way that prevents progress from ever being sent).
- **How we might find it:** Search the Ollama/runner source for “progress” and the load/status reporting; add a log line just before progress is sent; run again and see if that line appears in runner stderr or in our strace capture.

---

## Summary table

| Item | Description |
|------|-------------|
| **Current error** | Server times out: *"timed out waiting for llama runner to start - progress 0.00"* — runner never reports load progress. |
| **Phase 1 milestone** | One full GPU-mode generate in the VM (model load + inference on vGPU → host H100 → response back). |
| **Next: identify (blocking?)** | The exact call or RPC where the runner blocks so it never sends progress (guest transport/shim vs host mediator). |
| **Next: identify (exiting?)** | Whether the runner exits/killed before sending progress, and the exit code, signal, or host/guest error cause. |
| **Next: confirm progress path** | Where progress is sent in the runner and that the vGPU path can reach that point; add logging there if needed. |

Once we know **whether** the runner is blocking or exiting, and **where** (which call or which component), we can fix that point (e.g. fix a hanging RPC, fix a crash, or fix progress reporting on the vGPU path) and then re-test until a generate completes successfully and Phase 1 is achieved.

---

## Update: actual error identified (Mar 16)

**See ACTUAL_ERROR_FOUND.md.**

The runner is **blocking** (not exiting). The **last RPC** it sends before blocking is **`cuMemcpyHtoD_v2`** (host-to-device copy during model load). So the runner is stuck waiting for the **host’s response** to an HtoD chunk. Next: confirm on the host whether that request is processed and a reply is sent; if yes, fix the guest–host reply path; if no, fix host handling or request delivery.
