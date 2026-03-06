# vGPU Ollama: client behaviour during deployment

## Problem

- On vGPU, **deployment** (loading the model from the guest onto the host GPU) is **slow** (often 15–20+ minutes for a ~1.3 GB model) because every byte goes over the remoting pipe.
- After deployment, usage is normal: the model stays on the host GPU and inference runs without re-transferring the model.
- **Current failure mode:** The **client** (browser, curl, Ollama CLI, or other HTTP client) often has a **short timeout** (e.g. 30s, 3 min). It gives up waiting for the first response and closes the connection. The server then aborts the load ("client connection closed before server finished loading"). So deployment never completes and it looks like "every time" is slow, even though the design is **deploy once, then use**.

## Direction (what we want)

1. **Treat deployment like normal Ollama:** Load once (wait for it), then use. The client must not give up during deployment.
2. **Show progress (or at least activity) on the client side** so the user sees that something is happening and is less likely to cancel out of impatience.
3. **No automatic timeout during deployment:** The client must not close the connection just because deployment is taking a long time. Wait until deployment completes (or the user explicitly cancels).
4. **Only allow "giving up" on explicit user cancel:** e.g. Ctrl+C. So: no client-side timeout during load; only user-initiated cancel.

## Summary

| What | Goal |
|------|------|
| **During deployment** | Show progress/activity; do not timeout; wait until load completes. |
| **Cancel** | Only when the user explicitly cancels (e.g. Ctrl+C). |
| **After deployment** | Use the model normally (same as real GPU). |

## Implementation approach

- **Server:** Already configured with `OLLAMA_LOAD_TIMEOUT=20m` in vgpu.conf so the server does not abort the load early.
- **Client:** Use a **patient client** that:
  - Shows a clear message that deployment is in progress (e.g. "Deploying model to GPU (vGPU: may take 15–20 min or longer)…" and elapsed time).
  - Does **not** set any time limit on the request; it waits until deployment completes (deployment can take 20+ minutes or longer on vGPU).
  - Exits only on: success (response received), server error, or **user interrupt (Ctrl+C)**.
- Provide a small script or wrapper (e.g. `ollama_vgpu_generate.py` or usage in docs) so users can run inference from the VM (or against the VM) without giving up during deployment.

## Files

- **This document:** Direction and rationale.
- **Patient client script:** `ollama_vgpu_generate.py` — client that:
  - Waits for deployment with **no time limit** (never gives up on its own).
  - Prints: "Deploying model to GPU (vGPU: may take 15–20 min or longer). Waiting... Ctrl+C to cancel." and updates with elapsed time every 10 seconds. No time limit; only user Ctrl+C cancels.
  - Exits only on: success (prints response), server error, or **Ctrl+C** (user cancel).
  - Usage: `python3 ollama_vgpu_generate.py [MODEL] [PROMPT]`; set `OLLAMA_HOST` if Ollama is not on localhost.

## How to use (VM)

From the VM where Ollama runs (or from your machine with `OLLAMA_HOST=http://VM_IP:11434`):

```bash
cd /path/to/phase3
python3 ollama_vgpu_generate.py llama3.2:1b "Say hello."
```

First run: deployment (model load) will take 15–20 min; the script will wait and show elapsed time. Press **Ctrl+C** only if you want to cancel. Later runs (model already loaded): response returns much faster.
