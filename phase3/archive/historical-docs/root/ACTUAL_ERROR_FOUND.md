# Actual error found: runner blocks on cuMemcpyHtoD_v2 response

*Identified by inspecting `/tmp/vgpu_call_sequence.log` on the VM after a generate attempt.*

---

## Finding

The **last RPC** the llama runner sends (and then blocks waiting for the response) is **`cuMemcpyHtoD_v2`** (call_id 0x0032).

From the VM's `/tmp/vgpu_call_sequence.log`:

- Sequence: `cuInit` → `cuGetGpuInfo` → `cuDevicePrimaryCtxRetain` / `cuCtxSetCurrent` (×2) → `cuMemAlloc_v2` (×1) → **many `cuMemcpyHtoD_v2`** (0x0032).
- The **last line** in the log is `0x0032 cuMemcpyHtoD_v2`.

So the runner:

1. Completed init, context, one alloc, and many HtoD chunks (model weight upload).
2. Sent **one more** HtoD chunk (the one that appears last in the sequence log).
3. Is **blocked** in the transport layer waiting for the **response** to that chunk from the host.

The server then times out because the runner never sends load progress — the runner is stuck in `cuda_transport_call` (or the chunked HtoD loop) waiting for the host to respond to that last `cuMemcpyHtoD_v2` request.

---

## Root cause (direction)

The failure is in the **HtoD (host-to-device) path** during model load:

- **Guest:** Sends HtoD chunks; the last chunk’s request is sent (logged to `vgpu_call_sequence.log`), then the guest blocks in `read()` (or equivalent) waiting for the host’s response.
- **Host:** Must process `CUDA_CALL_MEMCPY_HTOD`, call `cuMemcpyHtoD`, and send the result back.

So either:

1. **Host never processes that request** — request lost, wrong vm_id, or host busy/crashed.
2. **Host processes but is very slow** — e.g. a very large chunk or slow PCIe/copy; guest times out before response arrives.
3. **Host sends response but guest never receives it** — transport/BAR1/shmem reply path broken or dropped.

---

## What to do next (to reach Phase 1)

1. **Host mediator log (read-only):** On the host, during a generate from the VM, check `/tmp/mediator.log` (or wherever the mediator logs). For the same time window as the generate, check whether the host log shows:
   - A `cuMemcpyHtoD` (and optionally `cuMemcpyHtoD SUCCESS`) for the same vm_id **after** the previous HtoD logs.
   - If **yes** → host is replying; the bug is likely in the **guest–host reply path** (guest not reading the response, or BAR1/shmem reply not reaching the guest).
   - If **no** → host is not processing that last request (request not reaching host, or host not handling it).

2. **Chunk size:** Confirm `max_single_payload()` on the guest (BAR1 size or shmem half). If the last chunk is large (e.g. several MB), the host might still be copying when the server times out; consider logging chunk sizes and host-side duration for HtoD.

3. **Guest transport:** In `cuda_transport_call` / `do_single_cuda_call`, add a timeout or log when waiting for the response; confirm that the guest is indeed blocked in the read that waits for the HtoD reply.

4. **Fix:** Depending on (1)–(3): fix host handling of that request, fix reply path (BAR1/shmem), or increase timeouts / reduce chunk size so the response arrives before the server’s load timeout.

---

## Scripts used

- **track_runner_error_short.sh** — Short run (2 min) that prints last 60 lines of `vgpu_call_sequence.log` and last syscalls per child (strace).
- **run_track_runner_error_short.py** — Runs that script on the VM via `connect_vm`.
- The existing **vgpu_call_sequence.log** on the VM (from a prior run) was enough to see that the last RPC is `cuMemcpyHtoD_v2`.
