# Host response verification log

## Purpose

Verify whether the guest accurately receives the host's completion response (status DONE/ERROR or `response_len`). Log file: **`/tmp/vgpu_host_response_verify.log`** (written by the guest shim in `cuda_transport.c`).

## Log format

- **Poll samples** (throttled: iter 1, then every 5 up to 100, then every 50):
  - `iter=N call_id=0xXXXX seq=M status=0xNN` or `... status=0xNN rlen=N`
  - `status=0x01` = BUSY, `0x02` = DONE, `0x03` = ERROR. `rlen` only present when iter ≥ 30.

- **Break reasons** (why the poll loop exited):
  - `BREAK reason=STATUS call_id=... status=0x02 iter=N` — guest saw DONE (or ERROR) in status register → **host response received via MMIO status**.
  - `BREAK reason=RESPONSE_LEN call_id=... status=0xNN rlen=M` — guest saw `response_len != 0` → **host response received via BAR0 response_len**.
  - `BREAK reason=TIMEOUT call_id=... status=0xNN iter=N` — guest never saw DONE or response_len → **host response not received** (or not in time).

## Review (Mar 17): response_len workaround for HtoD

- **Conducted:** Generate (tinyllama, 120s and 180s timeout) with verify log and call_sequence cleared.
- **Result:** No `SUBMIT call_id=0x0032` (or 0x0030); no `BREAK reason=RESPONSE_LEN`; no `BREAK reason=TIMEOUT`. Call sequence had only init/context (0x0001, 0x00f0, 0x0090, 0x0022); HtoD count 0.
- **Conclusion:** The runner never reached the first cuMemAlloc/HtoD in these runs, so the response_len workaround was **not triggered** and could not be evaluated for HtoD. Treat as **not worthwhile** to keep testing response_len for HtoD until the runner is confirmed to reach the alloc/HtoD path.

## Finding (tinyllama generate run)

- **Guest receives host response for init/context:** All observed RPCs (0x0001, 0x00f0, 0x0090, 0x0022) show `BREAK reason=STATUS ... status=0x02` — guest sees DONE.
- **No HtoD in this run:** `call_sequence.log` contained only 0x0001, 0x00f0, 0x0090, 0x0022; no 0x0030 or 0x0032. So the runner never sent the first cuMemAlloc/HtoD — the process that does model load may not be using the shim, or is stuck before the first HtoD. After adding `SUBMIT` lines for 0x0030/0x0032, a run that reaches HtoD will show `SUBMIT call_id=0x0032`; then we can see whether we get `BREAK reason=STATUS` (guest received) or `BREAK reason=TIMEOUT` (guest did not).

## Interpretation

- If you see `BREAK reason=STATUS status=0x02` for a given `call_id`: the guest **is** accurately receiving the host's completion for that call (MMIO status path works).
- If you see `BREAK reason=TIMEOUT` for a call: the guest never saw completion; either the host never sent it, or the value was lost/corrupted in the path (MMIO mismatch).
- Compare with mediator "CUDA result sent" and stub logs to correlate host-side completion with guest-side receipt.

## SUBMIT lines (HtoD / cuMemAlloc)

For `call_id` 0x0030 (cuMemAlloc) and 0x0032 (cuMemcpyHtoD_v2), the shim writes `SUBMIT call_id=0xXXXX seq=N (about to poll)` to the verify log **before** ringing the doorbell. If you never see `SUBMIT call_id=0x0032`, the runner never reached the first HtoD — the blocker is earlier (e.g. model load not using this runner, or stuck before first HtoD).

## Code location

Verification logging is in `guest-shim/cuda_transport.c`: poll loop writes to `/tmp/vgpu_host_response_verify.log` (BREAK on STATUS_DONE/ERROR, BREAK on RESPONSE_LEN, BREAK on TIMEOUT, throttled per-iteration samples); SUBMIT line for 0x0030/0x0032 before doorbell.
