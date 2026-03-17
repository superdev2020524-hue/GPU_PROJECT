# HtoD diagnosis results (Mar 16, 2026)

## 1. Mediator status

- **Confirmed:** Mediator is running on the host.
  - Process: `./mediator_phase3` (PID 2677655, since Mar 15).
  - Log: `/tmp/mediator.log` exists and is written.

---

## 2. What was run

- **VM:** Stopped ollama, cleared `/tmp/vgpu_call_sequence.log`, started patched `ollama.bin serve` with vGPU env, triggered one generate (llama3.2:1b, stream=false, 2-token), waited 2 min, then collected logs.
- **Result:** The track script timed out (260 s) before the VM script finished; VM logs were then fetched separately.

---

## 3. Guest (VM) findings

- **`/tmp/vgpu_call_sequence.log`:** 36 lines total.
  - Sequence: `cuInit` → `cuGetGpuInfo` → `cuDevicePrimaryCtxRetain` / `cuCtxSetCurrent` (×2), then again (bootstrap), then **1× `cuMemAlloc_v2`**, then **many `cuMemcpyHtoD_v2`** (0x0032).
  - **Last line:** `0x0032 cuMemcpyHtoD_v2`.
- **Conclusion:** The runner sent one alloc and ~25 HtoD chunks, then **blocked** in the transport waiting for the host’s response to the **last** HtoD request (consistent with ACTUAL_ERROR_FOUND.md).

- **`/tmp/ollama_run.log` (strings):**
  - `model load progress 0.00` (once).
  - Then: `client connection closed before server finished loading`, `Load failed ... error="timed out waiting for llama runner to start: context canceled"`.
  - Request ended with HTTP 499 after ~4m2s (client timeout).
- **Conclusion:** Runner never advanced past “progress 0.00” because it was stuck waiting for an HtoD response; the client (curl) gave up first.

---

## 4. Host (mediator) findings

- **`/tmp/mediator.log`** was inspected with `grep -a` (text mode; log can contain binary bytes).
- **Last HtoD activity for vm=9:**  
  `[cuda-executor] HtoD progress: 7672 MB total (vm=9)`  
  then only `[MEDIATOR STATS]` lines (Total processed 8494 → 8498).
- **No new HtoD lines** after that 7672 MB line.
- So:
  - The **7672 MB** HtoD run is from an **older** session (full model transfer).
  - The **current** run (only ~25 HtoD chunks, ~tens of MB) did **not** produce new `HtoD progress (vm=9)` lines in the tail we inspected.

---

## 5. Interpretation

- **Guest:** Runner is blocking on the response to an HtoD request (last RPC = `cuMemcpyHtoD_v2`). So the **symptom** is unchanged: load blocks during HtoD.
- **Host:** The most recent vm=9 activity in the log is the old 7672 MB run. For the run we just triggered we either:
  1. **Same VM (test-4 = vm_id 9):** The new run’s HtoD requests are not reaching the mediator, or are not being logged (e.g. different code path), or the run we see on the guest is not the one that produced vm=9’s 7672 MB.
  2. **Different vm_id:** Test-4 in this run might be using a different vm_id (e.g. the other socket: root-220 vs root-222). Then we’d need to grep for the other vm_id’s HtoD in the same time window.

So we have **not** yet confirmed that the host is processing and replying to the **same** HtoD request the guest is blocked on. The host **does** process HtoD for vm=9 in general (proven by the 7672 MB run).

---

## 8. Follow-up run (after mediator restart, Mar 16)

**Setup:** Mediator restarted on host (only root-222 socket). Fresh generate triggered on VM; wait 90s; then VM and host logs collected.

**Guest (test-4):**
- `vgpu_call_sequence.log`: 23 lines. Last line: `cuMemcpyHtoD_v2`. Sequence: init/getinfo/ctx (×2), 1× cuMemAlloc_v2, then **12× cuMemcpyHtoD_v2**. Runner is blocking waiting for the response to the **12th** HtoD.

**Host (mediator.log since restart):**
- **vm=9** confirmed: `CUDA_CALL_INIT vm=9 — pipeline live` (×2), `cuMemAlloc SUCCESS` (1313251456 bytes), **`HtoD progress: 10 MB total (vm=9)`**.
- Total processed: 27.
- No HtoD progress line beyond 10 MB for vm=9.

**Conclusion:** Request path works: test-4 is vm_id=9; the host receives and processes INIT, the big alloc, and at least the first HtoD chunk(s) (logs 10 MB progress). The guest sent 12 HtoD chunks and is blocked. So either:
- **Reply path:** The host sends the HtoD response(s) but the guest never receives them (BAR1/shmem or read path), so the guest stays blocked in `cuda_transport_call` waiting for the reply; or
- **Next request:** The host only processed up to 10 MB and the 12th (or an intermediate) request has not been processed yet or is stuck.

Most likely the **reply path** is failing: host completes HtoD and writes the response; the guest never sees it and blocks in poll/read. Next: add timeout or debug logging in the guest transport when waiting for the response, and/or inspect BAR1/shmem reply delivery on the stub/host side.

---

## 6. Next steps (updated after follow-up run)

1. **Reply path (priority):** Host processes HtoD for vm=9 and logs 10 MB progress; guest sent 12 HtoD and is blocked. So the host is sending a response; the guest is not receiving it (or not reading it). Focus on:
   - **Guest:** In `cuda_transport.c`, after ringing the doorbell, the guest polls BAR0 for STATUS_DONE/STATUS_ERROR. Confirm the host writes the response (result, status) to BAR0/BAR1 and that the guest’s MMIO read sees it. Add a short timeout + log (e.g. “waiting for response to call_id X, seq Y”) to confirm the guest is stuck in the poll loop.
   - **Stub/host:** Ensure the vGPU stub (QEMU) writes the mediator’s response back into BAR0/BAR1 so the guest’s MMIO read returns DONE and the result. If the reply is sent over a socket but never written to guest-visible BAR, the guest will block forever.

2. **Optional – mediator log more:** Log every HtoD completion (e.g. seq or size) so we can count how many responses the host actually sent for this run.

---

## 7. Stub log location and reply-path check (Mar 16)

- **Where to read stub output on host:** `/var/log/daemon.log`. Process: `qemu-dm-<domain_id>` (e.g. `qemu-dm-223` for Test-4). Command: `grep -a 'CUDA result\|DOORBELL RING' /var/log/daemon.log | tail -50`.
- **Finding:** In a run with 32 RPCs in `vgpu_call_sequence.log` (last = cuMemcpyHtoD_v2), the host daemon.log showed **"CUDA result applied seq=26" through "seq=33"** (status=0 DONE). So the stub is receiving the mediator response and applying it (setting status_reg = DONE). The reply path mediator → stub → BAR is working on the host. If the guest still blocks, the cause may be guest-side (BAR read not seeing the update, or run aborted before response arrived). See **REPLY_PATH_DEBUG.md** for the full stub log location section.

**Earlier “transmission working” reports:** See **TRANSMISSION_VERIFICATION_REFERENCE.md**. Docs such as END_TO_END_VERIFICATION_SUCCESS.md correctly verified **one** round-trip (guest saw RECEIVED status=DONE). The same transmission method is used today; the current issue appears under **many** HtoD round-trips. The reference doc explains the distinction and next steps (e.g. memory barrier before guest status read, longer client timeout).

## 8. Reply-path debug changes (Mar 16)

- **Guest:** Periodic 5s "waiting for response" log (stderr + `/tmp/vgpu_poll_wait.log`). Deployed via `transfer_cuda_transport.py`.
- **Stub:** Log on VGPU_MSG_CUDA_RESULT (applied vs IGNORED). Host rebuild done; stub output in `/var/log/daemon.log`. See **REPLY_PATH_DEBUG.md**.

## 10. 40-minute client-timeout run (Mar 16)

**Setup:** Server started manually **without** `OLLAMA_LOAD_TIMEOUT` (so server used default ~5 min load timeout). Client: `curl -m 2400` (40 min). Model: `llama3.2:1b` (Oyu not on VM).

**Result:** Request ended after **~302 s** with **HTTP 500**. Server log: `Load failed ... error="timed out waiting for llama runner to start - progress 0.00"`. So the **server** aborted the load at its default 5 min timeout, not the client.

**Guest logs:**
- `vgpu_call_sequence.log`: 37 lines — init, alloc, then many `cuMemcpyHtoD_v2`; last line = HtoD. Runner sent many HtoD RPCs then blocked.
- `vgpu_poll_wait.log`: **empty** (no 5s “waiting for response” lines). Either the runner was killed before 5 s in the poll loop, or the wait-log path wasn’t hit (e.g. different code path or log not writable).

**Host (daemon.log):** Stub showed CUDA DOORBELL RING and “CUDA result applied” for seq 1–38 (call_id 0x0032 = HtoD). So the host **did** receive and apply 38 results for this run. Mediator stats: Total processed 139 → 144.

**Conclusion:** Reply path (mediator → stub → BAR) is working on the host. The **server-side** load timeout (default ~5 min) stopped the run before we could use the full 40 min. For a true 40-min test we must start the server with **`OLLAMA_LOAD_TIMEOUT=40m`** (or use the systemd service which has `OLLAMA_LOAD_TIMEOUT=20m`).

**Next:** Re-run with `OLLAMA_LOAD_TIMEOUT=40m` in the server env and `curl -m 2400` so the load can run up to 40 min and we can see whether the guest eventually sees DONE and completes, or remains stuck (and whether `vgpu_poll_wait.log` gets entries).

---

## 11. Full 40-minute run with OLLAMA_LOAD_TIMEOUT=40m (Mar 16)

**Setup:** Server started with **OLLAMA_LOAD_TIMEOUT=40m** and vGPU env; client `curl -m 2400` (40 min). Model: llama3.2:1b.

**Result:** **Client** hit 40-minute timeout: `HTTP_CODE:000 TIME:2400.000033s EXIT=28`. The server did **not** abort at 5 min; the load never completed within 40 min.

**Guest:**
- **vgpu_call_sequence.log:** **295 lines** — init, alloc, then many `cuMemcpyHtoD_v2`; last line = HtoD. So the guest sent **295 RPCs** and was blocked waiting for the response to the last one (or one of the last).
- **vgpu_poll_wait.log:** **File not created** (wc/ls failed). So the 5s “waiting for response” log was never written — either the poll-loop logging path wasn’t hit, or the file couldn’t be created (path/permissions).
- **ollama_run.log:** No “Load failed” or “progress” lines in the grep; load was still in progress when the client timed out.

**Host (daemon.log):** Stub applied **“CUDA result applied”** for seq up to **292** (and DOORBELL RING for 292) in the tail we read. So the host received and applied at least 292 results; the last few (293–295) may have been applied just after the tail or the guest is stuck on one of them.

**Conclusion:** Over 40 minutes the guest sent **295** HtoD (and init/alloc) RPCs; the host applied **292+** results. So the **reply path is working** (mediator → stub → BAR). The guest nevertheless **never completed** the load: it stayed blocked in the transport (waiting for a response the host has already written). So the **guest’s MMIO read of REG_STATUS is not observing DONE** for at least one call, even with the guest-side memory barrier. Likely cause: **cross-thread visibility** in QEMU — the stub’s fd handler (main loop) sets `status_reg` and the vCPU thread runs the MMIO read callback; without a release/acquire pair the vCPU may not see the update.

---

## 12. Stub release/acquire fence (Mar 16) — next step applied

**Change:** In `src/vgpu-stub-enhanced.c`:
- **VGPU_STATUS_WRITE(s, v):** macro that sets `s->status_reg = (v)` then `__atomic_thread_fence(__ATOMIC_RELEASE)` so the write is visible to the vCPU thread.
- All CUDA result and legacy response paths that set `status_reg` (DONE/ERROR) now use `VGPU_STATUS_WRITE`.
- In **vgpu_mmio_read**, for **VGPU_REG_STATUS** we do `__atomic_thread_fence(__ATOMIC_ACQUIRE)` before returning `s->status_reg` so the vCPU thread observes the latest value.

**Rationale:** In QEMU with multiple threads, the fd handler runs on the main/iothread and the MMIO read runs on the vCPU thread. Plain stores/loads do not guarantee cross-thread visibility; release (writer) and acquire (reader) fences ensure the guest’s poll loop sees DONE after the stub applies the result.

**What you need to do:** Rebuild the vGPU stub on the host and restart the VM (see **HOST_STUB_REBUILD_INSTRUCTIONS.md**). Then re-run a 40-min generate (server with `OLLAMA_LOAD_TIMEOUT=40m`, client `curl -m 2400`). If the guest now sees DONE and the load completes, the fix is confirmed.

---

## 13. Post–release/acquire fence run (Mar 16)

**Setup:** Stub rebuilt and installed with VGPU_STATUS_WRITE + acquire fence in MMIO read; mediator restarted; VM (Test-4) started. Server with `OLLAMA_LOAD_TIMEOUT=40m`, client `curl -m 2400`, model llama3.2:1b.

**Result:** Same as §11: **client hit 40-minute timeout** (`HTTP_CODE:000`, `TIME≈2400s`, `EXIT=28`). Load did not complete.

**Guest:** 295 lines in `vgpu_call_sequence.log` (last = cuMemcpyHtoD_v2). `vgpu_poll_wait.log` still not created (no 5s wait entries). No "Load failed" or "progress" in ollama_run grep; load still in progress when client timed out.

**Host:** daemon.log shows stub applied "CUDA result applied" through **seq=292** (qemu-dm-224). Mediator: Total processed **300**.

**Conclusion:** The release/acquire fence did **not** resolve the issue. Guest still does not observe DONE within 40 min despite the stub applying 292+ results. Possible reasons: (1) QEMU on this host may not use a separate vCPU thread for MMIO, so the fence has no effect; (2) guest MMIO read path may be different (e.g. cached or not hitting the callback); (3) another ordering or visibility issue (e.g. Xen/QEMU MMIO handling). **Next options:** add guest-side logging that does not depend on file create (e.g. stderr-only every poll, or a known-writable path); try a short `sched_yield()` or `nanosleep` in the guest poll loop to change scheduling; or explore stub→guest notification (e.g. interrupt or explicit kick) instead of poll-only.

---

## 14. Actual error: guest reads BUSY (0x01) only (Mar 16)

**Diagnostic:** Guest transport now logs the **value read from REG_STATUS** every 1s (and once at first poll) to `$HOME/vgpu_status_poll.log`. A 90s run showed:

- **status=0x01** only (6 lines; seq=1). So the guest **never** sees 0x02 (DONE); it only ever sees **0x01 (BUSY)**.

**Conclusion:** The stub sets DONE (host log shows "CUDA result applied") but the guest’s MMIO read never returns DONE. So either (1) the vCPU thread never observes the stub’s write (memory ordering), or (2) something else overwrites or shadows the register. Fences alone did not fix it.

**Fix applied (stub):** All `status_reg` writes use **VGPU_STATUS_WRITE** (plain store + `__sync_synchronize()`); MMIO read uses **VGPU_STATUS_READ** (full barrier + load) so the vCPU sees the fd-handler write. Guest poll interval increased to 10 ms to avoid starving the iothread.

**Post-fix test (same failure):** After stub rebuild and VM restart, 90s run still showed guest **status=0x01 only** (host applied seq=1..17). So the guest’s MMIO read still does not see DONE. Conclusion: either (1) the guest’s read and the stub’s write target different state (e.g. two devices, or Xen/QEMU BAR mapping), or (2) the MMIO read path does not hit our handler. **Next:** Add stub-side log in `vgpu_mmio_read` when returning STATUS (log the value returned) to confirm the handler is called and what it returns; if stub logs “returning status=2” but guest sees 0x01, the fault is outside the stub (Xen/KVM path).

---

## 15. Stub returns DONE but guest sees BUSY (Mar 16)

**Test:** Stub rebuilt with “STATUS read returning 0x%x” when value is DONE/ERROR. 90s generate run; host daemon.log and guest vgpu_status_poll.log checked.

**Host (daemon.log):** **31** lines “STATUS read returning 0x2” (and a few “0x3”). So the stub’s `vgpu_mmio_read` **is** being called and **is returning DONE (0x2)** to the MMIO read.

**Guest (vgpu_status_poll.log):** **Only status=0x01** (BUSY). So the guest never sees 0x02.

**Conclusion:** The **stub returns 0x2 (DONE)** but the **guest receives 0x01 (BUSY)**. The bug is **not** in our stub or memory ordering; it is in the **path between QEMU’s MMIO read return and the guest**: Xen device model, KVM, or a second/cached BAR mapping so the guest’s read does not get the value our handler returned. **Next:** See **MMIO_MISMATCH_CAUSE_DIAGNOSIS.md** for possible causes (BAR1 not reaching stub, value corrupted in path, two devices, caching), correlation steps (stub logs BAR0/BAR1 status reads; guest logs which BAR and value), and how to narrow down the cause.

---

## 16. Host logs read via connect_host.py (Mar 16)

**Commands used (from phase3, as in §9):**
- Host daemon.log: `python3 connect_host.py "grep -a -E 'CUDA result|DOORBELL RING|STATUS read|ERROR|0x00ae|0x00b5' /var/log/daemon.log | tail -100"`
- Host mediator: `python3 connect_host.py "tail -200 /tmp/mediator.log | strings | grep -a -E 'FAILED|ERROR|vm=9|STATS|processed'"`

**Findings:**
- **daemon.log:** For vm_id=9, all CUDA results show **status=0 (DONE)**. Call IDs 0x0032 (HtoD), 0x0030 (cuMemAlloc), 0x0001 (cuInit), 0x00ac, **0x00ae**, **0x00b5** all have “CUDA result applied seq=N status=0 (DONE)” and “STATUS read returning 0x2”. **No status=3 (ERROR)** in the tail.
- **mediator.log:** HtoD progress for vm=9 up to >6 GB; cuMemAlloc SUCCESS for vm=9; no FAILED/ERROR in tail.
- **VM after 500 run:** `/tmp/vgpu_last_error` and `/tmp/vgpu_debug.txt` **not created** (checked with sudo). So the runner exit status 2 is **not** from our transport error path (timeout, STATUS_ERROR, or CUDA_CALL_FAILED), which would write those files.
- **vgpu_call_sequence.log:** Last lines 0x00b5, 0x00ae, 0x00b5 — many RPCs complete; BAR1 status mirror is in use and host is replying.

**Conclusion:** Host is returning DONE for all calls including 0x00ae/0x00b5. Exit status 2 is likely from the **runner/application layer** (Ollama or llama.cpp), not from our transport returning 2. **Next:** Capture runner stderr (e.g. `journalctl -u ollama` or run with `OLLAMA_DEBUG=1`) when generate returns 500 to see the actual message; or inspect Ollama/llama.cpp for what causes exit(2).

---

## 9. Commands used

- Mediator check:  
  `python3 connect_host.py "pgrep -af mediator; ps aux | grep -E 'mediator|phase3' | grep -v grep"`
- VM call sequence:  
  `python3 connect_vm.py "tail -80 /tmp/vgpu_call_sequence.log"`
- Host HtoD/vm=9:  
  `python3 connect_host.py "grep -a 'cuMemcpyHtoD\|vm=9\|vm_id' /tmp/mediator.log | tail -100"`
- Host last activity:  
  `python3 connect_host.py "tail -200 /tmp/mediator.log | strings | grep -a -E 'HtoD|cuMemcpy|vm=9|processed|STATS'"`
