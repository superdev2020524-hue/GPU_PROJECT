# Reply-path debug (Mar 16, 2026)

## Changes made

### 1. Guest transport (`guest-shim/cuda_transport.c`) — **deployed to VM**

- **Periodic log:** While polling for `STATUS_DONE`/`STATUS_ERROR` after ringing the doorbell, the guest now logs every **5 seconds** to stderr:
  - `[cuda-transport] waiting for response call_id=0xXXXX seq=N (elapsed M s)`
- **File log:** The same info is appended to **`/tmp/vgpu_poll_wait.log`** (one line per 5s: `timestamp call_id=0xXXXX seq=N elapsed=M`).

So if the guest is stuck waiting for a response, you will see:
- Repeated "waiting for response" lines on stderr (or in the process log), and
- Growing `/tmp/vgpu_poll_wait.log` on the VM.

After a run, on the VM:
```bash
cat /tmp/vgpu_poll_wait.log
# or
strings /tmp/ollama_run.log | grep "waiting for response"
```

### 2. Stub (`src/vgpu-stub-enhanced.c`) — **host rebuild required**

**Rebuild steps:** See **HOST_STUB_REBUILD_INSTRUCTIONS.md** for copy commands to the host, build, and install.

- When the stub receives **VGPU_MSG_CUDA_RESULT** from the mediator:
  - If **seq matches** `pending_seq`: logs  
    `[vgpu] vm_id=X: CUDA result applied seq=Y status=Z (DONE)`
  - If **seq does not match**: logs  
    `[vgpu] vm_id=X: CUDA result IGNORED (recv seq=Y pending_seq=Z) — guest will keep waiting`

These go to **QEMU’s stderr** on the host (where the VM is running). To see them you must:
1. Rebuild the vGPU stub (or the component that includes `vgpu-stub-enhanced.c`).
2. Restart the VM or the QEMU process so the new stub is loaded.
3. Capture QEMU stderr (e.g. journalctl for the VM, or redirect QEMU’s stderr to a file).

If you see "CUDA result applied" for the same seq the guest is waiting on, the reply is reaching the stub and the bug may be guest BAR read or timing. If you see "IGNORED" with seq mismatch, responses are out of order or stale. If you see no stub log at all for that request, the response is not reaching the stub (mediator→stub path).

### 3. Stub release/acquire fence (Mar 16)

To fix cross-thread visibility so the guest’s MMIO read of REG_STATUS sees DONE, the stub was updated to use **release** fences after every `status_reg` write (DONE/ERROR) and an **acquire** fence in the MMIO read handler for the status register. Rebuild the stub on the host and restart the VM for this to take effect. See **HtoD_DIAGNOSIS_RESULTS.md** §12 and **HOST_STUB_REBUILD_INSTRUCTIONS.md**.

---

## How to run a quick test (guest-only)

1. On the VM, clear the wait log and trigger a generate:
   ```bash
   rm -f /tmp/vgpu_poll_wait.log /tmp/vgpu_call_sequence.log
   # trigger generate (e.g. curl to /api/generate or run your script)
   ```
2. After the run hangs or times out:
   ```bash
   cat /tmp/vgpu_poll_wait.log
   tail -20 /tmp/vgpu_call_sequence.log
   ```
3. If `vgpu_poll_wait.log` has many lines with the same `call_id`/`seq` and growing `elapsed`, the guest is stuck in the poll loop waiting for that response (reply not seen or not delivered).

---

## Host stub log location (XCP-ng, read-only check)

On the host, stub messages (CUDA result applied / IGNORED, DOORBELL RING) go to **`/var/log/daemon.log`** (process name `qemu-dm-<domain_id>`, e.g. `qemu-dm-223` for Test-4). To verify if CUDA results are being applied:

```bash
grep -a 'CUDA result\|DOORBELL RING' /var/log/daemon.log | tail -50
```

If you see `[vgpu] vm_id=9: CUDA result applied seq=N status=0 (DONE)` for the same seq the guest is waiting on, the reply is reaching the stub and being written to BAR; the remaining issue may be guest-side (e.g. BAR read timing or cache).
