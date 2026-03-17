# Finding the cause of guest MMIO reads not matching host returns

*Stub returns DONE (0x2) but guest consistently sees BUSY (0x01). This doc lists possible causes and how to narrow them down.*

---

## Known facts (from HtoD_DIAGNOSIS_RESULTS.md, ACTUAL_ERROR_MAR16)

1. **Stub side:** When the stub's MMIO read handler runs for the status register, it returns **0x2 (DONE)** and the host log shows "STATUS read returning 0x2" (and sometimes 0x3 ERROR).
2. **Guest side:** The guest's poll loop **never** sees 0x2; it only ever sees **0x01 (BUSY)** in both:
   - `[cuda-transport] poll ... status=0x01` (and in `vgpu_status_poll.log`).
3. **BAR0 and BAR1 both tried:** Forcing the guest to read status only from BAR0 still showed 0x01. Using the BAR1 status mirror (last 4 bytes of BAR1) also showed 0x01. So the mismatch is not specific to one BAR.
4. **Conclusion so far:** The bug is in the **path between QEMU's MMIO read return and the guest** (Xen device model, BAR delivery, or a second/cached mapping).

---

## Possible causes

### A. Guest and stub are not the same read path

- **Hypothesis:** The guest's read (e.g. from BAR1) might **not** trigger the stub's read handler at all. For example, Xen might expose BAR1 as a **direct-mapped** region (grant/shared frame) that the guest reads from, while QEMU's `vgpu_bar1_read` backs a different view. So we update `bar1_status_mirror` in the stub, but the guest reads from another memory that is never updated.
- **Check:** After a run, compare:
  - **Stub:** Do we see **any** "BAR1 status read -> 0x2" in the host log? (Stub now logs BAR1 status when value is DONE/ERROR.)
  - **Guest:** Does the guest report `from=BAR1` and `status=0x01`?
  - If the stub **never** logs a BAR1 status read for DONE/ERROR but the guest is polling BAR1, then the guest's BAR1 reads are **not** reaching our handler → likely Xen maps BAR1 (or the last 4 bytes) to a different backing store.

### B. BAR0: same path but value corrupted

- **Hypothesis:** The guest's read of BAR0 offset 0x004 **does** reach the stub; the stub returns 0x2; but the value is **corrupted or overwritten** before the guest CPU sees it (e.g. in Xen or in the mechanism that injects the MMIO read result).
- **Check:** Stub logs "BAR0 STATUS read -> 0x2" when it returns DONE. If the guest at the same time reports `from=BAR0 status=0x01`, then the same logical read returns 0x2 from the stub but 0x1 to the guest → corruption or wrong delivery in the path.

### C. Two devices or two BAR views

- **Hypothesis:** There are two vGPU devices (or two views of the same BAR). The stub we log is one device; the guest might have opened and mmap'd the **other** device. So we set DONE on device A and log it; the guest reads from device B and sees BUSY.
- **Check:** Confirm the guest uses a **single** PCI device (one BDF) for both resource0 and resource1. `cuda_transport.c` uses the first matching VGPU device and the same `pci_path` for res0 and res1, so there should be only one device. If the VM has two vGPU devices, the guest could still pick the first one; verify on the VM that `lspci` shows only one vGPU device.

### D. Cached or stale MMIO region

- **Hypothesis:** The guest kernel (or Xen) caches the MMIO region. So the first read goes to QEMU and gets 0x2, but subsequent reads are served from a cache that still holds 0x01.
- **Check:** Harder to prove from logs. If correlation shows the stub is called and returns 0x2 but the guest sees 0x01, and we rule out two devices, caching in the MMIO path is a candidate (would require kernel/Xen or QEMU expertise to fix).

---

## Correlation steps (what was added)

### 1. Stub (host)

- **BAR0:** When the BAR0 MMIO read handler returns **DONE (0x2)** or **ERROR (0x3)**, it logs:  
  `[vgpu] vm_id=N: BAR0 STATUS read -> 0xN`
- **BAR1:** When the BAR1 read handler returns the status mirror as **DONE (0x2)** or **ERROR (0x3)**, it logs:  
  `[vgpu] vm_id=N: BAR1 status read -> 0xN`  
  and appends a line to `/tmp/vgpu_stub_bar1_done.log` when value is DONE.

So:

- If you see **BAR0** lines but **no BAR1** lines for the same run, and the guest reports `from=BAR1`, then the guest's BAR1 status reads are **not** reaching the stub (supports cause **A**).
- If you see **BAR0 STATUS read -> 0x2** (and/or BAR1 -> 0x2) while the guest log shows `status=0x01`, then the value is wrong on delivery (supports **B** or **D**).

### 2. Guest (VM)

- The transport logs:  
  `[cuda-transport] poll call_id=0x... seq=N iter=M status=0xNN from=BAR0` or `from=BAR1`  
  and writes the same to `$HOME/vgpu_status_poll.log` (or `/tmp/vgpu_status_poll.log` if HOME is not set).

So you can:

- Confirm which BAR the guest is using (`from=BAR0` vs `from=BAR1`).
- Correlate with host: for the same time window, does the stub see a read on that BAR with DONE?

### 3. Commands to run

**On host (after a generate attempt):**

```bash
# Stub log (QEMU stderr → daemon.log)
grep -a 'BAR0 STATUS read\|BAR1 status read' /var/log/daemon.log | tail -100
# BAR1 DONE log file (if any BAR1 read returned DONE)
cat /tmp/vgpu_stub_bar1_done.log | wc -l
```

**On VM (after the same run):**

```bash
# Which BAR and what value the guest saw
grep -E 'status=0x|from=BAR' $HOME/vgpu_status_poll.log 2>/dev/null || grep -E 'status=0x|from=BAR' /tmp/vgpu_status_poll.log
# Or from journal if transport logs to stderr
journalctl -u ollama -n 200 --no-pager | grep -E 'cuda-transport.*poll|status=0x|from=BAR'
```

**Interpretation:**

- **No BAR1 lines on host, guest uses BAR1:** Guest BAR1 reads do not hit the stub → investigate how Xen exposes BAR1 to the guest (trap-and-emulate vs direct map).
- **Host shows BAR0/BAR1 -> 0x2, guest shows status=0x01:** Same read returns different value to guest → bug in Xen/qemu-dm MMIO result delivery or caching.

---

## Next steps depending on result

1. **BAR1 reads never hit the stub (no BAR1 status read logs):**  
   Focus on **how Xen exposes PCI BAR1** to the guest. If BAR1 (or the last page) is direct-mapped to a frame that QEMU does not back with our `vgpu_bar1_read`, we need either to change that mapping so reads go through QEMU, or to use a completion mechanism that does reach the guest (e.g. interrupt, or a different region that is trap-and-emulate).

2. **BAR0/BAR1 reads hit the stub and return 0x2 but guest sees 0x01:**  
   Focus on the **Xen/qemu-dm path** that delivers the MMIO read result to the guest: possible caching, wrong register, or corruption. May require Xen or QEMU device model expertise or code review.

3. **Two vGPU devices present:**  
   Ensure the guest uses the same device for both BAR0 and BAR1 and that the stub we are logging is that device (e.g. by vm_id and single qemu-dm process per VM).

---

## Files changed

- **Stub:** `phase3/src/vgpu-stub-enhanced.c`  
  - BAR0: log when status read returns DONE/ERROR.  
  - BAR1: log when status mirror read returns DONE/ERROR; keep append to `/tmp/vgpu_stub_bar1_done.log` on DONE.
- **Guest:** `phase3/guest-shim/cuda_transport.c`  
  - Poll log and status file now include `from=BAR0` or `from=BAR1`.

Rebuild and deploy the stub on the host (see **HOST_STUB_REBUILD_INSTRUCTIONS.md**) and the guest transport (e.g. **transfer_cuda_transport.py**), then run one generate and collect the logs above to decide which cause applies.
