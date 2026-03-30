# Actual errors (Mar 16): status not visible to guest + shmem H2G map failed

## Goal and symptom

- **Phase 1:** Ollama in VM does GPU inference end-to-end (model load via vGPU, inference on host H100, response to VM).
- **Symptom:** Generate returns **HTTP 500** in ~7–10 s with `"llama runner process has terminated: exit status 2"`.

---

## Error 1: Guest never sees DONE when polling status (root cause of exit 2)

### What we proved

1. **Guest** logs: `[cuda-transport] poll ... status=0x01` (BUSY) for every poll, and never sees `status=0x02` (DONE), so it never leaves the poll loop.
2. **Host** (daemon.log): For the same vm_id=9 and the same run, the stub logs **`STATUS read returning 0x2`** after each `CUDA result applied ... (DONE)`.
3. We forced the guest to **always read status from BAR0** (no BAR1). Result: guest **still** saw `status=0x01` only.
4. With **BAR1 mirror** for status: guest reported `status_from=BAR1` but still saw `status=0x01` on every poll.

### Conclusion

- The **stub** does set and return **DONE (0x02)**; the host log shows that BAR0 status reads return 0x2.
- The **guest** MMIO read of the status register **does not** receive that value: it consistently sees **0x01 (BUSY)**.
- So the bug is in the **Xen/qemu-dm path**: the value returned by the stub’s MMIO read handler is not the value the guest CPU sees (e.g. wrong device, caching, or different BAR handling). This matches **HtoD_DIAGNOSIS_RESULTS.md** (§14–15): BAR0 status read is broken on this stack (guest sees BUSY).
- **BAR1** status mirror does **not** fix it in practice: either BAR1 reads do not reach the stub, or the same visibility issue applies to BAR1 on this platform.

So the **immediate** cause of “exit status 2” is: the runner stays in the transport poll loop (always seeing BUSY), hits a higher-level timeout (e.g. Ollama load timeout), and exits with status 2.

---

## Error 2: Shmem registration fails on host (H2G map)

### What we saw

- **Host** log: `[vgpu] vm_id=9: cpu_physical_memory_map(H2G) failed (gpa=0x667f9000 len=134217728)`.
- **Guest** log: `[cuda-transport] vgpu-stub rejected shmem registration (status=0x03 err=0x00000001:INVALID_REQUEST) — using BAR1`.

So when the guest tries to register shared memory (256 MB: 128 MB G2H + 128 MB H2G), the stub succeeds in mapping the **G2H** half but **cpu_physical_memory_map** fails for the **H2G** half (gpa+128M, len=128 MB). The stub then returns ERROR (0x03) and the guest falls back to BAR1 for data. That fallback is why the run can still send requests (e.g. cuMemAlloc, HtoD), but the **status** visibility issue above still prevents the guest from seeing DONE.

### Likely cause

- In Xen/qemu-dm, **cpu_physical_memory_map** for a second region (H2G at gpa+size/2) may fail (e.g. mapping policy, size, or GPA layout).
- Fix ideas: map the whole 256 MB in one go and split the host pointer; or try smaller chunks; or check Xen-specific mapping APIs for guest physical memory.

---

## What to do next (to reach Phase 1)

1. **Status visibility (blocking)**  
   Find a way for the guest to observe DONE:
   - **Option A:** Fix or work around the Xen/qemu-dm MMIO read path so that the guest’s read of the status register (BAR0 or BAR1 mirror) returns the value the stub wrote (e.g. Xen/qemu experts, or a different signalling mechanism that does propagate).
   - **Option B:** Use a different completion mechanism that does reach the guest (e.g. interrupt, or a different memory region that is known to be visible).

2. **Shmem (secondary)**  
   Fix **cpu_physical_memory_map(H2G)** so 256 MB shmem registration succeeds; then the guest can use shmem instead of BAR1 for data. This may improve performance and avoid BAR1 data path issues, but it does not fix the status visibility bug by itself.

3. **Stub debug (optional)**  
   The stub was changed to append a line to `/tmp/vgpu_stub_bar1_done.log` when the BAR1 mirror read returns DONE. After rebuilding the stub on the host and reproducing a run, check that file:
   - If it has lines → BAR1 reads reach the stub and return 0x02; the bug is between stub and guest.
   - If it is empty → BAR1 reads may not be reaching the stub (e.g. BAR1 mapped elsewhere by Xen).

---

## Files and commands

- **Guest transport:** `phase3/guest-shim/cuda_transport.c` — poll loop, BAR1 mirror, init logs (`status_from=BAR1` / BAR0).
- **Stub:** `phase3/src/vgpu-stub-enhanced.c` — BAR1 mirror, `bar1_status_mirror`, optional BAR1-done file log.
- **Host logs:** `python3 connect_host.py "grep -a 'vm_id=9\|STATUS read' /var/log/daemon.log | tail -80"`.
- **VM journal:** `python3 connect_vm.py "journalctl -u ollama -n 300 --no-pager"`.
- **Deploy guest:** `python3 transfer_cuda_transport.py`.
