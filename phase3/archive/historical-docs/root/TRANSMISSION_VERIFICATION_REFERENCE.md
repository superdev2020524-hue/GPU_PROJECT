# Transmission verification: earlier report vs current issue

## Earlier report (no other issues with transmission)

Several Phase 3 docs state that the **transmission path** or **end-to-end path** was working:

- **END_TO_END_VERIFICATION_SUCCESS.md** (Mar 1, 2026): “CONFIRMED: The complete end-to-end path is working!” VM logs showed **round-trip**: `RECEIVED from VGPU-STUB: call_id=0x0030 seq=1 status=DONE`. So the guest **did** receive a response for at least one call (cuMemAlloc).
- **PHASE3_VGPU_CURRENT_STATUS.md**: “Transport path … is **working**. Allocations and HtoD copies succeed.”
- **HOST_VERIFICATION_COMPLETE.md**: “End-to-end communication path is working.”

So the earlier verification was **not** wrong: it confirmed that **at least one full round-trip** (request → mediator → stub → BAR → guest sees DONE) works. The **transmission method** is the same then and now (BAR0, doorbell, poll REG_STATUS).

## Why we see a problem now

The current failure appears when:

- The guest sends **many** HtoD requests in sequence (dozens of round-trips during model load).
- The **host side** is fine: mediator processes HtoD, stub receives the response and applies it (daemon.log: “CUDA result applied seq=N status=0 (DONE)”).
- The **guest** still blocks on one of those responses (last line in `vgpu_call_sequence.log` = cuMemcpyHtoD_v2).

So the **reply path** (mediator → stub → BAR) is working on the host; the open question is why the **guest** sometimes does not see the updated REG_STATUS for a given response. Possibilities:

1. **Timing:** The run is aborted (e.g. client timeout) before that response arrives, so the guest never gets to see DONE for that seq.
2. **Guest read of BAR:** The guest’s MMIO read of REG_STATUS might not see the stub’s write (e.g. ordering/cache; adding a memory barrier before the read is a low-cost check).
3. **Different scenario:** Earlier verification may have been with a single or few calls; under many back-to-back round-trips something (e.g. QEMU/MMIO visibility) might behave differently.

## Summary

| Aspect | Earlier report | Current finding |
|--------|----------------|-----------------|
| **Transmission method** | Same (BAR0, doorbell, poll STATUS) | Same |
| **Single round-trip** | Verified (RECEIVED status=DONE) | — |
| **Many HtoD round-trips** | Not explicitly verified | Guest blocks on one response; host stub applies it |
| **Conclusion** | Report was correct for what was tested | Issue appears under sustained HtoD load; next: ensure guest sees BAR update (e.g. barrier, longer timeout). |
