# MMIO status read workaround: use response_len

## Problem

The guest's MMIO read of the status register (BAR0 0x004 or BAR1 last 4 bytes) returns **0x01 (BUSY)** even when the stub has set **0x02 (DONE)**. The mediator sends completions and the stub applies them (host log shows "BAR1 status read -> 0x2"), but the value delivered to the guest is wrong (Cause B/D in MMIO_MISMATCH_CAUSE_DIAGNOSIS.md).

## Workaround

**Note:** An earlier variant (“Fix 1”) checked `response_len` after **3** poll iterations; it was **reverted** because it broke long-duration transmission (guest could exit on stale data). The correct value is **30** (see VERIFICATION_REPORT, HtoD_DIAGNOSIS_RESULTS).

Use a **different register** that may be delivered correctly: **BAR0 response_len (0x01C)**. The guest's poll loop already had a fallback that checks `REG_RESPONSE_LEN` after 30 iterations; we:

1. **Stub:** When applying a CUDA result (`VGPU_MSG_CUDA_RESULT`), set `s->response_len = 1` so the guest can detect completion by reading BAR0+0x01C. Clear `s->response_len = 0` when starting a new request (doorbell) so the guest does not see a stale value.
2. **Guest:** Check `response_len` after **30** poll iterations (existing fallback); do not reduce to 3 — early check can break transmission (stale response_len causes early exit and corrupts state; see VERIFICATION_REPORT and long-duration runs).

If the BAR0 read at 0x01C is also corrupted by the same Xen/qemu-dm path, this workaround will not help; then an interrupt-based completion would be the next option.

## Code changes

- **phase3/src/vgpu-stub-enhanced.c**
  - In the `VGPU_MSG_CUDA_RESULT` branch, after setting timestamp: `s->response_len = 1;`
  - In the CUDA doorbell path (before sending to mediator): `s->response_len = 0;`
- **phase3/guest-shim/cuda_transport.c**
  - Keep `if (poll_iter >= 30)` for the `REG_RESPONSE_LEN` fallback. Do **not** use 3 — that choice was reverted (broke long-duration transmission; see VERIFICATION_REPORT and HtoD_DIAGNOSIS_RESULTS).

## Deployment

- **Guest:** Rebuild the transport shim and reinstall on the VM (e.g. `make guest`, copy `libvgpu-cuda.so.1` to `/opt/vgpu/lib/`, restart ollama). No host changes required for the guest part.
- **Stub:** The stub is part of QEMU. Rebuilding the stub requires rebuilding the QEMU device model that includes `vgpu-stub-enhanced.c` and restarting the VM or the QEMU process. See **HOST_STUB_REBUILD_INSTRUCTIONS.md** or your host QEMU build process. If you cannot rebuild QEMU on the host, only the guest change can be deployed (then the fallback will never see `response_len != 0` and the workaround will not take effect until the stub is updated).

## Verification

After deploying both stub and guest:

1. Trigger a generate from the VM (e.g. `ollama run llama3.2:1b "Hi"` or curl to `/api/generate`).
2. The runner should complete (model load and inference) instead of blocking on HtoD.
3. In the guest log, the transport may still log `status=0x01` from the status register, but the loop will exit when `response_len != 0` after 30 or more poll iterations (do not use fewer; see above).

If the guest still blocks, BAR0 at 0x01C may be subject to the same MMIO delivery bug; then the next step is to try MSI-X or legacy interrupt for completion.
