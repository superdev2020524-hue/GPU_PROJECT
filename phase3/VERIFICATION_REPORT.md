# Verification report (authority-granted checks)

**Date:** 2026-03-17

---

## 1. Guest shim (VM) — VERIFIED

- **Source:** `grep -n 'poll_iter >= 30' /home/test-4/phase3/guest-shim/cuda_transport.c` → **line 1153** (response_len fallback; correct value 30, not 3).
- **Deployed binary:** `/opt/vgpu/lib/libvgpu-cuda.so.1` — 246880 bytes, timestamp Mar 17 09:13 (after `transfer_cuda_transport.py`).
- **Service:** `ollama.service.d/vgpu.conf` sets `LD_PRELOAD=.../libvgpu-cuda.so.1`, `LD_LIBRARY_PATH=.../cuda_v12:/opt/vgpu/lib:...`, `OLLAMA_LLM_LIBRARY=cuda_v12`, `OLLAMA_NUM_GPU=1`.
- **Main process:** `ollama.bin.real serve` has LD_PRELOAD and LD_LIBRARY_PATH with vGPU libs.

**Update (post-revert):** The guest shim was reverted to **poll_iter >= 30** (was wrongly changed to 3). Checking at 3 iterations broke long-duration transmission (40-min runs with 295 HtoD RPCs had been possible; after the change only 12 init/context calls appeared and no HtoD). Early response_len check can exit on stale data and corrupt state. Restored value: **30**.

---

## 2. Mediator (host) — VERIFIED

- **Process:** `pgrep -a mediator_phase3` → PID 3551590, running.
- **Log:** `/tmp/mediator.log` contains **18** lines `[MEDIATOR] CUDA result sent vm_id=9 ... -> stub sets DONE` (from earlier runs).
- **Conclusion:** Mediator is running and has logged CUDA completions for vm_id=9 in the past.

---

## 3. Generate and data path — NOT COMPLETING

- **Tests:** Multiple generate requests (45s, 110s, 240s, 100s timeout); all ended with **curl exit 28** (timeout), **0-byte response**.
- **vgpu_call_sequence.log (VM):** 12 lines — only `cuInit`, `cuGetGpuInfo`, `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`. No `cuMemAlloc` or `cuMemcpyHtoD_v2`.
- **vgpu_status_poll.log (VM):** Empty in all runs (no poll-loop log lines).
- **Mediator count:** Before and after a 100s generate run: **18** both times. No new “CUDA result sent” lines during the generate.

**Conclusion:** During a generate, (1) the guest reaches only init/context calls in the log, (2) no HtoD calls appear, (3) the mediator receives no new CUDA requests from this VM in that window. So either the stub is not forwarding current guest traffic to the mediator, or the runner that handles the generate is not using the vGPU transport (e.g. different process or different library resolution).

---

## 4. Ollama GPU discovery — VERIFIED

- **Journal:** `msg="inference compute" ... library=CUDA compute=8.9 name=CUDA0 description="NVIDIA H100 80GB HBM3"`.
- **Conclusion:** Ollama reports CUDA and H100; discovery sees the vGPU.

---

## 5. Summary

| Item                         | Status   | Note |
|-----------------------------|----------|------|
| Guest shim (response_len)   | Reverted | poll_iter restored to 30; deploy to VM when possible. |
| Mediator running            | Verified | PID 3551590; log has 18 CUDA result lines. |
| Stub (QEMU)                 | Not checked here | Rebuilt and VM rebooted by user; no re-verify of stub binary. |
| Generate completes          | No       | Timeout, 0-byte response. |
| New CUDA traffic to mediator| No       | Count unchanged (18) during generate. |
| Guest call sequence         | Partial  | Only init/context; no alloc/HtoD. |

**Next steps (suggested):** (1) Confirm on the host that the VM’s QEMU process is using the new stub (e.g. from the rebuilt RPM). (2) Confirm stub→mediator connection for this VM (e.g. mediator log or stub debug for “connected” / “send”). (3) Confirm which process (serve vs runner) uses the vGPU shim and whether the runner inherits LD_LIBRARY_PATH so it loads `/opt/vgpu/lib/libvgpu-cuda.so.1` for libcuda.
