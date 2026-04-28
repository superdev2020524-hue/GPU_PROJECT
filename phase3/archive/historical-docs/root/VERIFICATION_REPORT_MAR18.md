# Verification report: VM and host status vs documentation

*Verified: Mar 18, 2026 — using granted permissions (VM: full; host: read-only).*

---

## 1. Permissions used

- **VM (test-4):** Commands run via `connect_vm.py` (SSH to test-4@10.25.33.12).
- **Host:** Log read via `connect_host.py` (SSH to root@10.25.33.10); no edits, no build/restart.

---

## 2. VM status (test-4)

| Check | Result | Matches docs? |
|------|--------|----------------|
| **Ollama service** | `active` | ✓ |
| **API** | `GET /api/tags` → HTTP 200 | ✓ |
| **GPU mode** | **Yes.** Journal: `inference compute` with `library=CUDA`, `description="NVIDIA H100 80GB HBM3"`, `total="80.0 GiB"`, `total_vram="80.0 GiB"`. | ✓ (CURRENT_STATE, ERROR_TRACKING) |
| **Discovery** | No log line containing "filtering device which didn't fully initialize". | ✓ (Issue A resolved) |
| **Refresh message** | Present: `msg="unable to refresh free memory, using old values"` (runner.go:356). | ✓ (REFRESH_AND_GPU_DETECTION_INVESTIGATION) |
| **Running binary** | `/usr/local/bin/ollama.bin.new serve` (PID 17365). Also present: `ollama.bin`, `ollama.bin.real` (same size/date as .new). | — |
| **cuMemAlloc_called.log** | **Missing** (file not present under `/tmp/`). | ✓ (docs: "empty" / load runner never calls cuMemAlloc) |
| **vgpu_call_sequence.log** | 168 lines; only call IDs **0x0001, 0x00f0, 0x0090, 0x0022** (cuInit, cuGetGpuInfo, cuDevicePrimaryCtxRetain, cuCtxSetCurrent). **Zero** lines with 0x0030 or 0x0032. | ✓ (docs: no alloc/HtoD from load runner) |
| **vgpu_host_response_verify.log** | Present (19829 bytes). | — |

**Conclusion (VM):** Ollama is operating in **GPU mode**. Discovery shows CUDA and H100 80 GiB; the "unable to refresh free memory" message still appears; the load runner does **not** reach cuMemAlloc or cuMemcpyHtoD (no 0x0030/0x0032). Status **matches** ERROR_TRACKING_STATUS.md and PHASE3_REVIEW_AND_RESUME.md.

---

## 3. Host status (mediator)

| Check | Result | Matches docs? |
|------|--------|----------------|
| **Mediator log** | `/tmp/mediator.log` readable. Heartbeats and stats present. | ✓ |
| **Sockets** | 2 server sockets: root-232, root-235. | — |
| **CUDA call log** | `grep` for `cuMemAlloc`, `cuMemcpy`, `module-load`, `0x0030`, `0x0032`: **none** in tail. Only `CUDA_CALL_INIT vm=9` (and RATE-LIMIT lines). | ✓ (docs: only init/context RPCs from guest; vm=9 = test-4) |
| **Processed** | Mediator stats: 252 processed, CUDA busy no, GPU util 0%. | ✓ (init-only traffic) |

**Conclusion (host):** Mediator is running and has received only **init/context** CUDA traffic for vm=9 (test-4). No alloc, HtoD, or module-load in the sampled log. Status **matches** documentation (load path not reaching GPU ops on host).

---

## 4. Ollama operating in GPU mode?

**Yes.** Evidence:

- Journal: `inference compute` with `library=CUDA`, `name=CUDA0`, `description="NVIDIA H100 80GB HBM3"`, `total="80.0 GiB"`, `available="78.0 GiB"`.
- `total_vram="80.0 GiB"` and vram-based default context.
- No "filtering device which didn't fully initialize" and no `library=cpu` in the sampled journal.

So **discovery** is in GPU mode. The **load** path is still not using the GPU for alloc/HtoD (runner never sends 0x0030/0x0032), which is the known blocker described in the docs.

---

## 5. Summary

| Item | Status |
|------|--------|
| **VM and host vs docs** | **Match.** GPU mode on, refresh warning present, no alloc/HtoD from load runner, host sees only init/context for vm=9. |
| **Ollama GPU mode** | **Yes** (discovery and device list). Load path still uses CPU (no cuMemAlloc/cuMemcpyHtoD). |
| **Interrupted operation** | Verified that current state is consistent with docs; no evidence that a later state (e.g. alloc/HtoD or module-load) was reached before interruption. |

---

## 6. Commands used (for re-run)

```bash
# VM
python3 connect_vm.py "systemctl is-active ollama"
python3 connect_vm.py "sudo journalctl -u ollama -n 150 --no-pager | grep -E 'inference compute|total_vram|library=CUDA|filtering device|unable to refresh'"
python3 connect_vm.py "ls -la /tmp/vgpu_*.log; wc -l /tmp/vgpu_call_sequence.log; grep -cE '0x0030|0x0032' /tmp/vgpu_call_sequence.log"
python3 connect_vm.py "test -f /tmp/vgpu_cuMemAlloc_called.log && wc -l /tmp/vgpu_cuMemAlloc_called.log || echo 'cuMemAlloc_called.log missing'"
python3 connect_vm.py "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:11434/api/tags"

# Host (read-only)
python3 connect_host.py "tail -200 /tmp/mediator.log"
python3 connect_host.py "grep -E 'vm=|cuMemAlloc|cuMemcpy|module-load|CUDA_CALL|0x0030|0x0032' /tmp/mediator.log | tail -50"
```
