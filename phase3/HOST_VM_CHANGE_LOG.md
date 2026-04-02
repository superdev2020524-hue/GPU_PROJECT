# Host and VM change log (Phase 3)

**Purpose:** Single place to record **dom0 (mediator host)** and **guest VM** changes, investigation baselines, and log resets so triage stays **reproducible** and avoids **misjudgments** (e.g. mixing historical `journalctl` lines, wrong VM, or stale `mediator.log`).

**Related:** `SYSTEMATIC_ERROR_TRACKING_PLAN.md`, `ERROR_TRACKING_STATUS.md`, `ASSISTANT_PERMISSIONS.md`, `DISCOVER_REFRESH_CUDA.md`, `REFRESH_AND_GPU_DETECTION_INVESTIGATION.md`.

---

## How to use this file

1. **Before** non-trivial host or VM changes: add a **Change log** row (what, where, why, who).
2. **After** log truncation or mediator restart: add a **Baseline** row with timestamp and backup path.
3. **During** investigations: paste **Checkpoint A–C** outputs here or in `ERROR_TRACKING_STATUS.md` and **cross-reference** this file if deploy/version changed.

---

## Systematic investigation (minimum)

Run in order; do not skip **correlation**.

| Step | Where | Action |
|------|--------|--------|
| 1 | Dom0 | `xe vm-list params=name-label,uuid,power-state` — which guests exist and run? |
| 2 | Dom0 | `xl list` — domid ↔ name (mediator socket path uses **root-&lt;domid&gt;**). |
| 2b | Dom0 | **`vgpu-admin scan-vms`** — **mediator `vm_id`** and Pool A registration (differs from domid; see recheck below). |
| 3 | Dom0 | `pgrep -a mediator_phase3`, `ls -la /tmp/mediator.log`, `find /var/xen -name vgpu-mediator.sock`. |
| 4 | Guest | `systemctl is-active ollama`, **last** `journalctl -u ollama -b \| grep 'inference compute' \| tail -3` (**use last line** for GPU vs CPU). |
| 5 | Dom0 | Grep **current** `mediator.log` for `module-load`, `401312`, `INVALID_IMAGE`, `rc=700` **after** any truncate (old greps are invalid). |

---

## Misjudgment prevention (read before conclusions)

1. **`inference compute`:** Journal is a **history**. Older **`library=cpu`** lines can coexist with newer **`library=CUDA`** on the same boot. **Decision rule:** use the **latest** line for the **current server PID**, not the first CPU line you see.
2. **Refresh vs initial discovery:** See `DISCOVER_REFRESH_CUDA.md`. A **failed refresh** (`unable to refresh free memory`) is **not** the same as “Ollama is in CPU mode” if the **latest** discovery still shows CUDA.
3. **`mediator.log`:** After **truncate/restart**, **E1/E4 greps from an old file do not apply.** Always note **file generation** (backup name or `wc -l` + mtime).
4. **VM identity — three different IDs:** (a) **Xen domid** from `xl list` → socket **`/var/xen/qemu/root-<domid>/...`**; (b) **mediator `vm_id`** from **`vgpu-admin scan-vms`** / SQLite / logs (**Test-10 = 10**, **Test-4 = 9**, **Test-3 = 8** in Pool A — **not** the same as domid); (c) guest **IP**. Do **not** assume `vm_id=9` means Test-10; old Test-4 notes used **9**.
5. **HtoD log lines:** `HTOD written ... first8=0` vs non-zero **source** requires **code-level** confirmation before treating as corruption (may be logging artifact).
6. **Dom0 disk:** If `/` is **>90% full**, builds and logs can fail independently of GPU logic — check `df -h` first.

---

## Inventory — baseline **2026-03-31** (UTC investigation window)

**Mediator host:** `xcp-ng-syovfxoz` (`10.25.33.10`), uptime ~4 days at survey.

| Guest name | UUID (short) | power-state (survey) | xl domid (survey) | Notes |
|------------|----------------|----------------------|-------------------|--------|
| Test-10 | `2e3042c0-5fa0-...` | **running** | **17** | Ollama Phase 3 focus; SSH `test-10@10.25.33.110`; vGPU socket `root-17`. |
| Test-5 | `d316002e-9dfa-...` | **running** | **15** | No SSH survey this pass. |
| Test-8 | `d2574299-9066-...` | **running** | **13** | No SSH survey this pass. |
| Test-4 | `ba77526f-8955-...` | halted | — | Historical Phase 3 notes refer to test-4. |
| Others (Test-1,2,3,6,7,9) | — | halted | — | — |

### Recheck — `vgpu-admin scan-vms` (operator terminal, dom0)

Command is **`vgpu-admin scan-vms`** (not `-scan-vms`).

**Pool A — registered / configured (3 VMs):**

| Name   | Xen UUID (prefix) | Mediator **VM ID** | Pool | Priority   |
|--------|-------------------|--------------------|------|------------|
| Test-4 | `ba77526f-895...` | **9**              | A    | high (2)   |
| Test-10| `2e3042c0-5fa...` | **10**             | A    | medium (1) |
| Test-3 | `4d2e3894-500...` | **8**              | A    | high (2)   |

**Unregistered (7 VMs)** — present in Xen inventory but **not** in vgpu-admin DB; message: run **`vgpu-admin register-vm`**:  
Test-8, Test-5, Test-2, Test-9, Test-7, Test-6, Test-1.

**Implication:** Guests can be **running** in Xen while **unregistered** (e.g. Test-5, Test-8 were running in an `xe`/`xl` survey). Do not assume WFQ / pool behavior for them until registered.

**Host resources (survey):** root FS **~92% used** on `nvme0n1p1` (**critical** — free space ~1.4G).  
**Toolkit (recheck — operator dom0 shell):** After **`mount ... VG_...-cuda_install /mnt/cuda_install`**, **`nvcc --version`** reports **CUDA 12.3** (V12.3.52). Non-interactive / minimal `PATH` SSH may still miss `nvcc` — align **`PATH`** or use the full compiler path when automating `make` on dom0.

**Mediator binary (survey):** `/root/phase3/mediator_phase3` (~122K), `cuda_executor.c` mtime **Mar 30** on dom0.

---

## Log reset — **2026-03-31** (fresh baseline)

**Action (dom0):** Prior `/tmp/mediator.log` **copied** to:

- `/tmp/mediator.log.bak.20260331_004607Z` (**11243** lines, ~1.8M)

Then `/tmp/mediator.log` **truncated**, **`mediator_phase3`** restarted from `/root/phase3`. New PID **2850347** (survey). Socket present: `/var/xen/qemu/root-17/tmp/vgpu-mediator.sock`.

**Test-10 (survey, no service reset this pass):** `ollama` **active**; latest `inference compute` lines show **`library=CUDA`**, **`compute=9.0`**; IP **10.25.33.110**; disk **~52%** on `/`.

**Reachability (from dev workstation):** `10.25.33.110` reachable; `10.25.33.15` reachable; `10.25.33.12` unreachable (Test-4 path default in old configs — expected if VM halted).

### Second pass — **2026-03-31 ~00:50 UTC** (mediator clear + Ollama journal vacuum)

**Dom0:** `/tmp/mediator.log` copied to **`/tmp/mediator.log.bak.clear_20260331_005059Z`**, file truncated, **`mediator_phase3`** restarted (`nohup` from `/root/phase3`). Survey PID **2852446**.

**Test-10:** `ollama` stopped; **`journalctl --rotate`**; **`journalctl -u ollama --vacuum-time=1s`** (freed **~160M** archived journals); `ollama` started — **active**; latest **`inference compute`** shows **`library=CUDA`**, **`OLLAMA_LLM_LIBRARY:cuda_v12`**.

---

## Change log (append new rows below)

| Date (UTC) | System | Change | Artifact / proof |
|------------|--------|--------|------------------|
| 2026-03-31 | Test-10 | Restored GPU-discovery baseline: backed up `vgpu.conf`, removed wrapper/`LD_PRELOAD` launch, set direct `ExecStart=/usr/local/bin/ollama serve` with path-only env matching repo `ollama.service.d_vgpu.conf` | `journalctl -u ollama` now shows `inference compute ... library=CUDA ... compute=9.0` at 04:26:42 local |
| 2026-03-31 | Test-10 | Removed `/opt/vgpu/lib/libcublas.so.12` and `libcublasLt.so.12` shim-name symlinks so `libggml-cuda.so` resolves cuBLAS from `cuda_v12` instead of `/opt/vgpu/lib` | `ldd /usr/local/lib/ollama/cuda_v12/libggml-cuda.so` now reports `libcublas.so.12 => /usr/local/lib/ollama/cuda_v12/libcublas.so.12` |
| 2026-03-31 | Test-10 | Narrowed bulk override from `VGPU_BULK_BAR1=1` to `VGPU_HTOD_BAR1=1` to keep BAR1 for HtoD only while allowing module bulk to leave the BAR1-forced path | `/etc/systemd/system/ollama.service.d/55-htod-bar1.conf`; `systemctl show ollama -p Environment` shows `VGPU_HTOD_BAR1=1` |
| 2026-03-31 | dom0 | `mediator.log` backed up + truncated; mediator restarted | `mediator.log.bak.20260331_004607Z`, PID 2850347 |
| 2026-03-31 | dom0 | Doc recheck: `vgpu-admin scan-vms`, `nvcc --version` after cuda_install mount | `HOST_VM_CHANGE_LOG.md` § Recheck |
| 2026-03-31 | dom0 | Second `mediator.log` backup + truncate + `mediator_phase3` restart | `mediator.log.bak.clear_20260331_005059Z`, PID 2852446 |
| 2026-03-31 | Test-10 | Ollama stop → journal rotate → `journalctl -u ollama --vacuum-time=1s` → start; ~160M freed | `systemctl active`; CUDA discovery in journal |
| 2026-03-31 | Test-10 | `ollama.service.d/vgpu.conf` aligned with repo timeouts + `GGML_CUDA_DISABLE_GRAPH_RESERVE` / `GGML_CUDA_DISABLE_BATCHED_CUBLAS` / `CUDA_TRANSPORT_TIMEOUT_SEC`; `ExecStart` kept `/usr/local/bin/ollama serve` (no `ollama.bin` on guest) | `connect_vm.py` + base64 `tee`; `systemctl restart ollama` |
| 2026-03-31 | Test-10 | `cuda_v12/libcublas.so.12` → `/opt/vgpu/lib/libvgpu-cublas.so.12` (was vendor `12.8.4.1`); `libcublasLt.so.12` confirmed → `libvgpu-cublasLt` | `connect_vm.py`; `systemctl restart ollama` |
| 2026-04-01 | dom0 | Restored XO Lite / direct host management by aligning partial XCP-ng update (`xapi-core` was already `26.1.3`, but `xcp-networkd`, `xcp-rrdd`, `xenopsd`, `xapi-xe`, `forkexecd`, `xcp-featured`, and `xcp-ng-xapi-plugins` were stale), then ran `xe-toolstack-restart` twice during validation | `ss -ltnp` shows `xapi` on `*:80` and `stunnel` on `*:443`; `xapi-wait-init-complete 15` returns `XAPI_INIT_OK`; `curl -k https://127.0.0.1/xolite.html` returns the XO Lite HTML |
| | | | |

---

## Next investigator checklist

- [ ] Re-run **Checkpoint A–C** after first CUDA traffic post-truncate (`SYSTEMATIC_ERROR_TRACKING_PLAN.md` §4).
- [ ] Record **vm_id** from fresh `mediator.log` for active guests.
- [ ] If rebuilding mediator with new **NVCC**: log exact `make` line and resulting binary **sha256** on dom0.
- [ ] For Test-5 / Test-8: obtain SSH or operator snapshot if those guests enter scope.
