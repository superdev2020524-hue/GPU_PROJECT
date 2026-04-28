# Phase 3 — resume session snapshot (2026-03-26)

**Purpose:** One place to pick up work after a break. This file is **point-in-time**; pair it with **`LONG_RUN_4H_LOG_PATHS.md`** and the **`LONGRUN_SESSION_<TS>.md`** for the run you care about.

---

## Stack (unchanged)

- **Guest:** `libvgpu-cuda` / `libvgpu-cublas` + **`cuda_transport`** → **BAR1** (shared-memory GPA often unavailable without `CAP_SYS_ADMIN` / pagemap — transport falls back to BAR1).
- **Host (dom0):** **`mediator_phase3`** + **`cuda_executor.c`** — real CUDA / cuBLAS on **H100** for mediated calls.
- **Config:** `vm_config.py` — **`VM_HOST`**, **`MEDIATOR_HOST`**, passwords (do not commit secrets).

---

## Implemented in this effort (do not redo blindly)

| Item | Notes |
|------|--------|
| **E4 / batched GEMM** | For **`batchCount == 1`**, executor uses **`cublasGemmEx`** instead of **`cublasGemmBatchedEx`** (avoids host illegal-address path seen with batched + sync). |
| **Post-GemmEx sync** | **`cuCtxSynchronize()`** after **`cublasGemmEx`** and **`cublasGemmStridedBatchedEx`** in **`cuda_executor.c`**; failure surfaces like other CUBLAS paths. |
| **VM native check** | **`guest-shim/test_gemm_batched_ex_vm.c`** — used to confirm fix on VM. |
| **Preflight before load** | **`guest-shim/test_gemm_ex_vm.c`** + **`run_preflight_gemm_ex_vm.sh`** — mediated **`cublasGemmEx`** + **`cuCtxSynchronize`** for shapes aligned with crash logs **(e.g. 2048×512×2048, 256×512×2048)**. Exit **0** = **`PREFLIGHT_OK`**. |
| **Orchestrated long run** | **`reset_and_start_longrun_4h.sh`:** backup + truncate **`/tmp/mediator.log`**, **`host_restart_mediator.sh`**, **`systemctl restart ollama`** on VM (via **`connect_vm.py`**), **preflight** (aborts if not clean), upload **`run_longrun_4h_capture.sh`**, start **4h** **`curl`** to **`/api/generate`**. |
| **10-minute monitoring** | Same script now starts **`phase3_longrun_10min_monitor.sh`** in the **background** with **`PHASE3_LONGRUN_TS`** matching the run. Output: **`LONGRUN_SESSION_<TS>.md`** on the workstation (see **`INCREMENTAL_RUN_MONITORING.md`**). |
| **Monitor false positives** | **`request_id=700`** in mediator lines is a **counter**, not CUDA **700** — alert regex adjusted so it does not fire on that. |

---

## Current failure mode (repeatable — not fixed by preflight)

- **Symptom:** After **~1h–1h25m** of **`tinyllama`** GPU load via long **`POST /api/generate`**: **`llama runner process has terminated: exit status 2`**, HTTP **500**, **`sched.go:575`** *error loading llama server*.
- **Preflight:** Mediated **GemmEx + sync** can be **fully green** while this still happens — points to **full runner / libggml-cuda / cgo** (or load-only paths), not the narrow GemmEx RPC smoke test.
- **Evidence:** **`LONGRUN_SESSION_20260326_200614.md`** (and earlier **`…175917…`**) — journal snippets with **register dumps** / crash context; host **`mediator.log`** often still shows **HtoD** progress (transport active up to failure window).
- **Artifacts:** VM **`/tmp/phase3_longrun_<TS>/`** — **`curl_generate.*`**, **`journal_ollama_follow.log`**, **`session_meta.txt`**, etc. Workstation: **`./collect_host_longrun_slice.sh`** for dom0 grep slice + **`fail401312.bin`** listing.

**Data hygiene:** If **`PHASE3_LONGRUN_TS`** was wrong when starting the monitor, **`LONGRUN_SESSION_*.md`** can mix sessions. Prefer the **`<TS>`** that matches **`/tmp/phase3_longrun_<TS>/`** on the VM.

---

## Error-tracking labels (short)

- **E1** — fatbin **401312** / **INVALID_IMAGE** (mediator dump path).
- **E3** — runner **exit 2** / load failure (guest/native llama).
- **E4** — **`CUDA_ERROR_ILLEGAL_ADDRESS` / rc=700** on mediated **GEMM** paths (mitigated for **batch==1** + sync instrumentation).

---

## Resume checklist

1. Confirm **VM** and **dom0** reachability; **`vm_config.py`** still correct.
2. Read **`LONGRUN_SESSION_<TS>.md`** for the last run (or start a new **`TS`** with **`reset_and_start_longrun_4h.sh`**).
3. For post-mortem: **`collect_host_longrun_slice.sh`**, full **`journalctl -u ollama`** slice on VM around failure time, **`/tmp/phase3_longrun_<TS>/`** files.
4. Next engineering fork: **symbolized runner crash** / core (if enabled), or **narrow GGML** to isolate **post-load** compute vs **mediation** (preflight already clears the latter for GemmEx).

---

## VM package recovery (when `dpkg` breaks after `apt`)

- **`VM_DPKG_NVIDIA_DKMS_RECOVERY_2026-03-27.md`** — NVIDIA DKMS **GCC 11** vs **`-ftrivial-auto-var-init=zero`** (kernel 6.8); **`gcc-12`** + **`dkms install`** + **`dpkg --configure -a`**.

---

## Key paths (repo)

| Path | Role |
|------|------|
| `phase3/src/cuda_executor.c` | Host GEMM / sync / CUDA RPC |
| `phase3/guest-shim/test_gemm_ex_vm.c` | Preflight binary source |
| `phase3/run_preflight_gemm_ex_vm.sh` | Workstation: scp, build, run preflight |
| `phase3/reset_and_start_longrun_4h.sh` | Full reset + preflight + 4h + 10-min MD |
| `phase3/phase3_longrun_10min_monitor.sh` | Incremental MD monitor |
| `phase3/LONG_RUN_4H_LOG_PATHS.md` | Log locations |
| `phase3/SESSION_CONTINUITY_LATEST.md` | Older continuity bullets + pointers |

---

## Update — 2026-03-27 (journal proof, session `20260326_200614`)

**Failure instant (VM `journalctl -u ollama`, local **08:34:34**):**

1. **`model load progress 1.00`** (`server.go:1474`) at **08:34:34.489**
2. **`[libvgpu-cublas] cublasGemmEx() CALLED (m=2048, n=512, k=2048, pid=38993)`**
3. **`[cuda-transport] poll call_id=0x00b5 seq=5 …`**
4. **`cublasGemmEx (m=256, n=512, k=2048)`** ×2 (same shapes as **`test_gemm_ex_vm`** preflight)
5. **`SIGSEGV: segmentation violation`** — **`signal arrived during cgo execution`**
6. Go stack: **`llamarunner.(*Server).loadModel`** … **`runner.go:849`** (load path, not an unrelated HTTP goroutine)

**Implication:** Preflight **passes** with the **same three GemmEx shapes** in a **standalone process**, but the **runner** dies on that **exact sequence** after **~1h21m** and **full weight upload**, at **progress 1.00**. Hypothesis space: **(a)** host return/sync or **guest stub** after the **third** GemmEx in **this** address space; **(b)** **memory corruption** or **invalid device pointers** accumulated during long BAR1/HtoD; **(c)** **secondary** `cuda-transport` connection in the same second (journal shows mmap fail → BAR1 **again** right before Gemm). **Not** E1 (no **`fail401312.bin`** on dom0 in recent slice).

**Artifacts:** Host slice saved as **`phase3/phase3_host_mediator_slice_resume_20260327_052600.txt`** (workstation).

---

*Written for resume handoff — 2026-03-26. Updated 2026-03-27.*
