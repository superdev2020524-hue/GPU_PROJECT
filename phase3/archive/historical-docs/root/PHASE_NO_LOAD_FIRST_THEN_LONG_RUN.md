# Phase: no full model load first, then long-run load

*Created: 2026-03-28 — ordered playbook: fix everything checkable **without** a full tensor upload, then run long-duration / full-model loads.*

**Supersedes ad-hoc triage for this track:** Work **one numbered item at a time** until it passes or is explicitly deferred. Only after **Phase A exit criteria** (or the **deadline** below) start **Phase B**.

**Related (already in repo):** **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`** (checkpoints A–C, E1–E5), **`INCREMENTAL_RUN_MONITORING.md`**, **`ERROR_TRACKING_STATUS.md`**, **`ASSISTANT_ROLE_AND_ANTICOUPLING.md` §5.4** (long loads require explicit approval).

---

## Definitions

| Term | Meaning |
|------|--------|
| **Full model load** | A generate that uploads **full** model weights to GPU for your target model (e.g. tinyllama at full GPU layers) **and** drives init through **`llama_init_from_model`**. |
| **Phase A** | All checks and fixes that do **not** require **full model load** (preflight, static binary proof, checkpoints, small host/VM smoke). |
| **Phase B** | Long-duration / full load **after** Phase A is complete or the **deadline** triggers. |

---

## Deadline (when to move to Phase B anyway)

Set **one** of these (edit this section when you choose):

- [ ] **Date/time:** _________________________ (operator: fill in), **or**
- [ ] **Phase A complete:** all **mandatory** Phase A items below are **PASS** or **N/A with written rationale** in **`ERROR_TRACKING_STATUS.md`**.

If the deadline arrives with items still **FAIL**, proceed to Phase B **with those failures documented** so long-run logs are interpreted against known gaps.

---

## Phase A — find and fix without full model load (do in order)

Each row: **Status** = `TODO` | `PASS` | `FAIL` | `N/A`. Record fixes and log paths in **`ERROR_TRACKING_STATUS.md`** as you go.

### A1 — Checkpoints (mandatory gates)

| # | Item | Command / evidence | Status |
|---|------|---------------------|--------|
| A1.1 | **Checkpoint A** — service + GPU mode | `SYSTEMATIC_ERROR_TRACKING_PLAN.md` §4 Checkpoint A | **PASS** — `ollama` **active**; journal **`inference compute`** **`library=CUDA`** **`compute=9.0`** (VM **2026-03-27** boot lines). |
| A1.2 | **Checkpoint B** — shim + compute consistency | §4 Checkpoint B (`inference compute`, `strings` on shim) | **PASS** — `/usr/lib64/libvgpu-cuda.so` present; live path logs **CC=9.0** / **HEXACORE vH100**; matches `compute=9.0`. |
| A1.3 | **Checkpoint C** — host `module-load` / **no** spurious **401312**/**INVALID_IMAGE** in current tail | §4 Checkpoint C | **PASS (vacuous)** — `/tmp/mediator.log` on dom0 was **0 lines** at check; **`401312`**/**`INVALID_IMAGE`** counts **0**; **`mediator_phase3`** running. **Caveat:** re-run Checkpoint C **after** CUDA traffic (or log rotation) **before** Phase B if you need a non-empty **module-load** tail. |

**If A1.1–A1.3 fail:** fix per **`GPU_MODE_DO_NOT_BREAK.md`**, **`FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`**, **`E1_ERROR_TRACING_NEXT_METHODS.md`** as applicable — **before** preflight or long runs.

### A2 — Deployed GGML / CUDA build proof (no inference required)

| # | Item | Status |
|---|------|--------|
| A2.1 | Confirm **which** `libggml-cuda` path Ollama loads (`cuda_v12`, symlinks). | **PASS** — **`/usr/local/lib/ollama/cuda_v12/libggml-cuda-v12.so` → `libggml-cuda.so`** (mtime **Mar 27**); Ollama **`libdirs=cuda_v12,ollama`**. |
| A2.2 | **`cuobjdump` / `strings`** — shipped **`.so`** contains expected **SM** (e.g. **sm_90** for Hopper) per **`BUILD_LIBGGML_CUDA_HOPPER.md`** / **`BUILD_AND_DEPLOY_LIBGGML_CUDA_PHASE3.md`**. | **PASS** — **`cuobjdump`** not installed on VM; **`strings`** shows **`.target sm_90`**. Install **`nvidia-cuda-toolkit`** on VM if you want full **`cuobjdump`** proof. |
| A2.3 | GGML build flags align with policy (e.g. **`GGML_CUDA_GRAPHS`**, arch list) — match what you intend to run in Phase B. | **PASS (partial)** — **sm_90** in binary; no **`GGML_CUDA_*`** in **`vm_config`**-style grep paths on VM (compile-time / deploy). Confirm graphs on/off in **build** used for **`libggml-cuda.so`** if MMQ/graph reserve matters. |

### A3 — Guest preflight (mediated CUDA, no full model)

| # | Item | Status |
|---|------|--------|
| A3.1 | **`run_preflight_gemm_ex_vm.sh`** — **`PREFLIGHT_OK`**, **`exit_code=0`** (three **GemmEx** shapes as designed). | **PASS** — **`PREFLIGHT_OK all GemmEx + sync passed`** (three shapes; **~4 min** wall over mediated path). |
| A3.2 | If you track **E4**: **`cublasGemmBatchedEx`** / **`host_test_gemm_batched_native`** per **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`** E4 and **`ERROR_TRACKING_STATUS.md`** — know whether it is **in scope** for this milestone or **documented deferral**. | **N/A (this pass)** — no **E4** signature in empty host log; **GemmEx** preflight passes. **E4** remains **if** **`cublasGemmBatchedEx`** / **`rc=700`** reappears — **see registry** / **Phase B**. |

### A4 — Registry sweep (E1–E5) without load

| ID | Action (no full load) | Status |
|----|------------------------|--------|
| **E1** | Host grep **`401312`/`INVALID_IMAGE`** on current **`mediator.log`**; if present, follow **E1** tracing docs — **not** a long generate. | **PASS** — counts **0** on current **`/tmp/mediator.log`** (empty file). |
| **E2** | **`compute=9.0`** (or intended) in journal vs shim — **E2** closed or documented. | **PASS** — **`compute=9.0`** in journal; shim **CC=9.0**. |
| **E3** | Preflight isolates **GemmEx**; full **E3** needs **load** — mark **“verify after Phase B”** if preflight alone cannot prove. | **Deferred to Phase B** — **GemmEx** ok; **full runner** not exercised. |
| **E4** | If batch-Gemm signature appears in host log, follow **`ERROR_TRACKING_STATUS.md`** E4 notes; native host repro if used. | **N/A** — no host log lines for batch signature in this snapshot. |
| **E5** | **MMQ** cannot be fully validated **without** hitting init — after A2, **E5** is **reduced risk** if arch/MMQ templates are proven; **final proof** is Phase B. | **Risk reduced** — **sm_90** in **`libggml-cuda.so`**; **final MMQ/graph-reserve proof** → **Phase B**. |

### A5 — Observability ready for Phase B

| # | Item | Status |
|---|------|--------|
| A5.1 | Core / traceback: **`CRASH_SYMBOLICATION_AND_COREDUMPS.md`** / service limits as you use. | **PASS (repo)** — doc present; **no VM change** this run. |
| A5.2 | **`phase3_longrun_10min_monitor.sh`** (or equivalent) understood; **`PHASE3_LONGRUN_TS`** convention documented for the next run. | **PASS** — script requires **`PHASE3_LONGRUN_TS`**; **`LONGRUN_SESSION_${TS}.md`** output; see script header. |

---

## Phase A exit criteria (all required to “prefer” Phase B)

- [x] **A1.1–A1.3** **PASS** (or **FAIL** with explicit fix in progress and operator sign-off to continue). — *Done **2026-03-28** (see A1.3 caveat).*
- [x] **A2.1–A2.3** **PASS** for the **`libggml-cuda`** you will use in Phase B. — *sm_90 confirmed via **strings**.*
- [x] **A3.1** **PASS**.
- [x] **A4** — **E1**/**E2** not silently failing; **E3**/**E5** acknowledged as **load-dependent** for final proof.

---

## Phase A completion report (automated run **2026-03-28**)

**Executed from workstation** against **`vm_config`** (`VM` **10.25.33.12**, **`MEDIATOR_HOST`** **10.25.33.10**). **No code changes** were required in the repo for **PASS** items.

| Note | Detail |
|------|--------|
| **Host** `mediator.log` | **Empty** (0 lines) while **`mediator_phase3`** running — **E1** grep vacuously clean; **re-check** after workload before Phase B if you need evidence. |
| **Preflight** | **`run_preflight_gemm_ex_vm.sh`** path: **~4 min**; must use **`LD_LIBRARY_PATH`** as in script (see **`guest-shim/test_gemm_ex_vm.c`**). |
| **Phase B** | **Not started** — requires **operator approval** per **`ASSISTANT_ROLE_AND_ANTICOUPLING.md` §5.4** and **`INCREMENTAL_RUN_MONITORING.md`**. |

---

## Phase B — long-term / full model load (after Phase A or deadline)

1. **Operator approval** for wall-clock and model per **`ASSISTANT_ROLE_AND_ANTICOUPLING.md` §5.4** and **`INCREMENTAL_RUN_MONITORING.md`**.
2. Use **`run_longrun_4h_capture.sh`** / **`reset_and_start_longrun_4h.sh`** or your chosen bounded script; **monitor every ~5–10 min** (rules in **`INCREMENTAL_RUN_MONITORING.md`**).
3. Append **`LONGRUN_SESSION_<PHASE3_LONGRUN_TS>.md`** via **`phase3_longrun_10min_monitor.sh`** when possible.
4. On failure: **`collect_host_longrun_slice.sh`**, VM **`journalctl`** slice, update **`ERROR_TRACKING_STATUS.md`** with **E*** classification.

**Purpose of Phase B:** catch **scale**, **full init / graph reserve**, **MMQ at real shapes**, and **duration-dependent** transport issues that Phase A cannot prove.

---

## One-line log (optional)

| Date | Phase | Item | Result |
|------|-------|------|--------|
| 2026-03-28 | A | A1–A5 remote checks | PASS (see tables above); Phase B pending |
| | B | | |

---

*End of playbook — execute A1 → A5 in order; then B.*
