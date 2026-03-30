# Systematic error tracking plan (Phase 3 / Phase 1)

*Created: Mar 22, 2026 — addresses ad-hoc “run 40 minutes and hope” triage.*

**Binding on assistants:** This plan is referenced from **`ASSISTANT_PERMISSIONS.md`** and **`ASSISTANT_ROLE_AND_ANTICOUPLING.md` §5**. PHASE3 triage must follow it unless the user explicitly opts out for a single message.

---

## 1. Why a plan matters

**Symptom:** Long model transfers and generates consume time even when the **failing layer** is already identifiable in **short, correlated** host + VM logs.

**Rule:** **Never start a long run** until **Checkpoint B or C** (below) proves the pipeline is in the state you intend to test. **Gate** expensive work on cheap checks.

---

## 2. Definitions

| Term | Meaning |
|------|--------|
| **Error** | A **specific** failing step with **log line(s)** + **return code** (e.g. host `module-load … data_len=401312 … rc=200 INVALID_IMAGE`). |
| **Hypothesis** | A proposed cause; **not** “the error” until a **falsifiable check** confirms it. |
| **Gate** | A **quick** command pair (VM + host) that must pass before the next phase. |

---

## 3. Single source of truth (per session)

For every triage **session** (same day / same experiment):

1. **Record:** VM id used by mediator (**`vm=9`** for test-4 — confirm in host log if it changes).
2. **Snapshot host:** `grep -E 'module-load|401312|INVALID_IMAGE|rc=200' /tmp/mediator.log | tail -30`
3. **Snapshot VM:** `journalctl -u ollama -n 80 --no-pager | grep -E 'inference compute|STATUS_ERROR|MODULE_LOAD|exit status'`

Store these three lines in a scratch file or chat: **date, host tail, VM tail**. Without correlation, long runs are **uninterpretable**.

---

## 4. Checkpoints (order is mandatory)

### Checkpoint A — Service + GPU mode (seconds)

| Check | Command (VM) | Pass criteria |
|--------|----------------|----------------|
| Service | `systemctl is-active ollama` | `active` |
| Discovery | `journalctl -u ollama -b --no-pager \| grep 'inference compute' \| tail -3` | `library=CUDA`, VRAM plausible |

**If fail:** Fix `vgpu.conf`, wrapper, `LD_PRELOAD`, `/opt/vgpu/lib` layout per `GPU_MODE_DO_NOT_BREAK.md` — **do not** run generate.

### Checkpoint B — Shim + CC consistency (minutes)

| Check | Command / evidence | Pass criteria |
|--------|-------------------|----------------|
| Shim installed | `ls -la /usr/lib64/libvgpu-cuda.so` | Present, recent mtime if you redeployed |
| Ollama “compute” vs shim | Compare `journalctl … inference compute … compute=X.Y` with `strings /usr/lib64/libvgpu-cuda.so \| grep CC=` | **Target:** Ollama logs **`compute=9.0`** for H100 path after fixes; **`8.9` + 401312 INVALID_IMAGE** is a **tracked discrepancy** |

**If Ollama still shows `8.9`:** **Stop.** Trace Go discovery path (which API supplies `types.go` “inference compute”) — **short** repro (restart + grep), not a 40 min load.

### Checkpoint C — Mediator module-load signature (seconds, host)

| Check | Command (host, read-only) | Pass criteria |
|--------|---------------------------|----------------|
| Pattern | `grep 'module-load' /tmp/mediator.log \| tail -20` | Understand: small loads `rc=0` vs **`401312` + `rc=200`** |

**If `401312` → `INVALID_IMAGE` still present:** The Phase 1 blocker is **host fatbin acceptance / Lt kernel package**, not weight transmission. **Do not** attribute failure to “HtoD” without seeing `0x0032` errors.

### Checkpoint D — Long generate (only after A–C)

| When | Use |
|------|-----|
| After C shows **no** `401312` failure **or** you are **explicitly** testing post-module behavior | `curl` generate with timeout ≥ `OLLAMA_LOAD_TIMEOUT` + client margin |

During the run, **every ~5 min:** host `grep 'module-load\|HtoD progress\|FAILED' /tmp/mediator.log | tail -15` — stop early if **Checkpoint C** failure reappears.

---

## 5. Error registry (current situation — Mar 22)

| ID | Error (observed) | Layer | Confirmed by |
|----|-------------------|--------|----------------|
| **E1** | `cuModuleLoadFatBinary` **`data_len=401312`** → **`CUDA_ERROR_INVALID_IMAGE` (rc=200)** | Host driver + fatbin content | `mediator.log` grep **confirms symptom only** — use **`E1_ERROR_TRACING_NEXT_METHODS.md`** (binary / provenance / path audit), not grep alone |
| **E2** | Ollama logs **`compute=8.9`** while shim strings advertise **Hopper / CC=9.0** path | Discovery / attribute source mismatch | `journalctl` vs `strings libvgpu-cuda.so` |
| **E3** | (Historical) Runner **exit status 2** after host-side success | Guest / GGML / later CUDA | Past journals; re-verify after E1 fixed |

**Primary Phase 1 blocker:** **E1** (with **E2** as likely **contributor** per `FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`).

---

## 6. Next-step plan (ordered, permissions-aligned)

### Step 1 — Baseline capture (assistant, ~5 min)

- [ ] VM: Checkpoint **A** + **B** snapshots (commands in §4).
- [ ] Host: Checkpoint **C** snapshot.
- [ ] Append one-line summary to `ERROR_TRACKING_STATUS.md` or a dated note: “E1 present: yes/no”, “compute log: X.Y”.

### Step 2 — Close E2 before another heroic generate (assistant VM + your Ollama source)

- [ ] Identify **exact** field feeding `msg="inference compute" … compute=` (Ollama `discover` / `runner` / NVML vs CUDA).
- [ ] Ensure **all** capability reads used for **kernel / library selection** see **9.0** (shim NVML + CUDA, or patch Go if it caches wrong props).
- [ ] **Pass criterion:** `journalctl` shows **`compute=9.0`** after `systemctl restart ollama`.

### Step 3 — Host fatbin path audit (**you**, dom0)

- [ ] Confirm `cuda_executor` **`load_host_module()`**: raw pointer to `cuModuleLoadFatBinary` **first** for magic **`0xBA55ED50`** (`OLLAMA_VGPU_REVISIONS_STATUS.md`).
- [ ] Re-run **tinyllama** short generate; **Step 1** host grep — does **`401312`** still → **200**?

### Step 4 — If E1 persists: cuBLAS Lt / libcublas alignment (**you**, dom0 + VM file deploy)

- [ ] Replace **`/usr/local/lib/ollama/cuda_v12/libcublasLt.so.12`** (and **`libcublas.so.12`** if needed) with a build appropriate for **H100 + dom0 driver** (see `FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`).
- [ ] **Pass criterion:** new `/tmp/fail401312.bin` (if dumped) **`cuobjdump`** shows **sm_90** or acceptable PTX, **or** `module-load … 401312 … rc=0`.

### Step 5 — Only then: full Phase 1 proof

- [ ] `curl` generate to completion; host shows module load success + subsequent ops; VM returns **200** + text; `/api/ps` sane.

---

## 7. What we stop doing

- **No** 40+ minute run **without** Checkpoint **C** confirming we’re past module-load **or** explicitly testing post-module.
- **No** treating “slow” as “broken layer unknown” when **`grep module-load`** already shows **E1**.
- **No** mixing **Mar 15** and **Mar 22** narratives without checking **current** `mediator.log` tail.

---

## 8. Related docs

- `ERROR_TRACKING_STATUS.md` — rolling log + Mar 22 synthesis  
- `FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md` — 401312 = sm_80 Lt-style  
- `ASSISTANT_PERMISSIONS.md` — VM full, host read-only (assistant)  
- `GPU_MODE_DO_NOT_BREAK.md` — do not break discovery with bad symlinks  

---

## 9. Future assistant replies (template)

When reporting progress, always include:

1. **Checkpoints passed:** A / B / C / D (which).  
2. **Error registry:** E1/E2/E3 — **confirmed / not observed / unknown**.  
3. **Evidence:** one **host** line + one **VM** line (or “unreachable”).  
4. **Next single step:** one checkbox from §6.

This is the **systematic** tracking standard for Phase 3 / Phase 1 from here on.
