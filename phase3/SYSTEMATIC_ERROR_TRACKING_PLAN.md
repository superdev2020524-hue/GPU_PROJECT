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

## 2b. Active error discipline (mandatory)

Maintain a strict queue:

1. Keep exactly **one active error** at a time.
2. When a **new** error is observed during tracing, add it to the **candidate list** with evidence, but do **not** replace the active error yet.
3. Continue tracing until the active error is **resolved**, **disproved**, or **proven superseded** by evidence.
4. Only then may the next candidate be promoted to the new active error.
5. If a candidate-side correction resolves the active error indirectly, record that closure explicitly and note the proof.
6. If the preserved `Plan A` canary regresses during `Plan B` work, that regression becomes the active error immediately.
7. A passing alternate model counts as transport-health or canary evidence unless the user explicitly says it is also the milestone target.

This prevents the investigation from drifting across multiple symptoms without closing the current blocker.

For each session note or assistant reply, include:

- **Lane** (`Plan A` canary or `Plan B` target)
- **Current Plan A state** (`pass`, `fail`, `unverified`)
- **Active error**
- **Candidate list**
- **Closure condition** for the active error
- **Last proven checkpoint**
- **Exact bounded repro**
- **Live artifact proof**
- **Evidence** showing whether the active error remains open

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
| **E4** | After **`cublasGemmBatchedEx`**, **`cuCtxSynchronize`** → **`rc=700`** **`CUDA_ERROR_ILLEGAL_ADDRESS`** (often after **`cuMemAlloc`/`HtoD`** in the same session); **`cuMemFree`** then fails with same sticky error | Host **`CUDA_CALL_CUBLAS_GEMM_BATCHED_EX`** + guest **`libvgpu_cublas.c`** RPC; async kernel / bad dims / pointer map | **`mediator.log`**: **`after cublasGemmBatchedEx: cuCtxSynchronize rc=700`**; see **`host_test_gemm_batched_native.c`** (native H100 can reproduce sync→700) |
| **E5** | **`mmq_x_best=0`** → **`mmq.cuh:3884: fatal error`** → **`ggml_abort`** / **`SIGABRT`** during **`ggml_backend_cuda_graph_reserve`** / **`mul_mat_q_case`** (Q4 MMQ path) | **`libggml-cuda-v12.so`** built **without** a usable **MMQ** template for **Hopper (sm_90)** or **arch mismatch** vs runtime GPU | **`journalctl`**: **`mmq_x_best`**, **`mmq.cuh`**, **`ggml_backend_cuda_graph_reserve`**; fix: **`BUILD_LIBGGML_CUDA_HOPPER.md`** / **`CMAKE_CUDA_ARCHITECTURES=90`** |
| **E6** | Bounded generate fails early with VM **`STATUS_ERROR: call_id=0x0030`** (`CUDA_CALL_MEM_ALLOC`) and **HTTP 500** / runner exit before fresh **module-load** proof | Pre-module allocation / status propagation / host-VM correlation gap | Fresh bounded repro plus VM journal; promote only when the request fails before any fresh **E1** evidence |
| **E7** | Stub selects **SHMEM** for **`CUDA_CALL_MEMCPY_HTOD_ASYNC`** (`0x003c`) even when SHMEM prefix is zero and BAR1 is non-zero, then sends an all-zero HtoD payload | Host stub bulk-source selection / SHMEM freshness / BAR1-vs-SHMEM arbitration | Host `daemon.log`: **`WARN authoritative shmem prefix is zero while BAR1 remains nonzero`** + **`HTOD payload before send ... path=shmem ... bytes=[00 ... 00]`** + **`FINAL_TX ... first8=0000000000000000`** |
| **E8** | **`CUDA_CALL_FUNC_GET_PARAM_INFO`** (`0x00bc`) returns **`status=801`** repeatedly during the deeper post-HtoD path | Host capability/support mismatch or unsupported function query on the running CUDA stack | Mediator log: **`call_id=0xbc result.status=801`**; guest verify log: matching **`STATUS_ERROR`** entries, but sequence continues into successful `cuLaunchKernel` |
| **E9** | During a clean deeper load, **`CUDA_CALL_LAUNCH_KERNEL`** (`0x0050`) reaches repeated launch success and then fails on sync with **`status=700`** / **`CUDA_ERROR_ILLEGAL_ADDRESS`**, after which the runner later aborts | Host kernel execution / sync fault during post-module load; likely a later-stage GPU fault rather than early transport or module-load failure | Host `mediator.log`: **`cuLaunchKernel sync FAILED: rc=700`** and **`call_id=0x50 result.status=700`**; VM current-call / journal show failure at **`cuLaunchKernel`** followed by runner core-dump / HTTP 500 |
| **E10** | Guest **shared-memory registration fails**, falls back to **`data_path=BAR1`**, and large **`CUDA_CALL_MEMCPY_HTOD_ASYNC`** (`0x003c`) chunks take minutes between guest-side **`pre_write_bulk`** and **`HTOD written`**, causing model-load timeout before the host sees later work | Guest transport setup / contiguous-GPA requirement / catastrophic BAR1 MMIO bulk-write throughput | VM journal: **`Exhausted shmem registration retries — using BAR1`**, **`Connected ... data_path=BAR1`**, and multi-minute gaps per 8 MiB HtoD chunk; host `mediator.log` shows the same HtoD chunks complete quickly once received, then sits idle |
| **E11** | Live VM boot comes up **without** the **`vgpu-cuda`** PCI device because the host launch path no longer injects Xenstore **`platform:device-model-args=-device vgpu-cuda,...`** into the live `qemu-dm` argv | Host boot / QEMU launch persistence / toolstack regression | Historical host proof: `/var/log/daemon.log.1` for **`qemu-dm-48`** logs **`Adding device-model-args from xenstore`** and the final `Exec:` line includes **`-device vgpu-cuda,...`**; fresh Apr 1 boots (`qemu-dm-5`, `qemu-dm-6`) omit both that injection log and the live `-device vgpu-cuda` arg even though Xenstore still contains the key. Guest: `lspci` lacks **`00:05.0 10de:2331`** until manual QMP hotplug |
| **E12** | Mediator does **not** auto-discover or attach to the live VM socket path after the recovered boot/hotplug, so the host only serves the fallback **`/tmp/vgpu-mediator.sock`** until a manual bridge is added | Host mediator VM-socket discovery / attach path | Host `mediator.log`: **`No QEMU chroot found, using fallback: /tmp/vgpu-mediator.sock`** while the live VM is running and serving CUDA traffic only after a manual bridge from `root-N/tmp/vgpu-mediator.sock` |
| **E13** | Post-repair runner startup can still show **`failure during GPU discovery before timeout`** or repeated runner launches before model load completes | Guest runner startup / discovery stability | VM journal: repeated **`starting runner --ollama-engine`** lines followed by **`failure during GPU discovery ... failed to finish discovery before timeout`** in some sessions, even though the repaired baseline can also reach valid GPU inference |
| **E14** | Repeated guest-side **`STATUS_ERROR`** on **`call_id=0x00bc`** continue even when host execution succeeds and end-to-end requests still return valid **`HTTP=200`** responses | Transport / status reporting consistency | VM journal: many **`STATUS_ERROR: call_id=0x00bc ... err=0x00000005`** entries; Host mediator: matching kernel launches and even successful `soft_max_f32` execution continue, and valid JSON `HTTP=200` responses are still produced |
| **E15** | After the `rc=400` param-layout blocker is removed, the first fresh terminating fault moves later to **`CUDA_CALL_LAUNCH_KERNEL`** (`0x0050`) sync failure **`rc=717`** on **`k_set_rows`**, ending the request with **`HTTP=500`** | Later-stage CUDA kernel execution / parameter packing or kernel-state fault during `set_rows` | Fresh Apr 1 clean retest on rebuilt host+guest: mediator logs repeated prior GGML launches succeeding, then **`cuLaunchKernel sync FAILED: rc=717 ... name=_Z10k_set_rows...`** with **`call_id=0x50 result.status=717`**; the VM request returns **`HTTP Error 500: Internal Server Error`** afterward |

**Primary Phase 1 blocker:** depends on the **earliest fresh failing or correctness-breaking step** in the current session. If the current live VM boot omits the **`vgpu-cuda`** PCI device and therefore cannot even present **`00:05.0 10de:2331`** in the guest, **E11** is primary until resolved or disproved. If the guest PCI device is present only after manual hotplug but the mediator still cannot discover the live VM socket path and requires a manual bridge to pass traffic, **E12** is primary for baseline stabilization. If a fresh bounded run fails at **`0x0030`** before module-load evidence, **E6** is primary. If `0x0030` is proven to complete and the first fresh anomaly is zero/incorrect HtoD payload selection on **`0x003c`**, **E7** is primary. If the guest first fails to establish shmem, logs **`Exhausted shmem registration retries — using BAR1`**, and then large **`0x003c`** HtoD chunks spend minutes in guest-side BAR1 writes before the host sees them, **E10** is primary until resolved or disproved. If the post-`E7` path reaches successful HtoD and repeated successful launches but then the first explicit terminating host fault is **`call_id=0x50 result.status=700`** / **`cuLaunchKernel sync FAILED`**, **E9** is primary until resolved or disproved. If the post-`E7` path can return valid **`HTTP=200`** responses but later fresh runs still fail during runner startup with repeated **GPU discovery timeout** logs before stable model load, **E13** becomes the next candidate for promotion after the earlier baseline blockers are closed. If a fresh rebuilt host+guest run removes the earlier **`rc=400`** launch blocker and the first terminating host fault moves later to **`call_id=0x50 result.status=717`** / **`cuLaunchKernel sync FAILED`** on **`k_set_rows`**, **E15** becomes primary until resolved or disproved. If **`call_id=0x00bc`** continues to log guest-side `STATUS_ERROR` while host-side work and end-to-end responses stay correct, **E14** remains a candidate inconsistency rather than the active blocker unless it becomes the earliest correctness-breaking step. If the post-`E7` path reaches successful HtoD and repeated successful launches but still ends in **`llama runner terminated` / `exit status 2`** with no earlier explicit terminating fault, **E3** is primary again. If **`401312`/`INVALID_IMAGE`** appears first, **E1** is primary (with **E2** as contributor per `FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`). If **Checkpoint C** is clean and **`rc=700`** appears after **`cublasGemmBatchedEx`** → **E4** is primary until resolved or disproved.

**Promotion rule:** the error registry is **not** a license to switch focus freely. Registry entries other than the current active blocker remain **candidates** until the active blocker is closed or explicitly superseded by proof.

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

### Step 3 — Host fatbin path audit (dom0 — assistant per **`ASSISTANT_PERMISSIONS.md`**)

- [ ] Confirm `cuda_executor` **`load_host_module()`**: raw pointer to `cuModuleLoadFatBinary` **first** for magic **`0xBA55ED50`** (`OLLAMA_VGPU_REVISIONS_STATUS.md`).
- [ ] Re-run **tinyllama** short generate; **Step 1** host grep — does **`401312`** still → **200**?

### Step 3b — **E4** (`rc=700` / **`CUDA_ERROR_ILLEGAL_ADDRESS`**) after **`cublasGemmBatchedEx`** (when Checkpoint **C** is clean)

- [ ] Host: `grep -E 'after cublasGemmBatchedEx|rc=700|GEMM_BATCHED dims'` **`/tmp/mediator.log`** — confirm **E4** signature (see **`ERROR_TRACKING_STATUS.md`** session **2026-03-26**).
- [ ] Repo: **`cuda_executor.c`** `CUDA_CALL_CUBLAS_GEMM_BATCHED_EX`; guest **`libvgpu_cublas.c`** `cublasGemmBatchedEx` RPC payload (**`m,n,k,lda`,** pointer table).
- [ ] Optional: run **`host_test_gemm_batched_native`** on dom0 to see if **native** path hits **sync → 700** (isolates cuBLAS/driver vs mediator).
- [ ] Compare logged **`GEMM_BATCHED dims`** to **`cuMemAlloc`** sizes in the same window (OOB → **700**).

### Step 4 — If E1 persists: cuBLAS Lt / libcublas alignment (dom0 + VM — assistant per **`ASSISTANT_PERMISSIONS.md`**)

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
2. **Active error:** one ID only, plus why it remains active.  
3. **Candidate list:** other observed IDs, each with **confirmed / not observed / unknown**.  
4. **Evidence:** one **host** line + one **VM** line (or “unreachable”).  
5. **Next single step:** one checkbox from §6 that advances or closes the active error.

This is the **systematic** tracking standard for Phase 3 / Phase 1 from here on.
