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

1. **Record:** VM id used by mediator (confirm in host / guest log — **VM‑6 Phase 3 guest** is typically **`vm_id=6`** in **[`cuda-transport`]** **`Connected`** lines; archival rows used **`vm=9`** for **test‑4**).
2. **Snapshot host:** **E1 / module-load:** `grep -F 'data_len=401312' /tmp/mediator.log | tail -10` and **`grep -F 'INVALID_IMAGE' /tmp/mediator.log | tail -5`** (counts: **`grep -c 'data_len=401312' … || true`** — see **`SYSTEMATIC_ERROR_TRACKING_PLAN.md` §6** Checkpoint **C**). **Do not** confuse **`[MEDIATOR] … request_id=401312`** (RPC sequence can collide with that number) with **fatbin **`data_len=401312`**. Optional context: **`grep 'module-load' /tmp/mediator.log | tail -15`**. If load ran: **`grep 'HtoD progress' /tmp/mediator.log | tail -5`**.
3. **Snapshot VM:** bounded **`journalctl`** (avoid **`journalctl -b`/`24h`/`grep`** over **`connect_vm`** default timeout — use **`--since`/`--until`** or **`-S '15 min ago'`** per **`PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md`** / **`TRANSPORT_SHMEM_CONTIGUITY.md`**):  
   `journalctl -u ollama -S '15 min ago' --no-pager | grep -E 'inference compute|STATUS_ERROR|MODULE_LOAD|exit status|from=BAR1' | tail -30`
4. **Transport plane (when load/generate ran):** paste at least one **`[cuda-transport] Connected … data_path=`** line from a narrow window around connect (**`TRANSPORT_SHMEM_CONTIGUITY.md`**).
5. **Runbook / artifact:** note whether the session used **`run_mar29_section8_chain.sh`**, **`run_resident_mar29_test4.sh`**, or manual **`PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md` §7/§8** steps (**`TRANSMISSION_OUTCOMES_AND_PROGRESS_ASSESSMENT.md` §5**).

Store **date, host tail, VM tail, transport line(s), script** in a scratch file or chat. Without correlation, long runs are **uninterpretable**.

---

## 4. Checkpoints (order is mandatory)

### Checkpoint A — Service + GPU mode (seconds)

| Check | Command (VM) | Pass criteria |
|--------|----------------|----------------|
| Service | `systemctl is-active ollama` | `active` |
| Discovery | `journalctl -u ollama -S '30 min ago' --no-pager \| grep 'inference compute' \| tail -3` (or **`-b`** on an interactive VM shell — wide **`journalctl`** can exceed **`connect_vm`** timeouts) | `library=CUDA`, VRAM plausible |

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

## 5. Error registry (current situation — updated through **2026-05-15**)

| ID | Error (observed) | Layer | Confirmed by |
|----|-------------------|--------|----------------|
| **E1** | `cuModuleLoadFatBinary` **`data_len=401312`** → **`CUDA_ERROR_INVALID_IMAGE` (rc=200)** | Host driver + fatbin content | `mediator.log` grep **confirms symptom only** — use **`E1_ERROR_TRACING_NEXT_METHODS.md`** (binary / provenance / path audit), not grep alone |
| **E2** | Ollama logs **`compute=8.9`** while shim strings advertise **Hopper / CC=9.0** path | Discovery / attribute source mismatch | `journalctl` vs `strings libvgpu-cuda.so` |
| **E3** | (Historical) Runner **exit status 2** after host-side success | Guest / GGML / later CUDA | Past journals; re-verify after E1 fixed |
| **E4** | After **`cublasGemmBatchedEx`**, **`cuCtxSynchronize`** → **`rc=700`** **`CUDA_ERROR_ILLEGAL_ADDRESS`** (often after **`cuMemAlloc`/`HtoD`** in the same session); **`cuMemFree`** then fails with same sticky error | Host **`CUDA_CALL_CUBLAS_GEMM_BATCHED_EX`** + guest **`libvgpu_cublas.c`** RPC; async kernel / bad dims / pointer map | **`mediator.log`**: **`after cublasGemmBatchedEx: cuCtxSynchronize rc=700`**; see **`host_test_gemm_batched_native.c`** (native H100 can reproduce sync→700) |
| **E5** | **`mmq_x_best=0`** → **`mmq.cuh:3884: fatal error`** → **`ggml_abort`** / **`SIGABRT`** during **`ggml_backend_cuda_graph_reserve`** / **`mul_mat_q_case`** (Q4 MMQ path) | **`libggml-cuda-v12.so`** built **without** a usable **MMQ** template for **Hopper (sm_90)** or **arch mismatch** vs runtime GPU | **`journalctl`**: **`mmq_x_best`**, **`mmq.cuh`**, **`ggml_backend_cuda_graph_reserve`**; fix: **`BUILD_LIBGGML_CUDA_HOPPER.md`** / **`CMAKE_CUDA_ARCHITECTURES=90`** |
| **E6** | **`SIGFPE: floating-point exception`** in **`launch_fattn<…>(…)`** / **`ggml_cuda_flash_attn_ext_mma_f16_case`** during **`evaluate_and_capture_cuda_graph`** → **`ggml_backend_cuda_graph_reserve`** → **`llama_init_from_model`** ( **`llama-context.cpp`**) | Guest **`libggml-cuda.so`** flash-attn **MMA** / tile path during **graph reserve** (can coexist with **`GGML_CUDA_DISABLE_GRAPHS=1`** — reserve path may still run); **`cublasGemmEx() RETURN ok`** lines are **not** the faulting frame. **GDB / math:** host **`idiv`** with **0** divisor when **`ggml_cuda_info().devices[id].nsm`** is **0** (**`fattn-common.cuh`** **`max_blocks=n_sm*…`**). **`patches/phase3_ggml_cuda_init_nsm_fallback.patch`**: **`ggml_cuda_init`** sets **`nsm=132`** if **`multiProcessorCount`≤0** (**`BUILD_AND_DEPLOY_LIBGGML_CUDA_PHASE3.md`** §**1** item **6**) — **VM retest** pending. **`phase3_ggml_fattn_launch_avoid_host_sigfpe.patch`**: **`launch_fattn`** **0/** guards — prior VM trial did **not** remove **E6** alone (**§1** item **5**). | **`vm_gdb_attach_sigfpe.sh`** + **`gdb`** **`thread apply all bt full`**; doc **`CRASH_SYMBOLICATION_AND_COREDUMPS.md`** §**4b**; optional **`PHASE3_GGML_CUDA_FA=OFF`** rebuild (trades into **E7** on VM-6 as of May 2026). |
| **E7** | After **`GGML_CUDA_FA=OFF`** **`libggml`** deploy: **`cublasGemmEx() RETURN ok`** logged with **`cublas_status=13`** (**`CUBLAS_STATUS_EXECUTION_FAILED`**), large **`m`** (**32000** and/or **5632**), guest **`CUDA error: the function failed to launch on the GPU`**, **`ggml-cuda.cu` CUDA error** | **GGML / FA-off** **`cublasGemmEx`** — **not** **E6** / **SIGFPE**. **May 2026:** mediated **`test_gemm_ex_vm e7` / `e7seq`** → **`cublas_status=0`**. **May 2026 (`GEMM_EX buffer OOB` on dom0):** failing **`m=5632`** shows **`A`** registered as **8388608** B (**≈2048²×fp16**) while **`OP_T`** + **`lda=2048`**, **`m=5632`** needs **~23 MiB** — **`ILLEGAL_ADDRESS`** aligns with **OOB**; sampled **`A`** addresses are **not** sub-ranges of the **~52 MiB** B/C pool — **root** is **GGML `row_diff` / buffer vs `cublasGemmEx` `m`**, not **`MEM_ALLOC`** under-reporting. **Executor:** span validation before **`cublasGemmEx`**; **`vm_find_mem`** / **`vm_find_mem_entry`** pick **largest** containing entry when rows overlap. | **`journalctl`** (**`cublas_status=`**). **Host:** **`call_id=0xb5`**, **`GEMM_EX buffer OOB`**, **`after cublasGemmEx: cuCtxSynchronize rc=700`**; **`correlate_e7_step5a.sh`** |

**Primary Phase 1 blocker:** depends on **current** `mediator.log`. If **`401312`/`INVALID_IMAGE`** appears → **E1** first (with **E2** as contributor per `FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`). If **Checkpoint C** is clean and **`rc=700`** appears after **`cublasGemmBatchedEx`** → **E4** is primary until resolved or disproved. If **Checkpoint C** is clean and the guest shows **`SIGFPE`** / **`launch_fattn`** after mediated Gemm ok → **E6** (see **`CRASH_SYMBOLICATION_AND_COREDUMPS.md`** §**4b**). If **`GGML_CUDA_FA=OFF`** is trialed and guest shows **`cublasGemmEx` … `cublas_status=13`** (wide **`m`**) → **E7** (**§6** Step **5a**).

---

## 6. Next-step plan (ordered, permissions-aligned)

### Mar 29 Stage 1 parity (**historical Test-4 / `vm=9` vs VM-6 / `vm=6`**)

**Pass criterion:** bounded **`POST /api/generate`** for **`tinyllama:latest`**, **`HTTP 200`**, JSON **`"done": true`**, with **time-aligned** dom0 **`/tmp/mediator.log`** in the same window.

**2026-05-15 (VM-6 / `vm=6`, current stack):** **Checkpoint D** **PASS** for the **archival March 29 `Test-4` JSON** (**`curl -m` ~185**, default GPU **after** **§1** preload + **`num_gpu:0`** CPU prime with the **same** `Hello` / `num_predict`) — canonical runbook **`PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md` §8** + **`ERROR_TRACKING_STATUS.md`** (cold chain + resident regression rows). **`load_duration`:** first heavy step ~**8 s** on the **CPU** prime leg; strict **`Test-4`** leg **resident** (~0.1 s class). **Single-request** default-GPU **cold ~7.46 s** matching historical **`vm=9`** remains **`TRANSPORT_SHMEM_CONTIGUITY.md`** if required.

**2026-05-16 (VM-6 / `vm=6`):** **`run_mar29_section8_chain.sh`** **(**Phase3 **§**8 **automation** — **`connect_vm` **`CONNECT_VM_COMMAND_TIMEOUT_SEC=2700`**, **`~1045`s **wall****) **:** **`PS_HIT` **w=**17**; **§**7 **`HTTPCPU:200` **`~8.96`s** **`load_duration≈8.14e9` **ns**; **strict **`Test-4` **`HTTPT4:200` **`~0.58`s** **`load_duration≈6.89e7` **ns** **+ **readable **`response`**; **`journalctl`** **shows **`compute=9.0`**, **`offloaded 23/23 layers to GPU`**, **`CUDA0 model buffer` ~571 MiB** **after** **preload** **(**Mar** **29 **default-GPU** **tensor** **evidence**)**. **CRLF** **in** **base64-decoded** **script **→ **`systemctl` **failure** **—** **fixed** **LF-only** **in** **worktree** **(**`ERROR_TRACKING_STATUS.md` **)**.**

**Historical branch (wrong `.so` / FA class):** Checkpoints **A–C** can pass while **D** fails — **FA-on** (**`libggml-cuda.so` ≈199 MiB**) may hit **E6** (**`SIGFPE`** after **`cublasGemmEx` RETURN ok**); **FA-off** (~144 MiB) trades into **E7** (§**6** Step **5a**, **`mul_mat`** / wide **`GemmEx`**). **Phase3** shim + **`cuda_executor`** mitigations **do not** replace **Ollama/GGML** changes on that branch — follow **`BUILD_AND_DEPLOY_LIBGGML_CUDA_PHASE3.md`**.

### Step 1 — Baseline capture (assistant, ~5 min)

- [x] VM: Checkpoint **A** + **B** snapshots (commands in §4). *(**2026-05-13** VM-6: **`ollama` active**, **`/api/tags` HTTP 200**, **`inference compute`** **`library=CUDA`**, **`compute=9.0`** — see **`ERROR_TRACKING_STATUS.md`**.)*
- [x] Host: Checkpoint **C** snapshot. *(**2026-05-13:** **`401312`/`INVALID_IMAGE` → 0**, **`mediator_phase3`** running.)*
- [x] Append one-line summary to `ERROR_TRACKING_STATUS.md` or a dated note: “E1 present: yes/no”, “compute log: X.Y”. *(**2026-05-13** session: **E1** not observed; **compute=9.0** in sampled journal.)*
- [x] **March 29 `Test-4` bounded replay (VM-6):** **`PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md` §8** when **C** clean — **`HTTP 200` + `done: true` + readable `response`** documented **2026-05-15** **`ERROR_TRACKING_STATUS.md`** *(cold restart + §1 + CPU prime + strict `Test-4`; resident §8 steps 4–5 regression also logged)*. **Resident **pair **automation:** **`run_resident_mar29_test4.sh`** **(**`connect_vm` **verified **2026-05-16**)**.

### Step 2 — Close E2 before another heroic generate (assistant VM + your Ollama source)

- [x] Identify **exact** field feeding `msg="inference compute" … compute=` (**`discover/types.go` `LogDetails` → `dev.Compute()`** → **`ml/device.go` `DeviceInfo.Compute()`** formats **`ComputeMajor`/`ComputeMinor`** → set in **`ml/backend/ggml/ggml.go` `BackendDevices()`** from **`C.ggml_backend_dev_get_props`** → **`props.compute_major` / `compute_minor`** — not a separate NVML-only field in **`types.go`**; see **`ERROR_TRACKING_STATUS.md`** session **2026-03-27**).
- [ ] On any **future** **`compute=8.9`** vs shim **`CC=9.0`** replay: trace **GGML/CUDA backend props** (and mediated **`cudaGetDeviceAttribute`** / discovery) — **do not** patch **`types.go`** alone; align **props** source with **kernel selection**.
- [x] **Pass criterion (VM-6, May 2026):** repeated **`journalctl`** snapshots show **`compute=9.0`** after service usage (**`ERROR_TRACKING_STATUS.md`** rows **2026-05-13**–**2026-05-15**); re-grep after **restart** whenever **E2** is suspected.

### Step 3 — Host fatbin path audit (dom0 — assistant per **`ASSISTANT_PERMISSIONS.md`**)

- [x] Confirm `cuda_executor` **`load_host_module()`**: raw pointer to `cuModuleLoadFatBinary` **first** for magic **`0xBA55ED50`**, then **0x466243b1** wrapper fallback — **`phase3/src/cuda_executor.c`** ~`693`–`726`; **`OLLAMA_VGPU_REVISIONS_STATUS.md`**. **2026-05-13** dom0 **`/root/phase3/src/cuda_executor.c`** **`sed -n '688,728p'`** matches this tree (**`CONNECT_HOST_FORCE_PEXPECT=1`**).
- [x] **Checkpoint C (same dom0 sample):** **`grep -c 'data_len=401312' /tmp/mediator.log` → `0`**, **`grep -c 'INVALID_IMAGE'` → `0`** — **no** current **E1** `module-load` failure corpus in grep (**`mediator_phase3` PID `3352101`**). **Tinyllama** fresh generate + re-grep **deferred:** **`connect_vm.py`** **`test-6@10.25.33.16`** **`Permission denied (publickey,password)`** — operator restores VM SSH / **`VM_PASSWORD`** to re-validate **Step 1** host tail after **D**.

### Step 3b — **E4** (`rc=700` / **`CUDA_ERROR_ILLEGAL_ADDRESS`**) after **`cublasGemmBatchedEx`** (when Checkpoint **C** is clean)

- [x] Host: `grep -E 'after cublasGemmBatchedEx|rc=700|GEMM_BATCHED dims'` **`/tmp/mediator.log`** — confirm **E4** signature (see **`ERROR_TRACKING_STATUS.md`** session **2026-03-26**). **(2026-05-13 dom0 sample:** **`grep -c 'after cublasGemmBatchedEx'` → `0`**, **`GEMM_BATCHED`** tail **empty** — **E4** *batched* **not** **in** **this** **file** **slice**; **`grep -c 'after cublasGemmEx'` → `3`**, **`cuCtxSynchronize rc=700`** **matches** **`3`**, **`CUDA_ERROR_ILLEGAL_ADDRESS`** **→ `6`** — **GemmEx**-path **ILLEGAL_ADDRESS** / **recovery** **lines** **present** — treat **under** **E7** **/** **executor** **`GEMM_EX`** **correlation** **`correlate_e7_step5a.sh`**, **not** **classic** **E4** **batched** **until** **`after cublasGemmBatchedEx`** **appears**.)
- [x] Repo: **`cuda_executor.c`** `CUDA_CALL_CUBLAS_GEMM_BATCHED_EX` (~`2456`–`2605`): **`CublasGemmBatchedExCallHdr`**, guest **pointer table** **`vm_find_mem`**, **`bc==1` → `cublasGemmEx`**, **`bc>1` → per-batch `cublasGemmEx` loop**, then **`cuCtxSynchronize`** → **`after cublasGemm%sEx`** / **`GEMM_BATCHED dims`** on failure (~`2575`–`2590`). Guest **`guest-shim/libvgpu_cublas.c`** **`cublasGemmBatchedEx`** (~`1084`–`1156`): builds payload, **`cublas_copy_ptr_table`**, **`cuda_transport_call(…, CUDA_CALL_CUBLAS_GEMM_BATCHED_EX, …)`**.
- [x] Optional: run **`host_test_gemm_batched_native`** on dom0 to see if **native** path hits **sync → 700** (isolates cuBLAS/driver vs mediator). **(2026-05-16 dom0:** **`/root/phase3/host_test_gemm_batched_native.c`** built with **`gcc`** + **`/usr/local/cuda`**) **`cublasSgemm`** / **`cublasGemmEx`** **+** **sync** **→** **0**; **`cublasGemmBatchedEx` DEFAULT** **status=0**, **`cudaDeviceSynchronize` → 700** **(ILLEGAL_ADDRESS)**; **PEDANTIC+ALGO0** **status=13**, **sync** **→** **700** — **matches** **`host_test_gemm_batched_native.c`** **header** **+** **`ERROR_TRACKING_STATUS.md` **2026-03-26** **row**.)**
- [x] Compare logged **`GEMM_BATCHED dims`** to **`cuMemAlloc`** sizes in the same window (OOB → **700**). **(2026-05-16 dom0:** **`grep 'GEMM_BATCHED dims' /tmp/mediator.log`** **tail** **empty** — **no** **fresh** **mediated** **batched-dims** **lines** **to** **pair** **this** **slice**; **archived** **`ERROR_TRACKING_STATUS.md` **2026-03-26** **`m=n=k=32`**, **three** **`cuMemAlloc` 4096** bytes**, **not** **OOB** **vs** **dims**; **`host_test_gemm_batched_native` **2026-05-16** **`sync→700`** **without** **remoting** — **same** **non-OOB** **geometry** **class** **as** **that** **row**.)** **Wide **`m`** OOB** **(**E7** **`GEMM_EX buffer OOB`**) **remains** **separate** — **plan** **§**6** **Step** **5a**.**

### Step 4 — If E1 persists: cuBLAS Lt / libcublas alignment (dom0 + VM — assistant per **`ASSISTANT_PERMISSIONS.md`**)

- [ ] Replace **`/usr/local/lib/ollama/cuda_v12/libcublasLt.so.12`** (and **`libcublas.so.12`** if needed) with a build appropriate for **H100 + dom0 driver** (see `FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`).
- [ ] **Pass criterion:** new `/tmp/fail401312.bin` (if dumped) **`cuobjdump`** shows **sm_90** or acceptable PTX, **or** `module-load … 401312 … rc=0`.

### Step 5a — **E7** after **`GGML_CUDA_FA=OFF`** (wide **`cublasGemmEx`**, **`cublas_status=13`**)

**When:** **Checkpoint C** clean, **E6** **`SIGFPE`** mitigated by FA-off **`libggml`** but generate still fails — see **`ERROR_TRACKING_STATUS.md`** session **2026-05-14** (**E7**).

**Guest log semantics** (**`libvgpu_cublas.c`**): **`cublasGemmEx() RETURN ok`** means the **RPC / transport** succeeded (**`tc_rc==0`**); the number printed as **`cublas_status=`** is the **authoritative** value returned from the host executor (e.g. **13** = **`CUBLAS_STATUS_EXECUTION_FAILED`**). It is **not** “cuBLAS succeeded.”

- [x] Host: **`grep -E '\[MEDIATOR\].*call_id=0xb5|call_id=0xb5'`** **`/tmp/mediator.log`** — **`CUDA_CALL_CUBLAS_GEMM_EX`**. **Interpretation:** **`result.status=0`** means executor **`CUresult`** success; **`cublasStatus_t`** for Gemm is in the **RPC result payload** (**guest `cublas_status=`**). **Do not** treat **`result.status=0`** as “cuBLAS ok.” Prefer **`bash correlate_e7_step5a.sh`** plus this grep. **(2026-05-16:** **`/root/phase3/correlate_e7_step5a.sh`** **installed** **+** **run** **—** **tail** **`0xb5`** **lines** **all** **`result.status=0`** **while** **`[cuda-executor]`** **`GEMM_EX buffer OOB`** **+** **`after cublasGemmEx … rc=700`** **present** **in** **file** **; **`call_id=0x26`** **non-zero** **samples** **`result.status=709`**, **`700`** **in** **correlator “grep non-zero sync”** **section**.)*
- [ ] Guest: capture full **`journalctl`** slice (shim **`RETURN`** lines + GGML **`ggml-cuda.cu`** error) for **one** repro; note **tensor/layout** if logged. **(2026-05-16:** **§**8 **Stage** **1** **PASS** **window** **(**`--since`/`--until` **pair** **to** **dom0** **correlator** **)** **—** **shim** **`cublasGemmEx`** **tail** **in** **`ERROR_TRACKING_STATUS.md` **rolling** **row** **; **mostly **`cublas_status=0`**, **some **`=7`** **(**`INVALID_VALUE` **class** **) **—** **not** **E7** **`13`** **; **still** **need** **FA-off** **/** **E7** **failure** **slice** **if** **chasing** **executor** **geometry** **.)**
- [ ] Compare **E7** (non-batched **`GemmEx`**, status **13** at guest) vs **E4** (batched path + host **`cuCtxSynchronize rc=700`**); do **not** treat **FA=OFF** as production fix until **E7** root cause is known — **rollback** to **FA-on** **`.so`** when not actively testing (see **`BUILD_AND_DEPLOY_LIBGGML_CUDA_PHASE3.md`** §**1**).
- [ ] **E7 allocator vs Gemm `m` (May 2026):** On dom0, **`GEMM_EX buffer OOB`** shows **`A`** **`cuMemAlloc`** **8 MiB** with **`m=5632`**, **`lda=2048`** (**need** **~23 MiB**). **Single next step in tree:** Ollama **`ggml-cuda.cu`** / **`ggml_cuda_op_mul_mat_cublas`** (**FA-off** path): **`row_diff`**, **`src0`** buffer geometry, and **`cublasGemmEx`** **`m`** — align with **`BUILD_AND_DEPLOY_LIBGGML_CUDA_PHASE3.md`** (rebuild **`.so`** after source fix) or pursue **FA-on** **E6** for Mar 29–style stack without **E7**.
- [x] **E7 GGML vs harness (May 2026):** **Step 5b** shows isolated **`test_gemm_ex_vm e7`** → **`cublas_status=0`**. **Archived** **`journalctl` (May 14 ~17:40:16):** many **`cublasGemmEx`** **`cublas_status=0`** (**`m` 256/2048/5632**) in the **same second** as **`m=32000` → `13`** — hypothesized **prior-Gemm** context. **`test_gemm_ex_vm e7seq`** (**May 2026**): **7** journal-like burst shapes **×4**, then wide **`m=32000`** on **VM-6** mediated path → **`E7SEQ_OK`**, **all** **`cublas_status=0`** (**~89 min** BAR1 wall; use **`CONNECT_VM_COMMAND_TIMEOUT_SEC` ≥ 7200** for **`connect_vm.py`**). **Does not** reproduce **13** in the **minimal** harness — **`E7`** remains **GGML / richer context** (**streams**, workspace, multiple handles, graphs) **or** **FA-off** + synced logs (**operator-approved**) for live **13**.
- [x] **E7 stream + workspace harness (2026):** **`test_gemm_ex_vm e7ws`** — non-default **`CUstream`**, **`cublasSetStream`**, **32 MiB** **`cublasSetWorkspace`**, wide **`m=32000`** mediated → **`E7WS_WIDE_OK`** (**VM-6**). **`e7seqws`** — burst **×4** + wide with same stream/workspace → **`E7SEQWS_OK`** (**VM-6**, ~**89 min** BAR1, **`CONNECT_VM_COMMAND_TIMEOUT_SEC` ≥ **7200**).
- [x] **E7 two-handle harness (2026):** **`e7dual`** — two **`cublasHandle_t`** on one context, small FP16 warmup Gemm on **h1** then **h0**, wide on **h0** → **`E7DUAL_OK`** (**VM-6**). **`e7seq2h`** — same burst schedule as **`e7seq`** but Gemms alternate **h0/h1** (expect ~**e7seq** BAR1 wall); run when closing multi-handle hypothesis.
- [ ] **Timing:** **`mediator.log`** must cover the **same** wall-clock interval as guest **`cublas_status=13`** (copy **`/tmp/mediator.log`** at repro time, or **`journalctl --since`** on both sides). A long-lived log may contain **zero** **`cublasGemmEx`** lines if the **E7** event was never written (rotation) or if only **success** Gemm calls ran (**executor** omits **`rc=`** line on **`CUBLAS_STATUS_SUCCESS`** unless verbose). Dom0 may still show **`[MEDIATOR] … call_id=0xb5 result.status=0`** even for **E7** — see first bullet.

### Step 5b — **E7** isolate: wide FP16 tensor-op Gemm (after Step **5a**)

- [ ] Extend **`guest-shim/test_gemm_ex_vm.c`** (or a **one-off** binary) to run **`cublasGemmEx`** with **`m=32000`**, **`n=512`**, **`k=2048`**, **`CUDA_R_16F`**, **`CUBLAS_COMPUTE_16F`**, **`CUBLAS_GEMM_DEFAULT_TENSOR_OP`** (match **`ERROR_TRACKING_STATUS.md`** **E7** journal slice). **(Done:** **`./test_gemm_ex_vm e7`**.**)**
- [ ] Run the same dims **natively on dom0** (no mediator). **Pass criterion interpretation:** if native **fails** → driver / libcublas / hardware policy; if native **passes** and mediated **fails** → remoting / executor / pointer mapping hypothesis. **(Done May 2026:** dom0 **`E7_WIDE_OK`** — native **pass**.**)**
- [ ] **May 2026 (mediated wide `e7` isolated):** **`test_gemm_ex_vm e7`** on **VM-6** → **`cublasGemmEx … cublas_status=0`**, **`E7_WIDE_OK`**, **`cuCtxSynchronize -> 0`**; dom0 **`call_id=0xb5`**, **`result.status=0`**. **Does not** reproduce **E7** (**`cublas_status=13`**) from **FA-off Ollama** — treat **E7** in GGML as **context / ordering**, not **bare** mediated Gemm. **Long BAR1** run: use **`nohup`** without **`exec`** if capturing exit **code**; else **`grep -a E7_WIDE`** on log. **Also:** **`test_gemm_batched_ex_vm`** **`cublasCreate` + Gemm **PASS** post-executor **`MEM_ALLOC`** patch; sync guest **`test_gemm_ex_vm.c`** from repo for **`e7`** if **`~/phase3`** lags.
- [ ] **Host grep (CREATE):** **`[MEDIATOR] … call_id=0xac`** shows **`result.status=0`** for **`CUDA_CALL_CUBLAS_CREATE`** — interpret like **`0xb5`**: **cuBLAS** status is in the **payload**, **not** in **`result.status`**.
- [ ] **Hypothesis (May 2026, ruled out for harness):** “**`cuMemAlloc` before `cublasCreate`** like Ollama” on **VM-6** → **`STATUS_ERROR`** on **`call_id=0x0030`** (**`CUDA_CALL_MEM_ALLOC`**), guest **`cuMemAlloc_v2 rc=2`**, BAR1 reconnect loop — **worse** than going straight to **`cublasCreate`**; **do not** reorder tests this way until **`0x0030`** is green on a **fresh** short-lived session.
- [ ] **Wire / guest decode:** **`err=0x05`** on **`STATUS_ERROR`** = **`VGPU_ERR_CUDA_ERROR`** (CUDA failure on host). Guest transport **`return 2`** → **`rpc_simple`** → values that can be **misread** as **`CUDA_ERROR_OUT_OF_MEMORY`** in logs; confirm host **`[cuda-executor] cuMemAlloc FAILED`** after repro.
- [ ] **Executor (repo May 2026):** **`CUDA_CALL_MEM_ALLOC`** uses **`ensure_vm_context`** + **`cudaSetDevice(0)`** like **`CUDA_CALL_CUBLAS_CREATE`** — **(deployed dom0 May 2026;** rebuild **`mediator_phase3`** if **`cuda_executor.c`** changes**).**

### Step 5 — Only then: full Phase 1 proof

- [x] `curl` generate to completion; host shows module load success + subsequent ops; VM returns **200** + text; `/api/ps` sane. **(2026-05-16 VM-6 + dom0:** **`connect_vm.py`** **`PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md` §**7** **then** **archival **`Test-4` **JSON** **(**`tinyllama:latest`**, **`Hello`**, **`num_predict`** **16**, **`temperature`** **0.3**, **`top_p`** **0.85** — **first **`num_gpu:0` **`curl -m 120`**, **then **omit **`num_gpu` **`curl -m 185`**)** **without** **service** **restart** **this** **sample** **; **both **`HTTP 200`**, **`done:true`**, **English **`response` **prefix** **; **`GET /api/ps`** **lists **`tinyllama:latest`** **with **sane **`expires_at`**. **Dom0:** **`grep -c 'data_len=401312'`** **and **`INVALID_IMAGE`** **`→ 0`** **(**`grep -c` **guarded** **with **`|| true`** **—** **GNU **`grep -c`** **exit** **1** **on** **zero** **matches****); **`[cuda-executor] HtoD progress`** **tail** **`vm=6`** **class** **`~22277` MB** **; **no** **fresh **`module-load` **string** **in** **last** **mediator** **tail** **(**rotation** **/** **older** **loads****) **— **Checkpoint **C** **pass** **remains** **E1** **gate**. **Guest **`journalctl -u ollama -S '3 hours ago' | grep 'inference compute'`** **:** **`library=CUDA` `compute=9.0`** **@** **May** **15** **22:26** **local** **(**Checkpoint **B** **family**). **Issue:** **`journalctl -u ollama -b | grep …`** **exceeded **`CONNECT_VM_COMMAND_TIMEOUT_SEC=90`** **—** **use **`--since` / **`-S 'N min ago'`** **per **`PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md` **/ **`ERROR_TRACKING_STATUS.md`**. **Caveat:** **runner** **log** **slice** **during** **this** **pair** **shows **`load_tensors` **/** **`llama_context`** **CPU** **buffer** **path** **(**first** **request **`num_gpu:0`** **)** — **default-GPU** **tensor** **load** **for** **strict** **leg** **not** **re-proven** **here**; **re-run **`§**8` **cold** **chain** **if** **Mar** **29** **proof** **must** **include** **fresh** **GPU** **runner** **evidence**.)**

---

## 7. What we stop doing

- **No** 40+ minute run **without** Checkpoint **C** confirming we’re past module-load **or** explicitly testing post-module.
- **No** treating “slow” as “broken layer unknown” when **`grep module-load`** already shows **E1**.
- **No** mixing **Mar 15** and **Mar 22** narratives without checking **current** `mediator.log` tail.

---

## 8. Related docs

- `ERROR_TRACKING_STATUS.md` — rolling log + Mar 22 synthesis  
- `correlate_e7_step5a.sh` — dom0 helper for §**6** Step **5a** (**E7** host log slices)  
- `FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md` — 401312 = sm_80 Lt-style  
- `ASSISTANT_PERMISSIONS.md` — VM full, host read-only (assistant)  
- `GPU_MODE_DO_NOT_BREAK.md` — do not break discovery with bad symlinks  

---

## 9. Future assistant replies (template)

When reporting progress, always include:

1. **Checkpoints passed:** A / B / C / D (which).  
2. **Error registry:** E1/E2/E3/**E4**/**E5**/**E6**/**E7** — **confirmed / not observed / unknown**.  
3. **Evidence:** one **host** line + one **VM** line (or “unreachable”).  
4. **Next single step:** one checkbox from §6 (use **Step 5a** when **E7** is active after **FA=OFF** trials; after **5a** correlation is done, use **Step 5b**).

This is the **systematic** tracking standard for Phase 3 / Phase 1 from here on.
