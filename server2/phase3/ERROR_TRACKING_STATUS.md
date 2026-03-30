# Error tracking status (from where we left off)

*Updated: Mar 22, 2026 — post–**cuBLAS** reinstall (**scp** dom0 → VM + **`install_cublas_align_from_dom0_on_vm.py`**): checkpoints **A–C** OK; **E1** **`grep -c`** still **2** until a new **401312** load is triggered.*

**Systematic tracking (checkpoints, gates, next steps):** **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`**

**PHASE3 “resolved before” vs today’s E1:** **`PHASE3_LEGACY_RESOLVED_VS_CURRENT_E1.md`** — older “full solution” / Mar 15 **INVALID_IMAGE resolved** ≠ **401312 sm_80 Lt** (see **`FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`**).

**Simple rule (2‑minute):** **`E1_TRACE_TWO_MINUTE_RULE.md`** — do not rely on long waits; E1 proof is already **static** (**`cuobjdump` → sm_80**).

**Next process after tracing:** **`E1_NEXT_PROCESS_CHECKLIST.md`** — VM verification done; **`E1_VM_LDD_VERIFICATION_MAR22.md`** — **`ldd`** under runner path confirms **cuda_v12** **Lt**; remaining = optional **host** repro + **matrix / libcublas** alignment if needed.

### Proceed session — checkpoints **A–C** (latest)

| Gate | Result |
|------|--------|
| **A** | **`ollama` active**; **`/api/tags`** **OK** (**8 s**); last **`inference compute`** → **`compute=9.0`**, **`library=CUDA`** (**Mar 22 13:55:27**, PID **447526**). |
| **B** | **`/usr/lib64/libvgpu-cuda.so`** present. |
| **C (host)** | **`mediator.log`** **2578** lines; **`data_len=401312`** count **2** (unchanged); tail **`module-load done`** mostly **`rc=0`**; routine **`MEDIATOR`** **`result.status=0`**. |

**Registry:** **E1** still **present** in log **history**; **E2** latest line **9.0**; **E3** not re-checked this message.

---

## E1 — tracing status (corrected)

**Previous mistake:** Treating **host `grep`** of **`mediator.log`** as “enough” to **move on**. That only repeats the **symptom**; it does **not** substitute for **binary forensics**, **guest provenance**, or **mediator path** checks.

| Layer | Status |
|-------|--------|
| **Symptom capture** (`401312`, **`INVALID_IMAGE`**, **`vm_id`**, tail patterns) | **Done** — useful but **not** root-cause tracing. |
| **Effective next methods** | **`E1_ERROR_TRACING_NEXT_METHODS.md`** — **`cuobjdump`** on **`fail401312.bin`**, contrast with **`rc=0`** loads, **libcublasLt** provenance on VM, **`cuda_executor.c`** chunk/context audit, optional minimal host repro. |
| **Prior deep signal** | **`FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`** — dump showed **sm_80** / **Lt**-style kernels; points to **CC routing + cuBLAS Lt**, not “grep harder”. |

**E1 is not “terminated” in favor of E2.** **E2** may **contribute** to wrong **Lt** selection; re-validate **E1** after CC/cuBLAS changes using the **methods** doc, not only log tail.

**Latest E1 tracing (Mar 22):** **Method 1** — **`cuobjdump`** on **`fail401312.bin`** → **`sm_80`** / Ampere GEMMs. **Method 2** — size/outcome table (**4432 / 9360 / 28120** → **`rc=0`**; **401312** → **`INVALID_IMAGE`**). **Method 4** — chunk sequence **6×65536 + 8096 = 401312**, magic intact → **reassembly OK**, failure = **blob arch**. **Method 3** — VM **Lt/cublas** → **12.3.2.9**. Full tables: **`E1_ERROR_TRACING_NEXT_METHODS.md`** § Execution log.

---

## E2 — contributor (not a replacement for E1)

| Resource | Role |
|----------|------|
| **`TRACE_E2_COMPUTE_89_ROOT_CAUSE.md`** | Where **`compute=`** comes from; **`libggml-cuda.so`** rebuild. |
| **Latest VM journal** | **Last** `inference compute` line **`compute=9.0`** (Mar 22); older **8.9** lines are **historical**. |

---

## Step 1 baseline (latest run) — assistant

Per **`SYSTEMATIC_ERROR_TRACKING_PLAN.md` §3–§4** and **§6 Step 1** (no Checkpoint **D**).

| Record | Value |
|--------|--------|
| **Mediator `vm_id`** | **9** (recent **`CUDA result sent vm_id=9`**) |
| **E1 in host tail** | **Yes** — **`data_len=401312`** → **`rc=200`** **`CUDA_ERROR_INVALID_IMAGE`** (historical pairs in **`module-load` tail**); interleaved with **`rc=0`** smaller fatbins (**9360 / 28120 / 4432**). |
| **E1 one-line** | **E1 present: yes** (log history). **`grep -c 'data_len=401312' /tmp/mediator.log`** → **2**; **`wc -l`** → **1939** lines (this run). |
| **Compute log (use last line)** | **Latest `inference compute`:** **`compute=9.0`**, **`library=CUDA`** — **Mar 22 13:55:27**, PID **447526**. Older boot lines include **one** **`compute=8.9`** (Mar 21 18:58:43) — **E2** archival mismatch vs current. |
| **Checkpoint A** | **`systemctl is-active ollama`** → **`active`**. |
| **Checkpoint B** | **`/usr/lib64/libvgpu-cuda.so`** present (**Mar 20** mtime); **`strings`** shows **CC=%d.%d** / live GPU info format — compare to **`compute=9.0`** on latest discovery. |
| **Checkpoint C (host)** | **`grep 'module-load' … tail -20`**: pattern of **401312 → INVALID_IMAGE** plus **success** loads — see host capture in session. |
| **E3 (VM)** | **No new** **`exit status 2`** this session; latest journal line still **Mar 22 01:51:23** (PID **333441**). Older **Mar 20** lines also in grep. |
| **`coredumpctl list`** | **Empty** (no rows returned). |

**Plan §3 correlation (scratch):** host tail shows **vm_id=9** module-load **401312/200** + **rc=0** mix; VM tail shows **inference compute** ending at **9.0** for current PID.

---

**E2 trace (why journal shows `compute=8.9`):** **`TRACE_E2_COMPUTE_89_ROOT_CAUSE.md`** — GGML reads `prop.major`/`minor` from `cudaGetDeviceProperties`; **libggml-cuda.so** must be rebuilt after patching `ggml-cuda.cu` (Go binary alone is insufficient).

---

## Mar 22 PM — bounded run (assistant)

**Scope:** Re-run **A–C**, **`coredumpctl list`**, then **short** `/api/generate` with incremental journal samples (**`INCREMENTAL_RUN_MONITORING.md`**). Host — read-only.

| Step | Result |
|------|--------|
| **Models (`/api/tags`)** | **`tinyllama:latest`**, **`llama3.2:1b`** — **`llama3.2:3b` not installed** (first curl returned **404**; no GPU work). |
| **A** | **`ollama` active**; last **`inference compute`** **Mar 22 13:55:27**, **`compute=9.0`**, PID **447526**. |
| **B** | **`pgrep -a ollama`** → **`447526 /usr/local/bin/ollama.bin.new serve`**. |
| **C (host)** | **`mediator.log`** **1735 → 1769** lines after load attempt; tail still shows **successful** small fatbins + historical **INVALID_IMAGE** pairs. |
| **`coredumpctl list`** | **Empty** (no cores listed). |
| **Generate** | **`curl -m 120`**, **`tinyllama:latest`**, **`num_predict=16`**: **`curl: (28)`** — **0 bytes** at 120s; journal **`model load progress 0.00`**; **`load_tensors`** / **BAR1** / **HtoD** started; then **`error loading llama server`** **`timed out waiting for llama runner to start: context canceled`** (client dropped). |
| **E1 (host)** | **`grep -c 'data_len=401312' /tmp/mediator.log`** → **2** (only two **`module-load start`** lines in full log) — **no new** **401312** lines this session vs prior history. |

**Takeaway:** Bounded client **did not** outlast slow **HtoD**; **E1** not shown to **re-fire** on this attempt (count stable). **Do not** rely on “longer **`curl -m`**” as the strategy — use **decoupled preload / `keep_alive` / smaller GPU scope / BAR1 fixes** per **`PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md`**.

---

## Error tracking — Mar 22 (priority: registry + gates; **no long runs**)

**Assistant role (short):** VM — deploy, commands, logs (**full**). Host — **read-only** logs (`connect_host.py`); **no** host edits/build/restart (**`ASSISTANT_PERMISSIONS.md`**). Plan — **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`**; **no** blind long generates unless you **explicitly** allow (**`INCREMENTAL_RUN_MONITORING.md`**).

### Checkpoints A–C (just executed)

| Gate | Result |
|------|--------|
| **A** | **`ollama` active.** Latest **`inference compute`**: **`compute=9.0`**, **`library=CUDA`** — **Mar 22 13:55:27**, new server PID **447526** (post–SIGKILL restart). |
| **B** | **`libcublas` / `libcublasLt`** in **`/usr/local/lib/ollama/cuda_v12/`** now symlink to **dom0-matched `12.3.2.9`** (align deploy). |
| **C (host)** | **`mediator.log` tail** still contains historical **`401312`** → **`INVALID_IMAGE` (rc=200)** interleaved with **`rc=0`** loads — **not cleared** by log grep alone; **new** run needed to see if align **reduces** new **401312** lines. |

### Registry (plain language)

| ID | Meaning | Status this pass |
|----|---------|-------------------|
| **Host module load “invalid image” (~401312 bytes)** | Driver rejects that fatbin on H100 | Still **visible** in log **history**; **unknown** if still occurs on **next** load until a **short**, monitored test is **explicitly** approved. |
| **VM “inference compute” vs H100** | Should show CUDA + plausible CC | **OK** — **`compute=9.0`** on latest line. |
| **Runner `exit status 2`** | Native crash after load/graph | Last journal line still **Mar 22 01:51:23** (historical); **no new** crash in this **no-generate** pass. |

### Checkpoint D

**Not run** — long-duration **`/api/generate`** / heavy HtoD **not permitted** per current directive.

### Next (VM-only, short, when approved)

- **`coredumpctl list`** after any **new** **`exit status 2`**; optional **`curl -m ≤120`** with **incremental** journal samples (see **`INCREMENTAL_RUN_MONITORING.md`**).

---

## Mar 22 (assistant): Live generate + checkpoint run

**Executed:** `connect_vm.py` — `curl` POST `/api/generate` (`tinyllama:latest`, `num_predict=8`, **`curl -m 1200`**); `connect_host.py` — mediator `/tmp/mediator.log` before/after.

| Checkpoint | Result |
|------------|--------|
| **B — VM `journalctl` `inference compute`** | Latest line still **`compute=9.0`** (Mar 21 22:41:53, same `ollama_vgpu_wrapper` PID). Mar 22 run did **not** print a new discovery line (no service restart). |
| **C — host `401312` / INVALID_IMAGE** | `grep 401312 /tmp/mediator.log` → **4** lines (**2×** `module-load start` + **2×** `dumped ... fail401312.bin`). **No new** `401312` lines in this session; log grew (**734 → 930** lines) with **HtoD (`0x0032`)** activity, not new fatbin failures. |
| **E2E inference** | **`curl` timed out at 20 min** (0 bytes). VM: weights upload **~22%** (`model load progress 0.22`), then **`client connection closed before server finished loading`** → **`499`**, `error loading llama server` / **`context canceled`**. |

**Interpretation:** The **client** dropped before the slow vGPU **HtoD** finished, so the runner never reached a **post-load** phase where **`401312`** would necessarily recur. To re-test **INVALID_IMAGE** vs. Hopper GGML: use a **longer** HTTP timeout (e.g. **2–4 h**) or **preload** the model, then tail **`module-load`** on the host.

**VM (same window):** `[cuda-transport] mmap shmem 256 MB failed: Resource temporarily unavailable` → **BAR1** path; **`poll call_id=0x0032`** with progress lines — transport active.

---

## Mar 22 plan execution (`SYSTEMATIC_ERROR_TRACKING_PLAN.md` §6)

### Step 1 — Baseline (A / B / C)

| Gate | Result |
|------|--------|
| **A** | `ollama` **active**; `journalctl -b` last **`inference compute`** lines include **`compute=9.0`** (current boot / PID **333441**); older lines on same boot still show **8.9** from prior PIDs — use **last** line for CC. |
| **B** | `/usr/lib64/libvgpu-cuda.so` present (Mar 20); **`strings`** shows **`CC=9.0`** / live GPU info paths. |
| **C (host)** | `grep -E 'module-load\|401312\|INVALID_IMAGE\|rc=200'` — pattern still includes **`401312` → `INVALID_IMAGE`** (historical pairs); **vm_id=9**. |

**Step 1 summary line:** **E1 present:** yes (in log tail); **compute (latest):** **9.0**; **vm_id:** **9**.

### Step 2 — Source of `inference compute` log (repo)

- **`discover/types.go`** ~L42: `slog.Info("inference compute", …, "compute", dev.Compute(), …)` — field is **`ml.DeviceInfo.Compute()`** (not NVML directly in this snippet).
- **`discover/runner.go`** / **`discover.GPUDevices`**: builds **`[]ml.DeviceInfo`** from runner bootstrap over **`libggml-*`** backends.
- **Pass criterion (E2):** Latest boot line **`compute=9.0`** — **met** for PID **333441** after **Hopper `libggml-cuda.so`** + GGML patch.

### Checkpoint D — Long generate (`curl -m 7200`, VM local)

**Executed:** `curl` POST `/api/generate` (`tinyllama:latest`, `num_predict=16`), **7200 s** client timeout; **`CONNECT_VM_COMMAND_TIMEOUT_SEC=8000`**.

| Outcome | Detail |
|---------|--------|
| **HTTP** | **`curl: (28)`** — timed out at **2 h**, **0 bytes** (client waited full window). |
| **Host** | **`HtoD progress`** reached **~708 MB** (vm=9); **`module-load`** tail: **9360 / 28120 / 4432** → **`rc=0`** (success). **`grep '401312' mediator.log`** line count **unchanged (4)** — **no new** `401312` lines this session. |
| **VM** | **~58 min** after request start: runner **`llama_context`** built KV cache, started **graph_reserve** / **`cuGetProcAddress`** spam, then **native crash** → **`llama runner terminated` `error="exit status 2"`** (large Go stack + register dump **01:51:23**). |

**Interpretation:** This run **passed** weight **HtoD** and **several** module loads without hitting a **new** **401312** failure; **Phase 1 proof** still blocked by **runner exit 2** during **context/graph** setup (guest native), not by client timeout alone.

### Steps 3–4 (host / libcublas) — unchanged

- **§6 Step 3–4** remain **human (dom0)** actions: **`load_host_module()`** audit, **libcublasLt** alignment per **`FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`** if **E1** reappears on loads that include **401312**.

### Step 5 — Full proof — not achieved

- No **200** JSON response; **`/api/ps`** not validated post-success.

### Follow-up executed (Mar 22 — exit 2 instrumentation)

- **Coredump drop-in:** already **`LimitCORE=infinity`** (`coredump.conf` present).
- **`coredumpctl`:** available on VM; **`coredumpctl list`** currently **empty** (no core from prior crashes — capture on **next** repro).
- **`/tmp/vgpu_call_sequence.log`:** last lines include **`cuModuleLoadFatBinary`**, **`cuModuleGetFunction`**, **`cuStreamCreateWithFlags`**, **`cuStreamDestroy`** — **`vgpu_current_call.txt`** showed **`cuStreamDestroy`** (**0x0063**) at seq **846** when checked.
- **VM `apt`:** partial **dpkg/DKMS** failure logged during an install attempt — **avoid apt upgrades** until fixed; see **`RUNNER_EXIT2_NEXT_STEPS_MAR22.md`**.

**Doc:** **`RUNNER_EXIT2_NEXT_STEPS_MAR22.md`** — gdb / core capture, host stream-destroy correlation, dom0 libcublas steps.

**Code (repo, dom0 deploy):** **`src/cuda_executor.c`** — **`CUDA_CALL_STREAM_DESTROY`**: if guest handle **not** in host map, return **`CUDA_ERROR_INVALID_HANDLE`** (was implicit **SUCCESS**).

### Mar 22 — Checkpoints A–C + D started (post–host deploy)

| Gate | Result |
|------|--------|
| **A** | **`ollama` active**; last **`inference compute`** → **`compute=9.0`** (Mar 21 22:41:53, PID 333441). |
| **B** | **`/usr/lib64/libvgpu-cuda.so`** present; **`strings`** includes **CC=9.0** path. |
| **C** | **E1** still in **`mediator.log`** tail (**401312** → **INVALID_IMAGE**); also successful smaller loads. |
| **D** | Long **`curl`** **`/api/generate`** (**tinyllama**, **`curl -m 7200`**) started **in background** from workspace; log **`/tmp/phase3_checkpoint_d_generate.log`** (poll until **`HTTP_CODE`** or timeout). |

---

## Mar 22: Cross-read synthesis (what PHASE3 history + live logs say is *the* error)

**Scope note:** The `phase3/` tree contains **hundreds** of `.md` files and **700+** tracked paths; prior sessions alternated between (a) **transport/HtoD**, (b) **scheduler/runner/Go** bugs, (c) **host OOM / GEMM / mapping**, and (d) **module load**. **Do not** collapse all of that into one story — but **for Phase 1 today**, one failure mode dominates the documented, log-correlated evidence.

### The error (single sentence)

After **successful** model **HtoD** and **successful** smaller **`cuModuleLoadFatBinary`** loads, the host **`cuModuleLoadFatBinary` for payload `data_len=401312`** returns **`CUDA_ERROR_INVALID_IMAGE` (rc=200)** — the **CUDA driver rejects the fatbin for execution on the physical H100**, not “weights didn’t transmit.”

### Live host confirmation (read-only, `connect_host.py`)

`grep 'module-load' /tmp/mediator.log | tail -50` shows a **repeating pattern**:

- `28120` → **rc=0**  
- `401312` → **rc=200 INVALID_IMAGE**  
- (other sessions) `9360`, `28120`, `4432` → **rc=0** (different call sequence / workload phase)

So **`401312` is still the smoking gun** in the **current** mediator log, not a stale ghost entry.

### What PHASE3 analysis says that fatbin *is* (Mar 21)

**`FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`** (supported by host `cuobjdump` on `/tmp/fail401312.bin`): the blob is **sm_80 / cuBLASLt-style** (`ampere_h16816gemm_*`), **not** the Hopper **`libggml-cuda.so`** TU set. Rebuilding **`libggml-cuda.so`** with **`sm_90`** **does not change** this 401312 image — therefore the failure is **wrong embedded kernel package for H100**, strongly tied to **`libcublasLt.so.12`** under Ollama’s **`cuda_v12`** tree and **how libraries choose kernels from advertised compute capability**.

### Doc drift to ignore (or treat as historical)

- **`CURRENT_STATE_AND_DIRECTION.md`** §“Where it fails” still frames the fix as “GGML has no sm_90” — **superseded** by Mar 21 fatbin analysis (GGML can have sm_90 while **this** load still pulls an **sm_80** Lt image).
- **`OLLAMA_VGPU_REVISIONS_STATUS.md`** (Mar 15) says raw fatbin pointer **resolved INVALID_IMAGE** and module **rc=0** — **contradicts** Mar 19–22 logs for the **second** (401312) load. Treat Mar 15 as **time-slice** / different failure stage; **Mar 19–22 + live grep** win for “what breaks Phase 1 **now**.”

### VM live discrepancy (actionable)

**`journalctl`** still shows Ollama **`inference compute` … `compute=8.9`** (Mar 20) while the installed **`/usr/lib64/libvgpu-cuda.so`** **strings** include **`cuInit() … CC=9.0`** and **`GPU info (live) … CC=%d.%d`**. So **either** Go discovery reads capability from a **path that bypasses** the forced `g_gpu_info`, **or** it formats **major/minor as 8.9** from another probe — this **must** be traced in Ollama **`discover`/`runner`** (not yet fully in-repo) until **`compute=9.0`** appears in that log line.

### Permissions-aligned “what actually fixes Phase 1”

| Who | Action |
|-----|--------|
| **You (host)** | Ensure **`cuda_executor` `load_host_module()`** matches revision notes (raw `cuModuleLoadFatBinary` for `0xBA55ED50` **first** — see **`OLLAMA_VGPU_REVISIONS_STATUS.md`**); if **`INVALID_IMAGE` persists**, replace **`cuda_v12`** **`libcublasLt`/`libcublas`** with builds whose embedded kernels/PTX are **valid for dom0 driver + H100**, then re-`cuobjdump` **`fail401312.bin`** after a generate. |
| **Assistant (VM)** | Keep **`/opt/vgpu/lib`** free of **`libcublas*.12`** per **`GPU_MODE_DO_NOT_BREAK.md`**; redeploy shims from repo; **trace** why Ollama still logs **`compute=8.9`**; long **`curl`** + **`journalctl`** correlation after each change. |

---

## Mar 21: Fatbin root cause + CC fix (read this first)

- **Doc:** **`FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`** — 401312 blob is **sm_80 / cuBLASLt-style**; **Hopper `libggml-cuda.so`** alone does not change it; **reported compute=8.9** (from **`gpu_properties.h` 8/9 workaround**) likely steers wrong kernel package.
- **Mitigation applied in repo:** **`guest-shim/gpu_properties.h`** → **CC 9.0**; **`fetch_gpu_info()`** forces **`compute_cap_*`** to **`GPU_DEFAULT_*`** after host overlay.
- **Hopper GGML lib:** local Docker build **`out/libggml-cuda.so`** already deployed to VM (MD5 match). Re-verify **`module-load`** / **`cuobjdump`** after guest-shim reinstall + **`systemctl restart ollama`**.

---

## Goal and Phase 1 (do not lose focus)

- **Overall goal (Phase 3):** All GPU-utilizing software in the VM sees the VGPU-STUB as a GPU and uses the data flow to the host and back.
- **Phase 1 / Stage 1 milestone:** Successfully complete **GPU-mode inference in Ollama** in the VM (proof of the path end-to-end): discovery in GPU mode, model load over vGPU, inference, response back.
- Before diving into tracking/fixing any issue: **confirm Ollama is running and operating in GPU mode** (e.g. `systemctl is-active ollama`, API 200, `journalctl | grep "inference compute".*library=CUDA`). Then fix the blocker so the runner reaches alloc/HtoD and the guest sees completion (MMIO or response_len).

### Current primary blocker (Mar 19 PM → late, host + VM logs)

- **Symptom:** After successful bulk **HtoD (0x0032)**, the **second** **`cuModuleLoadFatBinary` (0x0042)** fails on the host with **`CUDA_ERROR_INVALID_IMAGE` (rc=200)** — guest sees **`STATUS_ERROR` seq=826** and **`MODULE_LOAD chunk failed … total=401312`**.  
- **Not:** “HtoD / transmission of weights failed” — mediator showed **status=0** for 0x32 through high `request_id`.  
- **VM `libggml-cuda.so` (Mar 19 late):** Rebuilt on **this host** with **Docker** `nvidia/cuda:12.4.0-devel-ubuntu22.04`, **`CMAKE_CUDA_ARCHITECTURES=90`**, deployed to VM **`/usr/local/lib/ollama/cuda_v12/libggml-cuda.so`**; VM **sha256** matched **`ac68e2fc46639000b98e22132ebaedc4dacd2f650febadb307cf899642cfaed3`**. **`strings`** shows **`.target sm_90`**.  
- **Still to correlate:** A **grep** of **`/tmp/mediator.log`** can still show **`module-load … INVALID_IMAGE`** for **`data_len=401312`**; **tail -300** at one point had **no** 401312 (current long generate may still be in HtoD). **Next:** time-correlate mediator lines with **`journalctl -u ollama`** / **`vgpu-fp`** after deploy to confirm whether **INVALID_IMAGE** is from the **post–new-.so** run or an **older** session. If it **still** fails after correlation, prioritize **host** `cuModuleLoadFatBinary` path (driver / mediated executor / fatbin semantics), not “missing sm_90 on VM.”  
- **Mar 20 confirmation:** Rebuilt/redeployed guest `libvgpu-cuda.so.1` from current `guest-shim/cuda_transport.c` and re-ran. Host now shows the 401312 payload split as **`65536 * 6 + 8096`** with **FIRST/MIDDLE/LAST** flags (not SINGLE), then `module-load ... data_len=401312` still returns **`CUDA_ERROR_INVALID_IMAGE`**. This rules out the previous “single large chunk transport” hypothesis; remaining root cause is host-side `cuModuleLoadFatBinary` acceptance/compatibility or fatbin-content semantics for this image.
- **Separate UX issue:** HTTP client **`curl -m 2400`** can still end with **499 / context canceled** while load is **~50%** — increase client timeout if you want the server to keep the load session open for slower vGPU paths.

---

## 1. Review result: response_len workaround for HtoD

- **Done:** Ran generate (tinyllama, 120s and 180s timeout) with `/tmp/vgpu_host_response_verify.log` and `/tmp/vgpu_call_sequence.log` cleared.
- **Result:** No `SUBMIT call_id=0x0032` or `0x0030`; no `BREAK reason=RESPONSE_LEN`; no `BREAK reason=TIMEOUT`. Call sequence contained only init/context (cuInit, cuGetGpuInfo, cuDevicePrimaryCtxRetain, cuCtxSetCurrent); HtoD count 0.
- **Conclusion:** The runner **never reached** the first cuMemAlloc or HtoD in these runs. The response_len workaround was **not triggered** and could not be evaluated for HtoD. **Not worthwhile** to keep testing response_len for HtoD until the runner is confirmed to reach the alloc/HtoD path.

---

## 2. Current blocker: runner never sends alloc/HtoD *(historical — pre Mar 18/19)*

- **Observed (older runs):** On generate, the verify log shows only the same 6 RPCs (init + context), each completing with `BREAK reason=STATUS status=0x02`. No SUBMIT for 0x0030 (cuMemAlloc) or 0x0032 (cuMemcpyHtoD_v2).
- **Implication:** The process that performs those 6 RPCs either (a) exits or hands off before model load, or (b) the model-load path never issues the first alloc/HtoD (e.g. falls back to CPU after "unable to refresh free memory", or blocks before first alloc).
- **Superseded:** After sched/GetRunner/LoadModelFromFile fixes, runs **do** reach alloc/HtoD (thousands of 0x0032). **§10** is the **current** failure after HtoD.

---

## 3. "unable to refresh free memory" and refresh patch

- **Journal:** When the generate runner starts, we see `inference compute library=CUDA` then `unable to refresh free memory, using old values`.
- **VM code:** On the VM, `discover/runner.go` line 340 already has the correct refresh order: `bootstrapDevices(ctx, []string{dir, ml.LibOllamaPath}, devFilter)`. So the refresh **patch is applied**.
- **Conclusion:** Refresh may still fail for another reason (e.g. timeout, or a CUDA call in the refresh path failing). The scheduler may still use "old values" and proceed with GPU load; the next step is to confirm whether the **load path** actually uses GPU and where it stops.
- **Rebuild retest (Mar 18):** Rebuilt ollama on VM from current source (refresh patch present), installed, restarted. **"unable to refresh free memory" still appears** → refresh failing for a reason other than dir order. See REFRESH_AND_GPU_DETECTION_INVESTIGATION.md.

---

## 4. Root cause: runner LD_LIBRARY_PATH wrong (fixed)

- **Finding:** The generate **runner** process had **LD_LIBRARY_PATH=/usr/lib64** only (no `/opt/vgpu/lib`). The **serve** process had the correct env from systemd (`/usr/local/lib/ollama/cuda_v12:/opt/vgpu/lib:...`). So the runner was not loading the vGPU shims for model load (alloc/HtoD), which is why we never saw SUBMIT 0x0030/0x0032.
- **Fix applied:** In `llm/server.go`, when passing env to the runner we now **prepend `/opt/vgpu/lib`** to `LD_LIBRARY_PATH` if the value does not already contain it (`transfer_ollama_go_patches.py` and VM apply script). Rebuilt and installed on the VM (PATCHED_SERVER_UPGRADE, BUILD_EXIT=0). VM `server.go` now contains the defensive block.
- **Verification (Mar 17):** Added server-side debug log (`runner env LD_LIBRARY_PATH`). After rebuild and restart, journal shows: `value="LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/opt/vgpu/lib:/usr/local/lib/ollama"` when starting runner. So the runner **now receives** `/opt/vgpu/lib` first. Despite that, verify log and call_sequence still show **no SUBMIT 0x0030/0x0032** and only the same 6 init/context RPCs. So the blocker is **not** LD_LIBRARY_PATH anymore; the runner has the right env but model load (alloc/HtoD) is still not reaching the vGPU shim path.

## 5. Scheduler NumGPU fix (Mar 18) — applied

- **Root cause:** When the API omits `num_gpu`, Go decodes it as **0**. In `server/sched.go`, **`pending.opts.NumGPU == 0`** forces **`gpus = []`** and the load uses CPU, so no alloc/HtoD.
- **Fix:** Patch `sched.go` so we always call `getGpuFn`; if `NumGPU == 0` but GPUs are returned, set `NumGPU = -1` and use that list (see **STAGE1_SCHED_NUMGPU_FIX.md**). Applied on VM via `patch_sched_numgpu.py`; rebuilt and installed.
- **Verification:** After a **clean** restart (ensure new binary is running), trigger generate and check for 0x0030/0x0032 in `vgpu_call_sequence.log`. If still 0, confirm running binary and that `getGpuFn` returns non-empty.

## 6. Confirmed: load path never calls cuMemAlloc (Mar 18)

- In `libvgpu_cuda.c` `cuMemAlloc_v2` we append to `/tmp/vgpu_cuMemAlloc_called.log` on each call. Deployed shim, restarted ollama, ran generate.
- **Result:** `/tmp/vgpu_cuMemAlloc_called.log` stayed **empty** → load runner **never** invokes `cuMemAlloc` → model loads on **CPU**.
- **VM build:** Use **`/usr/local/go/bin/go`** on the VM (verified go1.26.1). Journal shows **gpu_count=1**. **GPULayers fallback (Mar 18):** Patched llm/server.go in two places so that when `len(gpus) > 0 && gpuLayers.Sum() == 0` we force `gpuLayers = ml.GPULayersList{{DeviceID: gpus[0].DeviceID, Layers: []int{0}}}`. Added slog "Phase3 GPULayers fallback applied" in the first path. **Result:** Fallback **never logged** → createLayout is returning **non-empty** gpuLayers, so the server is sending GPU layers to the runner. Load runner still does **not** call cuMemAlloc (cuMemAlloc_called.log empty, 0x0030/0x0032 count 0). **Conclusion:** Problem is runner-side: either the load runner uses CPU backend despite receiving non-empty GPULayers (backend selection at init or GGML ignores GPULayers), or the runner that receives the load is not the one we're tracing. Next: inspect runner/backend selection (allocModel, how CUDA vs CPU is chosen).

## 7. GetRunner cold-load enqueue fix (Mar 18) — root cause for “load_start never seen”

- **Root cause:** In `server/sched.go` **GetRunner**, when `runner == nil` (no model loaded) or `needsReload`, the request was **never** sent to `pendingReqCh`. Upstream ollama has `} else { select { case s.pendingReqCh <- req: ... } }` so that cold-load and reload requests are enqueued; on the VM the `select` was only inside the `if runner != nil && !needsReload` branch, so only “use existing runner” requests were enqueued.
- **Fix:** Added the missing **else** branch so that when runner is nil or needs reload we send the request to `pendingReqCh`. Rebuilt and deployed on VM.
- **Verification:** After cold restart and one generate: **phase3_sched_run_loop.txt** contains `run_loop_calling_load_fn`, **phase3_sched_load_path.txt** contains `load_start` and `before_lock`. So the scheduler load path is now reached. **alloc/HtoD still not seen** (vgpu_cuMemAlloc_called.log missing, no 0x0030/0x0032 in call_sequence) → blocker has moved to NewLlamaServer/runner (see §7b).

## 7b. Blocker: server stuck in LoadModelFromFile (Mar 18)

- **Observation:** On cold generate we see **entry** and **before_load_model** in phase3_newllama_entry.txt; **before_start_runner** and **after_start_runner** are **not** written. So the server blocks inside **llama.LoadModelFromFile(modelPath, llama.ModelParams{VocabOnly: true})** (or the tokenizer branch that leads to it). That call loads the model file with VocabOnly; it may block on I/O, or on CUDA/GGML init inside the library.
- **Next:** Confirm whether LoadModelFromFile returns after long time (slow) or never (deadlock/CUDA). If it returns, we should see after_load_model then before_start_runner; then focus on runner/alloc. If it never returns, inspect llama package / CGo for VocabOnly load path (e.g. CUDA init, file mmap).

---

## 8. Next steps (error tracking)

1. **LoadModelFromFile bypass (Mar 18 — done):** Patched `llm/server.go` to try **model.NewTextProcessor(modelPath)** whenever `tok == nil`. **Result:** Server no longer blocks in C; runner reaches alloc (cuMemAlloc 3×). **Follow-up fixes:** (a) **useOllamaEngine:** Only set true when tok came from the new-engine branch (not Phase3 fallback), so GGUF models use the **llama** runner instead of ollama-engine (avoids panic in reserveWorstCaseGraph). (b) **VRAM layout:** When createLayout returns 0 layers but we have gpus and current gpuLayers > 0, **continue nextOperation** so we proceed to commit with current gpuLayers instead of looping. **Current:** Commit and HtoD confirmed. **Timeout alignment (Mar 18):** OLLAMA_LOAD_TIMEOUT=40m and CUDA_TRANSPORT_TIMEOUT_SEC=2700 set; long generate (curl -m 2400) running. **Latter-part investigation:** curl still running (pid 80251), writing to /tmp/generate_out.json on completion. **call_sequence:** 109 lines, **100× 0x0032 (HtoD)**, **0× 0x0033 (DtoH)**. **verify log:** 0 TIMEOUT, 0 RESPONSE_LEN; last BREAK reason=STATUS for 0x0032 seq 105–107 status=0x02. **Journal:** model load progress 0.11 → 0.14 (14%); [cuda-transport] poll 0x0032 seq up to 109. So load is progressing (HtoD completing successfully, no transport timeouts); inference (DtoH) not reached yet. **Next:** Let run continue to load 1.0; then check for 0x0033 and generate_out.json. If load completes but no DtoH, inspect host kernel launch handling.
2. **If alloc/HtoD still don't appear**
   - Confirm running binary is the new one; ensure `systemctl stop` completes before install.
   - Run with **OLLAMA_DEBUG=1** and capture "load_backend", "using device", or CPU fallback.
   - Check whether **getGpuFn** returns empty (refresh path) so the scheduler still gets no GPUs.

3. **If the runner does use GPU for load but we still see no SUBMIT 0x0030/0x0032**
   - The first alloc/HtoD might be coming from a different process (e.g. a separate loader process that does not use the vGPU shim), or the runner might block before the first CUDA alloc (e.g. waiting on server, file I/O, or refresh).

4. **If the runner falls back to CPU after "unable to refresh free memory"**
   - Dig into why refresh fails despite the correct `dirs` order (e.g. bootstrap timeout, `cuMemGetInfo` failure in refresh context, or library path in that context).

5. **Re-test response_len once alloc/HtoD are observed**
   - After the runner is confirmed to reach the alloc/HtoD path (SUBMIT 0x0030/0x0032 in verify log), run again and check for `BREAK reason=RESPONSE_LEN` vs `BREAK reason=TIMEOUT` to see if the workaround helps.

6. **Sched block before llama.Load() (Mar 19)**
   - When phase3_sched_load_entered.txt is written but phase3_before_llama_load.txt and phase3_load_path.txt are **not**, the server is blocked between start of sched `load()` and the call to `llama.Load()` — i.e. inside **newServerFn()** (create/start runner). See PROPOSAL_WAIT_UNTIL_RUNNER_LAUNCHED_INSTRUMENTATION.md § Load-path entry result.
   - **Instrumentation added:** `patch_sched_after_loading_first_model.py` writes to `/tmp/phase3_sched_after_loading_first_model.txt` immediately after the "loading first model" log in server/sched.go. Run on VM: `python3 patch_sched_after_loading_first_model.py /home/test-4/ollama/server/sched.go`, then rebuild, install, restart, trigger generate.
   - **Interpret:** If phase3_sched_after_loading_first_model.txt is **missing** → block is before the log. If it **exists** but phase3_before_llama_load.txt is missing → block is after the log and before `llama.Load()` (inside newServerFn). Then inspect where newServerFn starts the runner and waits for readiness; fix runner health/port or that wait. Full context: **PHASE3_INVESTIGATION_SUMMARY.md**.
   - **Mar 19 run (after applying patch + rebuild):** All instrumentation files present (sched_load_entered, sched_after_loading_first_model, before_llama_load, load_path); vgpu_call_sequence had **763** alloc/HtoD (0x0030|0x0032). So load path and alloc/HtoD are being reached. Run still failed with **"timed out waiting for llama runner to start: context canceled"** (sched.go:575). Blocker has shifted to **runner start/health timeout**. See **WORK_NOTE_MAR19_SCHED_INSTRUMENTATION.md**.
   - **Mar 19 WaitUntilRunning (long timeout):** With 600s client timeout, journal shows **waitUntilRunnerLaunched** succeeds (runner responded, port 38331). Then **WaitUntilRunning** logs "waiting for llama runner to start responding" and status="llm server loading model"; runner **never** reports ServerStatusReady within 10 min. "context canceled" is the **client** (curl) disconnecting. So the blocker is **runner never reports Ready**, not health check. Progress log added in WaitUntilRunning. See **WORK_NOTE_MAR19_WAIT_UNTIL_RUNNING.md**.
   - **Mar 19 loadModel blocker:** For tinyllama the **llama** runner is used (llamarunner). Instrumentation: file write at start of loadModel() and after **llama.LoadModelFromFile()** returns. Result: **phase3_loadmodel_started.txt** present ("llama"), **phase3_loadmodel_returned.txt** absent after 55s. So the runner blocks inside **llama.LoadModelFromFile()** (C/GGML load over vGPU) and never returns, so it never sets Ready. See **WORK_NOTE_MAR19_LOADMODEL_BLOCKER.md**.
   - **Mar 19 runner crash (83 min, exit 2):** A long run (40+ min client timeout) ended with **llama runner terminated exit status 2**. Journal: goroutine in WaitGroup.Wait for **83 minutes**; **rip=0x7f88670969fc** (native code). Host had already completed HtoD ~1.6 GB, module-load rc=0, post-module allocs. So the runner **crashed** in C/GGML or shim code after host finished its part. VM call sequence: last entries **0x0071** (cuEventCreateWithFlags) and **0x0030** (cuMemAlloc_v2). **Next steps:** Enable coredumps so the next crash yields a core; run `gdb /usr/local/bin/ollama.bin.new /path/to/core` and `bt full` to symbolicate rip. Use `/tmp/vgpu_current_call.txt` (overwritten each request) to see which call is blocking when stuck. See **LOG_ANALYSIS_HOST_VM_MAR19.md**, **CRASH_SYMBOLICATION_AND_COREDUMPS.md**, **enable_coredump_ollama_vm.sh**.
   - **Mar 19 cuGetExportTable (shim):** Journal showed **UNKNOWN UUID** then **CUDA_ERROR_NOT_SUPPORTED** from the shim. **Fix:** `libvgpu_cuda.c` `cuGetExportTable` — try `dlsym(RTLD_NEXT, "cuGetExportTable")` and forward; else return a non-failing wrapper table (`g_context_wrapper`) instead of `NOT_SUPPORTED`. Deploy with `transfer_libvgpu_cuda.py`. After deploy, re-check journal for that pair; continue long generate (`curl -m 2400`, `CONNECT_VM_COMMAND_TIMEOUT_SEC` ≥ 2500) to see if runner reaches JSON response vs exit 2. Doc: **WORK_NOTE_MAR19_CUGETEXPORTTABLE_FALLBACK.md**.
   - **Mar 19 call log clarity:** `cuda_transport.c` `call_id_to_name` extended with **cuMemsetD8/D16/D32_v2** (`0x0035`–`0x0037`) so sequences no longer show `?(call_id)` for memset. Deploy with `transfer_cuda_transport.py`.

---

## 10. Mar 19 PM — `INVALID_IMAGE` on second fat binary (0x0042), log-correlated

**Context:** Ollama stopped for triage; logs pulled via **`connect_host.py`** (host, read-only) and **`connect_vm.py`** (VM).

| Check | Result |
|--------|--------|
| **HtoD** | Host: many `CUDA result sent … call_id=0x32 result.status=0`; guest: `vgpu_call_sequence.log` dominated by **cuMemcpyHtoD_v2**. **Weight transmission path succeeded at the RPC level.** |
| **First module load** | Host: `module-load start … data_len=28120` → **`rc=0` CUDA_SUCCESS**. |
| **Second module load** | Host: `module-load start … data_len=401312` → **`rc=200 CUDA_ERROR_INVALID_IMAGE`**, `module=(nil)`; `CUDA result sent … request_id=826 call_id=0x42 result.status=200`. |
| **Guest** | `[cuda-transport] STATUS_ERROR: call_id=0x0042 seq=826 …`; `MODULE_LOAD chunk failed … total=401312 rc=2`. |

**Protocol:** `0x0042` = **`CUDA_CALL_MODULE_LOAD_FAT_BINARY`** (`cuModuleLoadFatBinary`).

**VM `libggml-cuda.so` (assistant-verified on test-4):**

- Path: `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so` (~187 MB, Mar 15 mtime on VM).
- **`strings | grep sm_90`:** multiple **`.target sm_90`** → Hopper SASS **is present** in the installed bundle.
- **sha256:** `73e478b717095efe4a02382b9d6430b808e0d4bc347aab2256bf2fa4732babd9`

**Conclusion:** The failure is **after HtoD**, at **host module load** of the **401312-byte** fat binary. Because the on-disk GGML CUDA library **already shows sm_90**, treat **`INVALID_IMAGE`** here as **not fully explained by “missing Hopper build” alone** — add hypotheses: **chunk reassembly / corruption**, **fatbin vs host driver**, or **wrong embedded image** for that specific load.

**Next steps (within permissions):**

1. **VM (done eve Mar 19):** Guest **`cuda_transport.c`**: for **`CUDA_CALL_MODULE_LOAD_FAT_BINARY`** when **`send_len > 64 KiB`**, cap chunk size to **64 KiB** (FIRST/MIDDLE/LAST on host). **`libvgpu_cuda.c`**: fingerprint log → **`/var/tmp/vgpu_fatbinary_fingerprint.log`** (not `/tmp` — **PrivateTmp** hides service `/tmp` from SSH). **Re-test result (same session):** host still reports **`module-load done … data_len=401312 … rc=200 INVALID_IMAGE`** — transport path change **did not** clear the error; likely **driver/kernel validity** of that fat binary on H100, or further guest/host byte comparison needed.    
2. **VM:** If still **`INVALID_IMAGE`**, optional redeploy of a freshly built `libggml-cuda.so` via **`deploy_libggml_cuda_hopper.py`**.  
3. **Host (user, optional):** Compare fingerprint / hexdump with executor `first8` lines; only if guest fix insufficient.

**Doc:** **`WORK_NOTE_MAR19_INVALID_IMAGE_SECOND_FATBINARY.md`**.

---

## 11. References

- **Investigation summary (goals, authority, config, errors, next steps):** `PHASE3_INVESTIGATION_SUMMARY.md`
- **Host/VM log analysis (Mar 19), runner crash (83 min, exit 2):** `LOG_ANALYSIS_HOST_VM_MAR19.md`
- **Crash symbolication, coredumps, current-call file:** `CRASH_SYMBOLICATION_AND_COREDUMPS.md`
- **Host cuda_executor: event/stream invalid-handle now returns errors (not silent success):** `WORK_NOTE_HOST_EVENT_STREAM_FIX.md` — rebuild/deploy executor on **GPU host** (assistant does not deploy to host per ASSISTANT_PERMISSIONS.md).
- **cuGetExportTable fallback (guest shim):** `WORK_NOTE_MAR19_CUGETEXPORTTABLE_FALLBACK.md`
- **Second fat binary INVALID_IMAGE (Mar 19 PM):** `WORK_NOTE_MAR19_INVALID_IMAGE_SECOND_FATBINARY.md`
- **Verification log format:** `HOST_RESPONSE_VERIFY_LOG.md`
- **Runner env / discovery:** `ROOT_CAUSE_RUNNER_SUBPROCESS.md`, `OLLAMA_RUNNER_LD_PRELOAD_PATCH.md`, `DISCOVER_REFRESH_CUDA.md`
- **HtoD / MMIO:** `HtoD_DIAGNOSIS_RESULTS.md`, `MMIO_WORKAROUND_RESPONSE_LEN.md`, `VERIFICATION_REPORT.md`
