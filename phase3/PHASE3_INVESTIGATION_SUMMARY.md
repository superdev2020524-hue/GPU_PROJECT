# Phase 3 — Investigation Summary

*Created: Mar 19, 2026 — from full review of PHASE3 path (documentation and code).*

---

## 1. Overall goals (Phase 3)

| Level | Goal |
|-------|------|
| **Ultimate (Phase 3)** | All GPU-using software in the VM sees the **VGPU-STUB** as a GPU and uses the data path: guest → shims → VGPU-STUB → host mediator → physical GPU (host CUDA) → results back. General-purpose vGPU remoting, not Ollama-only. |
| **Phase 1 / Stage 1 milestone** | Prove the path end-to-end by completing **GPU-mode inference in Ollama** in the VM: discovery in GPU mode, model load over vGPU, inference on host H100, **response back to the VM**. |

**Success criterion for Phase 1:** At least one full generate completes (model load + inference over vGPU → host H100 → HTTP 200 with response). Until that happens, Phase 1 is not done.

**Refs:** PHASE3_REVIEW_AND_RESUME.md, ERROR_TRACKING_STATUS.md § Goal and Phase 1, INVESTIGATION_FULL_ACCOUNTABILITY.md.

---

## 2. Phase 1 milestones (checklist)

| Milestone | Description | Status (from docs) |
|-----------|-------------|--------------------|
| **Discovery** | Ollama sees vGPU as CUDA device (library=CUDA, "NVIDIA H100 80GB HBM3", total="80.0 GiB"). | ✓ Done (skip CUDA init validation in ml/device.go). |
| **Runner env** | Runner subprocess gets `LD_LIBRARY_PATH` with `/opt/vgpu/lib` first so it loads guest shims. | ✓ Done (llm/server.go prepends `/opt/vgpu/lib`). |
| **Scheduler NumGPU** | When API omits `num_gpu` (decodes as 0), scheduler still uses GPU list from `getGpuFn`. | ✓ Done (patch_sched_numgpu.py). |
| **GetRunner cold-load** | Cold-load and reload requests are enqueued to `pendingReqCh`. | ✓ Done (else-branch in sched GetRunner). |
| **Load path** | Load runner uses GPU: issues `cuMemAlloc` / `cuMemcpyHtoD` via shims (not CPU-only). | ❌ Blocker: load path often does not reach alloc/HtoD. |
| **Past LoadModelFromFile** | Server does not block in C in `LoadModelFromFile` (VocabOnly). | ✓ Bypass with `model.NewTextProcessor(modelPath)` when `tok == nil`. |
| **Past newServerFn / runner ready** | Sched reaches `llama.Load()`; runner responds to health/status so load is sent. | ❌ Observed: sched load entered, but **never** reaches line before `llama.Load()` — **newServerFn() likely blocking**. |
| **Module load (host)** | Host loads CUDA fat binary (e.g. Hopper sm_90). | Known failure without Hopper build: INVALID_IMAGE; fix: BUILD_LIBGGML_CUDA_HOPPER.md + deploy to VM. |
| **End-to-end** | One full generate completes. | Not achieved. |

---

## 3. Authority granted to the assistant

**Source:** ASSISTANT_PERMISSIONS.md, ASSISTANT_ROLE_AND_ANTICOUPLING.md.

| Scope | Allowed | Not allowed |
|-------|--------|-------------|
| **Host** (e.g. 10.25.33.10) | Check host logs (e.g. `/tmp/mediator.log`); **read** file contents for investigation. | No **editing** of host files; no copy to host, no build, no `make`, no restart of mediator or stub. |
| **VM (test-4)** | **Full:** run commands, configure, deploy guest artifacts, edit VM files, read VM logs, rebuild and install (e.g. ollama.bin, guest shims), restart services. | — |

**Role:** On error, search PHASE3 first for past resolutions; verify no negative impact on working behavior (e.g. GPU mode, runner env); for VM build always check `/usr/local/go/bin/go version` before concluding the VM cannot build.

**Consequence:** Host-side fixes are **documented** only; you apply them on the host. VM-side fixes are implemented by the assistant via VM interaction (scripts, patches, deploy).

---

## 4. Host and VM configuration (from docs and code)

### VM (test-4)

- **Connection:** `test-4@10.25.33.12` (phase3/vm_config.py: VM_USER, VM_HOST). Password auth via connect_vm.py (pexpect).
- **Ollama:** Service `ollama`; binary often installed as `/usr/local/bin/ollama.bin.real` or `ollama.bin.new`. Build on VM with `/usr/local/go/bin/go` (Go 1.26.1 verified).
- **Guest shims:** `/opt/vgpu/lib` (e.g. libvgpu-cuda.so, libvgpu-cudart.so). Used via `LD_LIBRARY_PATH` for both serve and runner.
- **Discovery:** CUDA device reported as "NVIDIA H100 80GB HBM3", 80 GiB; vGPU-STUB backed by host mediator. Skip CUDA init validation (ml/device.go) so discovery does not filter the device.
- **Refresh:** Message "unable to refresh free memory, using old values" still appears; refresh dir order is correct in source; failure is another cause (timeout or CUDA in refresh path).

### Host (mediator)

- **Connection:** `root@10.25.33.10` (vm_config.py: MEDIATOR_HOST, MEDIATOR_USER). Read-only for assistant.
- **Process:** `mediator_phase3`; logs in `/tmp/mediator.log`.
- **Role:** Receives CUDA RPCs from VMs (e.g. vm_id=9 for test-4), dispatches to cuda_executor on physical H100, returns results to stub/shim.
- **Current behavior:** When load path works, mediator sees init/context then alloc/HtoD (0x0030, 0x0032); when it does not, only init/context (no 0x0030/0x0032). Post-transfer failure: `cuModuleLoadFatBinary` can return INVALID_IMAGE if VM’s libggml-cuda.so lacks sm_90.

### Data path

Guest process → (LD_PREload / LD_LIBRARY_PATH) → guest shims in `/opt/vgpu/lib` → cuda_transport / protocol → VGPU-STUB (MMIO/chunked) → host mediator → cuda_executor on physical H100 → results back.

---

## 5. Errors currently encountered (Phase 1)

| Error | Cause / status |
|-------|-----------------|
| **Load path never reaches alloc/HtoD** | Verify log and vgpu_call_sequence.log show only init/context (0x0001, 0x00f0, 0x0090, 0x0022); no SUBMIT 0x0030 (cuMemAlloc) or 0x0032 (cuMemcpyHtoD). `/tmp/vgpu_cuMemAlloc_called.log` empty → load runner does not call cuMemAlloc → model load on CPU or load never reaches runner. |
| **Sched never reaches llama.Load()** | phase3_sched_load_entered.txt is written (sched.load() entered) but phase3_before_llama_load.txt and phase3_load_path.txt are **not** written. So execution is **blocked** between start of sched `load()` and the call to `llama.Load()`. The only non-trivial work there is **newServerFn()** (create/start runner). Conclusion: **newServerFn() is likely blocking** (e.g. wait for runner to become ready never completes). |
| **Runner load handler not hit** | When server logs "loading first model", "Phase3 sending load to runner" never appears; runner’s load() in runner/ollamarunner/runner.go is not called (runner_load_entered.txt / runner_load_gpulayers.txt missing). So either server never sends load (stuck in waitUntilRunnerLaunched or earlier) or wrong process/port. |
| **waitUntilRunnerLaunched not entered** | phase3_wait_entered.txt not created; no "Phase3 waitUntilRunnerLaunched" in journal. So the code path that would call waitUntilRunnerLaunched (inside Load()) is never reached — consistent with sched blocking in newServerFn() before llama.Load() is ever called. |
| **"unable to refresh free memory"** | Still present; refresh dir order is correct; refresh fails for another reason (timeout, refresh-runner env, or cuMemGetInfo in refresh path). Scheduler may use "old values" and still proceed with GPU list. |
| **(After transfer) INVALID_IMAGE** | Host `cuModuleLoadFatBinary` returns rc=200; VM’s libggml-cuda.so not built for sm_90. Fix: build with CMAKE_CUDA_ARCHITECTURES=90 and deploy (BUILD_LIBGGML_CUDA_HOPPER.md). |

---

## 6. How previous work was implemented (verification)

Patches and scripts are applied **on the VM** to the Ollama tree at `/home/test-4/ollama/` (paths in scripts may use that literal). Connection and deploy use phase3 scripts; host is read-only.

| Fix | How it was applied | Verification |
|-----|--------------------|--------------|
| **Skip CUDA init validation** | ml/device.go: `NeedsInitValidation` → return true only for ROCm. | Discovery shows CUDA, 80 GiB; no "filtering device which didn't fully initialize". |
| **Runner LD_LIBRARY_PATH** | llm/server.go: prepend `/opt/vgpu/lib` when passing env to runner. | Journal shows runner env with `/opt/vgpu/lib` first. |
| **Refresh dir order** | discover/runner.go: `bootstrapDevices(ctx, []string{dir, ml.LibOllamaPath}, ...)`. | Source has correct order; "unable to refresh" still appears (other cause). |
| **Scheduler NumGPU** | `patch_sched_numgpu.py` on server/sched.go: always call getGpuFn; if NumGPU==0 and GPUs returned, set NumGPU=-1. | `grep "Phase3/vGPU" /home/test-4/ollama/server/sched.go` → line 205. |
| **GetRunner cold-load enqueue** | server/sched.go: add else branch so cold-load/reload requests go to pendingReqCh. | phase3_sched_run_loop.txt has run_loop_calling_load_fn; phase3_sched_load_path.txt has load_start, before_lock. |
| **LoadModelFromFile bypass** | llm/server.go: when tok==nil use model.NewTextProcessor(modelPath). | Server no longer blocks in C; runner can reach alloc (cuMemAlloc 3×) when load path is used. |
| **NewLlamaServer / StartRunner entry** | patch_newllamaserver_entry.py: file writes at entry and after StartRunner. | phase3_newllama_entry.txt shows entry, before_load_model; after_start_runner appears after bypass. |
| **Load path entry (ollama/llama Load)** | patch_load_path_entry.py: write to phase3_load_path.txt at start of ollamaServer.Load and llamaServer.Load. | Used to see that sched never reaches llama.Load() (phase3_load_path never written in blocking scenario). |
| **waitUntilRunnerLaunched instrumentation** | patch_wait_until_runner_launched.py: poll counter and slog in waitUntilRunnerLaunched. | Confirmed waitUntilRunnerLaunched is not entered when sched blocks in newServerFn. |
| **Runner GPULayers log** | inject_runner_load_gpulayers_log.py in runner load handler. | Used to confirm load handler not hit when server never sends load. |
| **cuMemAlloc shim log** | libvgpu_cuda.c: append to /tmp/vgpu_cuMemAlloc_called.log on cuMemAlloc_v2. | Empty when load runner does not use GPU path. |

Deploy: transfer scripts to VM (e.g. via scp or shared tree), run Python patch scripts against VM paths, rebuild with `/usr/local/go/bin/go build -o ollama.bin .`, install binary (e.g. cp to /usr/local/bin/ollama.bin.real), restart ollama. Host-side changes (e.g. cuda_executor primary context) are documented only; not applied by assistant.

---

## 7. Resolution plan and next steps

### Root cause (current blocker)

Execution blocks **inside sched.load()** after the function is entered and **before** the line that calls `llama.Load()`. The only substantial work in that span is the call that creates/starts the server (newServerFn or equivalent). So that call is either blocking indefinitely (e.g. waiting for runner health) or not returning for another reason.

### Intended resolution

1. **Confirm exact block location**  
   Add instrumentation in **server/sched.go** in the span between “load() entered” and “call llama.Load()”:
   - Write to a file (e.g. `/tmp/phase3_sched_after_loading_first_model.txt`) immediately **after** the line that logs "loading first model".
   - This narrows whether the block is before or after that log. If phase3_sched_load_entered exists but phase3_sched_after_loading_first_model does not, the block is before the log; if both exist but phase3_before_llama_load does not, the block is between the log and llama.Load() (i.e. inside newServerFn / server creation).

2. **If block is inside server creation / runner start**  
   - Ensure the runner process actually starts and listens (e.g. correct port; pgrep -fa "ollama.*runner", ss -tlnp).
   - Ensure the runner’s HTTP server responds to the health/status check (e.g. GET /health) that the server uses before sending load. If the server waits on getServerStatus() inside waitUntilRunnerLaunched but never reaches Load() because it blocks earlier (in newServerFn), then newServerFn likely contains a similar wait — inspect that path (e.g. where the runner is started and where it waits for “ready”).
   - Fix: either make the runner respond correctly to that check, or fix port/process so the server talks to the right runner, or adjust the wait (e.g. timeout, or skip wait if not required for first load).

3. **Once load path reaches alloc/HtoD**  
   - Re-verify: vgpu_cuMemAlloc_called.log and vgpu_call_sequence.log show 0x0030/0x0032; host mediator shows alloc/HtoD for vm=9.
   - If host then fails with INVALID_IMAGE at module load: deploy Hopper-built libggml-cuda.so per BUILD_LIBGGML_CUDA_HOPPER.md.

4. **Ongoing**  
   - After any change, re-verify: Ollama in GPU mode (journal “inference compute” library=CUDA), runner env, and no regression in discovery or sched (ASSISTANT_ROLE_AND_ANTICOUPLING.md).

### Script added for next step

- **patch_sched_after_loading_first_model.py**  
  Injects a single file write in server/sched.go immediately after the line containing the string `"loading first model"`, writing to `/tmp/phase3_sched_after_loading_first_model.txt`. Run on VM:  
  `python3 patch_sched_after_loading_first_model.py /home/test-4/ollama/server/sched.go`  
  Then rebuild, install, restart ollama, trigger one generate, and check:
  - If phase3_sched_load_entered.txt exists but phase3_sched_after_loading_first_model.txt does not → block is **before** the "loading first model" log.
  - If both exist but phase3_before_llama_load.txt does not → block is **after** that log and **before** llama.Load() (i.e. in newServerFn / server-creation path).

---

## 8. References (key docs in PHASE3)

- **CURRENT_STATE_AND_DIRECTION.md** — Pipeline, blocker (INVALID_IMAGE after transfer), permissions.
- **ERROR_TRACKING_STATUS.md** — Goal, Phase 1, all fixes (GetRunner, NumGPU, LoadModelFromFile bypass, etc.) and current blockers.
- **PHASE3_REVIEW_AND_RESUME.md** — Goals, milestones, current blocker (load runner not calling cuMemAlloc), resume steps.
- **ASSISTANT_PERMISSIONS.md** — Host read-only; VM full.
- **ASSISTANT_ROLE_AND_ANTICOUPLING.md** — Search PHASE3 first; verify no regression; VM build with /usr/local/go/bin/go.
- **PROPOSAL_WAIT_UNTIL_RUNNER_LAUNCHED_INSTRUMENTATION.md** — waitUntilRunnerLaunched and load-path entry instrumentation; conclusion that newServerFn() is likely blocking.
- **NEXT_STEP_LOAD_HANDLER_NOT_HIT.md** — Load handler not called; server never sends load (stuck before initModel).
- **REFRESH_AND_GPU_DETECTION_INVESTIGATION.md** — Refresh still fails; dir order correct.
- **BUILD_LIBGGML_CUDA_HOPPER.md** — Build and deploy libggml-cuda.so with sm_90 for host module load.
- **VERIFICATION_REPORT_MAR18.md** — VM and host status vs docs.
- **connect_vm.py**, **vm_config.py** — VM connection (test-4@10.25.33.12).
- **connect_host.py** (or host access) — Host log read (read-only).
