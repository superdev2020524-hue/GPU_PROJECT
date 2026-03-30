# Assistant role and anti-coupling (Phase 3)

*Added: Mar 18, 2026 — must be followed when working in PHASE3.*

---

## 1. On error: search PHASE3 first

When an error or failure occurs (build failure, missing tool, unexpected behavior):

1. **First** search and read the **PHASE3 path** (this directory and subdirs) for:
   - How the same or similar case was **resolved in the past** (docs, scripts, comments).
   - Existing procedures (e.g. VM build: use `/usr/local/go/bin/go`; see BOOTSTRAP_FIX_SKIP_CUDA_INIT_VALIDATION.md, CURRENT_STAGE_SUMMARY_AND_NEXT_STEPS.md).
2. **Do not** assume something is missing or impossible (e.g. “Go 1.23+ is not on the VM”) **without verifying on the target** (e.g. run `/usr/local/go/bin/go version` on the VM).
3. If not found in PHASE3, then search the rest of the repo or find a new solution; document the resolution in PHASE3 for next time.

---

## 2. Anti-coupling: verify no negative impact

When implementing a fix or new behavior:

1. **Before** considering the fix “done”: explicitly verify that **previously working behavior still works** (e.g. Ollama in GPU mode, discovery showing `library=CUDA`, runner env with `/opt/vgpu/lib`, refresh path, existing patches).
2. **Do not** only check that the new code path has no errors; also confirm that:
   - No working code paths were removed or overwritten.
   - No config or env that was correct was reverted or replaced incorrectly.
3. If a change might affect multiple components (e.g. sched + discovery + runner env), **re-verify each** after the change (e.g. journal “inference compute” CUDA, `LD_LIBRARY_PATH` for runner, alloc/HtoD in call sequence).
4. Document what was verified (and, if needed, what was reverted) so the next step does not assume something was “destroyed.”

---

## 3. VM build: always check Go path

- The VM may have **both** a system Go (e.g. 1.18) and **Go 1.23+** at `/usr/local/go/bin/go`.
- **Always** run on the VM: `go version` and `/usr/local/go/bin/go version` before concluding “the VM cannot build.”
- Use **`/usr/local/go/bin/go build`** for Ollama when available (see BOOTSTRAP_FIX_SKIP_CUDA_INIT_VALIDATION.md, transfer_ollama_go_patches.py, install_go_and_build_ollama_on_vm.py).

---

## 4. Error triage and escalation protocol (mandatory)

When any new error appears or a run regresses:

1. **Immediate health check (VM + host first):**
   - VM: verify `ollama` service is active, API is reachable, and journal still indicates GPU path (`library=CUDA`, tensor offload lines, load progress).
   - Host: verify mediator is processing VM calls (`vm_id`, `0x0032`/HtoD progress, `0x0042` module-load results).
2. **Report errors immediately:** summarize exact signatures (call_id, rc/error name, seq, timestamps) before changing anything.
3. **Search PHASE3 history first:** check prior docs/work notes for the same signature and apply known verification/fix flow if it matches.
4. **If not matched in PHASE3:**
   - Search global/public sources (NVIDIA docs/forums, project issues, developer discussions).
   - Prioritize architecture/driver/toolchain compatibility guidance and API-specific behavior notes.
5. **If public sources are insufficient:**
   - Use latest technical references (release docs, compatibility guides, and current research literature) and propose the smallest testable hypothesis.
6. **Anti-coupling re-check:** after any fix, re-verify previously working behavior (GPU mode, runner env, load progression, host execution path) did not regress.

---

## 5. Systematic error tracking (mandatory)

**Authoritative procedure:** **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`**. From Mar 22, 2026 onward, any assistant working in PHASE3 **must** treat that document as part of this role: **checkpoints**, **gates**, **error registry**, and **no long blind runs**.

### 5.1 Obligations

1. **Before** starting or recommending a **long** model load / generate (more than a few minutes):
   - Pass **Checkpoint A** (service + `library=CUDA`).
   - Pass **Checkpoint B** (shim presence; note Ollama `inference compute` vs shim CC — **E2** in the plan).
   - Pass **Checkpoint C** (host `grep module-load` / **`401312`** / **`INVALID_IMAGE`**) so the **failing layer** is known **before** waiting on HtoD.
   - **If Checkpoint C is clean** (no **`401312`** / **`INVALID_IMAGE`** in the **current** `mediator.log`) **but** the active blocker is **post-load compute**, continue with **E4** (**`rc=700`** / **`CUDA_ERROR_ILLEGAL_ADDRESS`**) tracking: host lines around **`cublasGemmBatchedEx`** and **`cuCtxSynchronize`**, guest **`libvgpu_cublas.c`** RPC path, and executor **`CUDA_CALL_CUBLAS_GEMM_BATCHED_EX`** in **`cuda_executor.c`**. Do not assume **E1** is still the primary failure without re-grepping the **current** log.
2. **Name errors** using the plan’s **registry** (E1/E2/E3/**E4**…): each entry = **log proof** (host line + VM line where applicable), not a vague symptom.
3. **Correlate** VM and host logs from the **same session** (timestamp / `vm_id`); do not conclude “transmission failed” if **Checkpoint C** shows **module-load** `rc=200` on **401312** while HtoD succeeded.
4. **Update** **`ERROR_TRACKING_STATUS.md`** when a run **changes** the registry (new signature, E1 cleared, E2 resolved, etc.) — one short dated note is enough.
5. **Permissions:** follow **`ASSISTANT_PERMISSIONS.md`**. As of **2026-03-25**, the operator **granted** the assistant **dom0 edit/build/restart** for PHASE3 work under the **non-destruction** condition; do **not** treat the host as read-only unless that file is **updated** to revoke or narrow the grant.

### 5.2 Required content in assistant status updates

Every substantive progress report to the user **must** include:

| Item | Content |
|------|--------|
| **Checkpoints** | Which of A / B / C / D were run and **pass/fail** |
| **Registry** | E1 / E2 / E3 / **E4** — **observed / not observed / unknown** this session (**E4** = **`rc=700`** **`CUDA_ERROR_ILLEGAL_ADDRESS`** after **`cublasGemmBatchedEx`** / **`cuCtxSynchronize`**, see **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`**) |
| **Evidence** | At least **one** host line and **one** VM line (or “host unreachable”) |
| **Next step** | **Exactly one** next checkbox from **`SYSTEMATIC_ERROR_TRACKING_PLAN.md` §6** (or the plan’s current ordered list) |

Skipping this structure is **out of role** for PHASE3 triage unless the user explicitly asks for a one-line answer only.

### 5.3 What to avoid

- **No** 40+ minute generate **without** Checkpoint **C** unless the **explicit** goal is post-module behavior and **C** was already clean for that experiment.
- **No** mixing dated narratives (e.g. Mar 15 “INVALID_IMAGE resolved”) with **current** `mediator.log` **without** re-grepping the log tail.

### 5.4 Long-duration model loading — operator approval required

**Binding for assistants:** **Long-term** model loading or generate (e.g. multi-hour **`curl`**, **`OLLAMA_LOAD_TIMEOUT`-scale** runs, **`reset_and_start_longrun_4h.sh`**, or any client/server window **intended** to span **tens of minutes or more** of wall time for a single load) **must not** be started or recommended **unless the operator has explicitly approved** that run in the conversation.

- **Short** bounded checks (preflight binaries, **`curl -m`** of a **few minutes**, **`/api/tags`**, checkpoint **A–C** greps, **`journalctl`** tails) are **not** “long-term model loading” and remain in scope without special approval.
- If approval is unclear, **default to no** long run and ask once.

This rule **supplements** §5.1–5.3 (checkpoints and gates); it does **not** replace them.

### 5.5 Transmission / load-performance is a first-class track

The assistant must **not** treat "the model eventually loads" as sufficient progress if the
weight-transfer path is unacceptably slow.

From now on, Phase 3 work must pursue **both** tracks in parallel:

1. **Correctness / error tracing**
   - reproduce, classify, and eliminate current blockers such as E1/E3/E4/E5
   - keep the host/guest evidence chain for the active failure
2. **Transmission / load-performance**
   - explain why load time is slow
   - verify whether the real data path is **shared memory** or **BAR1**
   - identify serialization points in guest transport, vgpu-stub, mediator, and executor
   - reduce model-load wall time as part of Phase 3, not as an optional future task

Minimum required checks when model load is being discussed or tested:

- VM evidence showing the active transport path (`shmem` registered vs `using BAR1`)
- Host evidence showing `HtoD progress`
- whether the run is using the intended fast path or a fallback path
- whether the current architecture is still single-flight / blocking for `cuMemcpyHtoD`

Do **not** present shared memory as "the current fast path" unless the live run proves it is
actually active.
