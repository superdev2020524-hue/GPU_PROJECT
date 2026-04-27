# Assistant role and anti-coupling (Phase 3)

*Added: Mar 18, 2026 - must be followed when working in PHASE3.*

---

## 0. Server 2 role registration and isolation (mandatory)

For the current Server 2 mission, the assistant must follow
**`SERVER2_ISOLATION_AND_MISSION_RULES.md`** as part of this role.

### 0.1 Registered working boundary

- **Editable / authoritative for this mission:** `server2/phase3/`
- **Protected parallel track:** root `phase3/` (Server 1)
- **Current host track:** Server 2 on `10.25.33.20`
- **Current VM track:** the Server 2 target VM chosen by the user

### 0.2 Non-mixing rule

1. Do not edit the root `phase3/` tree for this Server 2 mission.
2. If a useful procedure exists only in the root `phase3/` history, read it,
   then copy or adapt it under `server2/` before modifying it.
3. Treat any root `phase3/` document as Server 1 unless the user explicitly
   says otherwise.
4. When reporting progress, keep Server 2 conclusions grounded in
   `server2/phase3/` artifacts, scripts, and live evidence.

### 0.3 Mission priority

Optimize for the fastest working Server 2 delivery that preserves:

- `lspci` HEXACORE presentation in the VM
- normal operation of general GPU software in the VM
- easy repetition on a fresh VM using simple host commands plus VM-side SSH work

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
2. **Name errors** using the plan’s **registry** (E1/E2/E3…): each entry = **log proof** (host line + VM line where applicable), not a vague symptom.
3. **Correlate** VM and host logs from the **same session** (timestamp / `vm_id`); do not conclude “transmission failed” if **Checkpoint C** shows **module-load** `rc=200` on **401312** while HtoD succeeded.
4. **Update** **`ERROR_TRACKING_STATUS.md`** when a run **changes** the registry (new signature, E1 cleared, E2 resolved, etc.) — one short dated note is enough.
5. **Permissions unchanged:** host remains read-only for the assistant; steps that require dom0 edits are **documented for the human** per **`ASSISTANT_PERMISSIONS.md`**.

### 5.2 Required content in assistant status updates

Every substantive progress report to the user **must** include:

| Item | Content |
|------|--------|
| **Checkpoints** | Which of A / B / C / D were run and **pass/fail** |
| **Registry** | E1 / E2 / E3 — **observed / not observed / unknown** this session |
| **Evidence** | At least **one** host line and **one** VM line (or “host unreachable”) |
| **Next step** | **Exactly one** next checkbox from **`SYSTEMATIC_ERROR_TRACKING_PLAN.md` §6** (or the plan’s current ordered list) |

Skipping this structure is **out of role** for PHASE3 triage unless the user explicitly asks for a one-line answer only.

### 5.3 What to avoid

- **No** 40+ minute generate **without** Checkpoint **C** unless the **explicit** goal is post-module behavior and **C** was already clean for that experiment.
- **No** mixing dated narratives (e.g. Mar 15 “INVALID_IMAGE resolved”) with **current** `mediator.log` **without** re-grepping the log tail.
