# Full investigation: goal, authority, current vs previous error, GPU mode, refresh

*Created: Mar 18, 2026 — no excuses; documented facts and gaps.*

---

## 1. Overall goal and Phase 1 milestones (authoritative)

**Source:** PHASE3_PURPOSE_AND_GOALS.md, ERROR_TRACKING_STATUS.md.

| Level | Statement |
|-------|-----------|
| **Ultimate goal (Phase 3)** | All GPU-utilizing software in the VM sees the VGPU-STUB as a GPU and uses the data flow: guest → shims → VGPU-STUB → host mediator → physical GPU → results back. General-purpose vGPU remoting, not Ollama-only. |
| **Phase 1 / Stage 1 milestone** | Successfully complete **GPU-mode inference in Ollama** in the VM as **proof of the path end-to-end**: discovery in GPU mode, model load over vGPU, inference on host H100, **response back to the VM**. |
| **Phase 1 not achieved when** | No single full generate completes with GPU load + inference + returned response. So far: transfer and/or module load can succeed, then runner exits with status 2 → **Phase 1 is not done.** |

**Implication:** Repeating “transfer completed, module load succeeded, then exit 2” does **not** satisfy Phase 1. The milestone requires a **successful end-to-end generate**. I have been reporting the same post-transfer failure pattern without closing the loop to a working generate.

---

## 2. Authority granted to me

**Source:** ASSISTANT_PERMISSIONS.md (Mar 18, 2026).

| Scope | Allowed | Not allowed |
|-------|--------|-------------|
| **Host** | Read host logs and file contents (e.g. `/tmp/mediator.log`). | No editing of host files. No copy to host, no build, no `make`, no restart of mediator or stub. |
| **VM (test-4)** | Full: run commands, configure, deploy, edit VM files, read logs, rebuild and install (ollama.bin, guest shims), restart services. | — |

**Role (ASSISTANT_ROLE_AND_ANTICOUPLING.md):** On error, search PHASE3 first for past resolutions; verify no negative impact on working behavior (e.g. GPU mode, runner env); for VM build always check `/usr/local/go/bin/go version` before concluding the VM cannot build.

**Consequence:** Any **host-side** fix (mediator, cuda_executor, stub) I can only **document**; I cannot apply it. I **can** act on the VM (deploy Hopper libggml-cuda, add logging, rebuild ollama, restart). So if the failure is host-side (e.g. OOM, mediator bug), I have been correctly limited to describing it—but I should have been explicit that Phase 1 is still not achieved and what remains in my scope vs yours.

---

## 3. Current error vs previous errors (no conflation)

| Dimension | Previous errors (documented in PHASE3) | Current error (this run, Mar 18) |
|-----------|----------------------------------------|-----------------------------------|
| **Symptom** | HTTP 500, “llama runner process has terminated: exit status 2” | Same: HTTP 500, exit status 2. |
| **When** | Various: (A) ~7–10 s; (B) after ~5 min timeout (progress 0.00); (C) after long transfer + allocs; (D) after module load + post-module allocs; (E) after GEMM. | **After ~82 minutes**: HtoD completed (686 MB), module load **rc=0**, post-module allocs SUCCESS; then runner crash (register dump, same PID). |
| **Root cause (previous)** | (A) Guest never sees DONE (status stuck 0x01 BUSY) → reply path / Xen-MMIO. (B) Server timeout waiting for runner progress. (C) Host INVALID_IMAGE at module load (no sm_90). (D) Host OOM (cuMemAlloc FAILED rc=2). (E) Crash after GEMM, before launch/sync. | **Different**: Reply path and INVALID_IMAGE are **not** the cause this time (we saw DONE, module-load rc=0). OOM not seen in this run’s host tail. So: **new** failure point = after post-module allocs, before or during next step (e.g. 0x0071 event create, or code between allocs and CUBLAS/launch). |
| **Call sequence (guest)** | Past: often only init/context; or HtoD then block; or 6× cuMemAlloc then crash. | **This run**: HtoD (0x0032) then **0x0071** (event create), **0x0030** (cuMemAlloc). So we are **past** module load and into event/alloc phase; crash **after** that. |
| **Host log (this run)** | — | HtoD to 686 MB; module-load **rc=0**; multiple cuMemAlloc SUCCESS (46M, 205M, 1K, 128K, 8M, 8.8M, 2.2M×2). No INVALID_IMAGE, no cuMemAlloc FAILED in tail. |

**Conclusion:** The **current** error is **not** the same as “guest never sees DONE” or “INVALID_IMAGE at module load.” Those were fixed or bypassed. The **current** failure is **after** module load and post-module allocs—likely in event creation (0x0071), or in guest/host handling of the next CUDA/CUBLAS call, or in GGML/runner (e.g. segfault at rip 0x7e0c9a0969fc). So I was wrong to imply “the same explanation”: the **location** of the failure has moved; the **symptom** (exit 2) is the same. I should have stated that clearly and focused on **where** it fails now (e.g. 0x0071 path, or next call after the last 0x0030).

---

## 4. Is Ollama currently operating in GPU mode?

**Before the crash (same run):**  
- Runner had been up ~82 minutes, using vGPU (HtoD, module load, allocs on host for vm=9).  
- So **during** that run, Ollama was using the GPU path (CUDA backend, vGPU shims, mediator).

**After the crash (current state):**  
- **Ollama service:** `active`; API returns 200.  
- **Journal:** Last 300 lines did not show a new “inference compute” or “library=CUDA” line in the grep; the most recent relevant line was the exit-status-2 error at 18:44:35. So **I did not re-verify** “Ollama is in GPU mode” **after** the crash with a fresh discovery/bootstrap.  
- **Check performed after this investigation:** `systemctl restart ollama` was run on the VM; after 8 s, journal showed (new PID 81607):
- `msg="inference compute" ... library=CUDA ... description="NVIDIA H100 80GB HBM3" ... total="80.0 GiB" available="78.0 GiB"`
- `msg="vram-based default context" total_vram="80.0 GiB"`
- No `filtering device which didn't fully initialize` in the tail.

**Answer:** During the long run, Ollama was in GPU mode. **After restart**, Ollama again detects the GPU (CUDA, 80 GiB); discovery is working.

---

## 5. Does it detect the GPU again after a refresh?

**Source:** REFRESH_AND_GPU_DETECTION_INVESTIGATION.md (Mar 18).

- **Initial discovery:** Confirmed earlier as correct (library=CUDA, 80 GiB, no “filtering device which didn’t fully initialize”).  
- **Refresh:** The message **“unable to refresh free memory, using old values”** still appears when a generate runner starts. The **refresh dir order** fix (`[]string{dir, ml.LibOllamaPath}`) is in the VM source and was rebuilt; the warning persists, so refresh is failing for **another reason** (e.g. refresh bootstrap timeout, or cuMemGetInfo in refresh context).  
- **So:** GPU is detected at **initial** discovery; at **refresh** (during load), the refresh step still fails—we do **not** “detect the GPU again” successfully in that refresh path; the scheduler may be using “old values.”

**Answer:** Initial detection: yes (when last checked). **After refresh:** no—refresh still reports failure (“unable to refresh free memory, using old values”). That is unchanged from the earlier investigation.

---

## 6. Mistakes and what I should do differently

1. **Same-sounding summary:** I repeatedly said “transfer completed, module load succeeded, then exit 2” without clearly saying that (a) the **failure point has moved** (past reply path, past INVALID_IMAGE), and (b) **Phase 1 is still not achieved** and the same symptom does not mean the same root cause.  
2. **Authority:** I did not clearly separate what I can fix (VM) vs what only you can fix (host), and that host-side fixes are documented, not applied by me.  
3. **Verification:** I did not re-run the “confirm Ollama in GPU mode” and “confirm GPU after refresh” checks after the crash and (attempted) restart; I should do that and report the exact journal lines.  
4. **Next step:** Identify the **exact** failing point for this run: e.g. host/guest handling of **0x0071** (CUDA_CALL_EVENT_CREATE_WITH_FLAGS), or the instruction at **rip 0x7e0c9a0969fc** (which library/call). That requires runner stderr, core dump, or targeted logging at 0x0071 and the next call after the last 0x0030.

---

## 7. References

- **PHASE3_PURPOSE_AND_GOALS.md** — Ultimate goal and Phase 1 milestone.  
- **ERROR_TRACKING_STATUS.md** — Goal and Phase 1; confirm GPU mode before fixing.  
- **ASSISTANT_PERMISSIONS.md** — Host read-only; VM full.  
- **ASSISTANT_ROLE_AND_ANTICOUPLING.md** — Search PHASE3 first; verify no regression.  
- **OLLAMA_VGPU_REVISIONS_STATUS.md** — Module load, post-module allocs, exit 2, CUBLAS/GEMM.  
- **REFRESH_AND_GPU_DETECTION_INVESTIGATION.md** — Refresh still fails; initial GPU detection OK.  
- **ACTUAL_ERROR_MAR16_STATUS_AND_SHMEM.md** — Reply path (guest never sees DONE).  
- **CURRENT_STATE_AND_DIRECTION.md** — INVALID_IMAGE, module load.  
- **HOST_LOG_FINDINGS.md** — OOM (cuMemAlloc FAILED rc=2).  
- **include/cuda_protocol.h** — 0x0071 = CUDA_CALL_EVENT_CREATE_WITH_FLAGS.
