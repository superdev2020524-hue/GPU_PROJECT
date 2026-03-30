# Phase 3: Purpose and Goals

*Document created: Mar 15, 2026 — to state clearly the purpose we intend to pursue in PHASE 3.*

---

## Ultimate goal

**We are not a VGPU-STUB that operates solely for Ollama.**

The **ultimate goal** of this work is:

- **All software that utilizes the GPU**, when running in the VM, should **recognize the VGPU-STUB as a GPU**.
- Those applications should **follow the corresponding data-flow path**: guest → shims → VGPU-STUB → host mediator → physical GPU (host CUDA) → results back to the VM.
- So: any GPU-using workload in the VM (Ollama, other ML frameworks, CUDA apps, etc.) sees the vGPU as a normal GPU and gets its compute and data carried to the host and back.

The VGPU-STUB and mediator are **general-purpose** for GPU remoting, not tied to a single application.

---

## First-stage achievement goal

**Successfully completing GPU-mode inference in Ollama in the VM is only the first-stage achievement goal.**

- It is a **milestone** to prove the path end-to-end: discovery, model load, inference, and results over the vGPU pipeline.
- Once Ollama GPU inference works reliably in the VM, the same architecture (shims, stub, mediator, host CUDA) should support other GPU-utilizing software in the VM.
- Do **not** treat Phase 3 as “Ollama-only.” Ollama is the first target; the design goal is broader.

---

## Context: direction change during test-3

- While working on **test-3**, the work had shifted toward **specializing for Ollama** (e.g. Ollama-specific patches, discovery, model load).
- There was also a view that **the host’s CUDA needed to be updated to match the VM’s CUDA** (e.g. version alignment between guest and host).
- **That direction was stopped.** You instructed a **different direction**: the design should not depend on matching host CUDA to VM CUDA in that way, and the purpose should remain the broader one (all GPU software in the VM, not only Ollama).
- **During that process, the test-3 VM was destroyed.** Work continued on test-4 with the clarified purpose.

---

## Summary

| Level | Goal |
|-------|------|
| **Ultimate** | All GPU-utilizing software in the VM sees the VGPU-STUB as a GPU and uses the data flow to the host and back. |
| **First stage** | Successfully complete GPU-mode inference in Ollama in the VM (proof of the path). |
| **Design** | General-purpose vGPU remoting; not Ollama-only. Avoid requiring host CUDA to “match” VM CUDA as a design premise. |

---

## Related documents

- **OLLAMA_INVESTIGATION_PLAN.md** — notes that “vGPU works for general GPU projects (Python, CUDA, etc.)”; investigation there is Ollama-focused as one use case.
- **VGPU_CLIENT_DEPLOYMENT_DIRECTION.md** — client behavior and deployment for the Ollama case (patient client, timeouts).
- **CURRENT_STATE_AND_DIRECTION.md** — current test-4 state, pipeline, permissions, and next steps.
- **REFOCUS_ON_ACTUAL_GOAL.md** — earlier “real goal” framed as “Enable GPU mode in Ollama”; that is the **first-stage** goal, not the full purpose above.

This document (**PHASE3_PURPOSE_AND_GOALS.md**) is the authoritative statement of **purpose and goals** for Phase 3: ultimate goal = all GPU software in the VM over the vGPU path; first stage = Ollama GPU inference success.
