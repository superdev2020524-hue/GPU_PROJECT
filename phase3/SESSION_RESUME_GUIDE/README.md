# Phase 3 Session Resume Guide

This folder is a compact handoff pack for future sessions.

Goal: a future assistant should be able to read only this folder and immediately understand:

- what Phase 3 is trying to achieve
- what Phase 1 means
- what authority and role apply
- what has already been verified live
- what the next experiments should be
- what to avoid

Read in this order:

1. `01_GOAL_ROLE_AND_GUARDRAILS.md`
2. `02_VERIFIED_STATE_2026-03-28.md`
3. `03_NEXT_ACTION_PLAN_MMQ_FIRST.md`
4. `04_IDEAS_FROM_EXISTING_GPU_REMOTING.md`
5. `05_OPERATIONAL_CHECKLIST_AND_COMMANDS.md`
6. `06_TRANSMISSION_AND_LOAD_PERFORMANCE_TRACK.md`

Current one-paragraph truth:

- Phase 3 is not Ollama-only. The real goal is general-purpose vGPU remoting: guest shims -> VGPU-STUB -> host mediator -> physical H100 -> results back.
- Phase 1 is the proof milestone: one complete Ollama GPU-mode generate with a real HTTP 200 response.
- As of 2026-03-28, the pipeline is much further along than it may feel: Ollama is in GPU mode, `compute=9.0` is live, mediated `cublasGemmEx` preflight passes, and the current live blocker is no longer the older E1 invalid-fatbin story. The strongest current blocker is the later MMQ / graph-reserve failure (`mmq_x_best=0`, `mmq.cuh:3884`, runner abort after load reaches `1.00`).
- The weight-transfer problem is also real and active: recent long runs show the guest falling back to `BAR1` instead of shared memory, and the transport path remains serialized enough that model load is far slower than an acceptable customer path.
- Therefore the next move is not another blind long run. Work must continue on **two tracks at once**: fix the active model-load blocker and investigate / reduce transmission delay.

Do not lose these points:

- Do not treat Phase 3 as Ollama-only.
- Do not treat old E1-heavy notes as the whole current story.
- Do not start long blind loads before quick checkpoints.
- Do not treat load-performance as a secondary issue after correctness.
- Do not assume shared memory is active without live proof; recent runs show `BAR1` fallback.
- Do not forget that the current permissions allow host work too, but only under the non-destruction rule.

If session memory is lost, start from this folder, not from the whole `phase3` tree.
