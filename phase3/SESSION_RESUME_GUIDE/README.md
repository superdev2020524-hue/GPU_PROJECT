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
2. `07_PHASE2_PHASE3_METHOD_FREEZE.md`
3. `02_VERIFIED_STATE_2026-03-28.md`
4. `03_NEXT_ACTION_PLAN_MMQ_FIRST.md`
5. `04_IDEAS_FROM_EXISTING_GPU_REMOTING.md`
6. `05_OPERATIONAL_CHECKLIST_AND_COMMANDS.md`
7. `06_TRANSMISSION_AND_LOAD_PERFORMANCE_TRACK.md`

Current one-paragraph truth:

- Phase 3 is not Ollama-only. The real goal is general-purpose vGPU remoting: guest shims -> VGPU-STUB -> host mediator -> physical H100 -> results back.
- Phase 1 is now closed under the current approved three-lane definition: `Plan A` is the preserved checked-in canary baseline, revised `Plan B` is the approved Tiny-model gate, and `Plan C` is the client-facing normal-usage lane.
- `Plan A` currently passes on the repaired GPU-backed path with `qwen2.5:0.5b`, and that canary must remain the first regression detector for all later work.
- The revised approved `Plan B` Tiny gate also passes on the repaired path, and `Plan C` now passes on the same service with dedicated client model `qwen2.5:3b`, so Phase 1 closure is explicit rather than inferred.
- The correct continuation method is: prove GPU mode, prove `Plan A`, choose exactly one downstream lane (`Plan B` or `Plan C`) for active work, make one bounded change, write closure evidence, then force-clean `/api/ps` and re-check the preserved lanes serially.
- Therefore future Phase 2 and final Phase 3 work must preserve the current method and artifact set instead of inventing a new process or silently redefining milestone closure.

Do not lose these points:

- Do not treat Phase 3 as Ollama-only.
- Do not treat old E1-heavy notes as the whole current story.
- Do not stop running the validated Phase 1 canary after risky runtime or transport changes.
- Do not start long blind loads before quick checkpoints.
- Do not assume GPU mode without current proof.
- Do not treat a passing alternate model as automatic closure of a failing target model.
- Do not say "Phase 1 is fully closed" unless the preserved canary, the Tiny target lane, and the client-facing lane are all explicitly accounted for.
- Do not abandon one-active-error discipline just because the next milestones are broader.
- Do not run `Plan A`, `Plan B`, and `Plan C` gates concurrently; force-clean residency and run them serially.
- Do not forget that the current permissions allow host work too, but only under the non-destruction rule.

If session memory is lost, start from this folder, not from the whole `phase3` tree.
