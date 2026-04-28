# 00 - Preserve Ollama Baseline

## Purpose

Preserve the current proven Phase 1 baseline before broader vGPU work.

This milestone is the recovery point for all later milestones. If later work
breaks the baseline, stop the downstream milestone and repair this one first.

## Preserved Lanes

- Plan A: `qwen2.5:0.5b`, checked-in canary gate.
- Plan B: `tinyllama:latest`, approved revised Tiny gate.
- Plan C: `qwen2.5:3b`, client-facing CLI lane.

## Required Method

1. Start or verify host mediator.
2. Start or verify VM `ollama`.
3. Confirm vGPU PCI device and GPU-mode service configuration.
4. Confirm `/api/ps` is clean before cross-model gates.
5. Run Plan A first.
6. Run Plan B or Plan C only when required by the current downstream work.
7. Force-clean residency after cross-model checks.
8. Record host and VM evidence.

## Current Baseline Reference

See `../ERROR_TRACKING_STATUS.md` session `2026-04-27` for the latest full
re-baseline before general vGPU work.

Latest post-Milestone-03 preservation follow-up:

- Plan A passed:
  `/tmp/phase1_milestone_gate_serial_00_after_planc_fix.json`.
- Plan B Tiny passed:
  `/tmp/phase1_plan_b_serial_00_after_planc_fix.json`.
- Plan C client lane passed after correcting the gate invocation to feed prompts
  through stdin:
  `/tmp/phase1_plan_c_serial_00_after_m03_fixed_clean.json`.
- Final `/api/ps` was clean: `{"models":[]}`.

Plan C note:

- The timeout seen after Milestone 03 was not classified as a runtime regression.
- `ollama run MODEL PROMPT_AS_ARG` timed out in the non-interactive SSH context,
  while `printf PROMPT | ollama run MODEL` worked for both `qwen2.5:0.5b` and
  `qwen2.5:3b`.
- The gate now uses stdin prompt delivery and strips terminal-control output
  before exact-answer comparison.

## Closure Criteria

- Plan A passes.
- Required downstream lane passes.
- Host mediator shows physical GPU execution.
- VM remains in GPU mode.
- `/api/ps` is clean after the run.
- No active preservation error remains.
