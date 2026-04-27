# Phase 1 fast-track direction and strict principles

*Created: 2026-04-05*

This document is the execution contract for finishing the Phase 1 milestone quickly, without repeating already-verified work.

---

## Final direction (three-lane milestone structure)

Phase 1 now has three explicit lanes that must not be conflated:

1. **Plan A: preserved canary baseline**
   - model: `qwen2.5:0.5b`
   - gate: checked-in default `phase1_milestone_gate.py` + `phase1_milestone_test_suite.json`
   - purpose: prove the repaired GPU-backed serving path still works and catch regressions quickly
2. **Plan B: explicit follow-on milestone target**
   - model: `tinyllama:latest` unless the user redefines the target
   - purpose: prove the same repaired path also closes the intended Tiny-model milestone rather than only the canary
3. **Plan C: client-facing normal-usage lane**
   - model: `qwen2.5:3b`
   - gate: `phase3/PHASE1_PLAN_C_CLIENT_GATE.md` + `phase3/phase1_plan_c_client_gate.py`
   - purpose: prove standard one-shot user prompts behave correctly on the same repaired GPU-backed service without mutating `Plan A` or `Plan B`

`Plan A` passing is mandatory and must be preserved, but it is not by itself permission to say Phase 1 is fully complete if `Plan B` or `Plan C` remain explicit milestone targets.

Phase 1 is fully closed only when one of the following is true:

1. `Plan A` passes, `Plan B` passes its agreed gate, and `Plan C` passes its agreed client-facing gate on the repaired path.
2. The user explicitly redefines Phase 1 closure so that `Plan B` or `Plan C` is demoted or removed as a required milestone target.

---

## One-active-error discipline (binding)

Use strict queue control for fast execution:

- Keep exactly one **active error**.
- Record other observations as **candidates** with evidence.
- Promote a candidate only when the active error is resolved, disproved, or superseded by proof in the same causal chain.
- Every substantive update must include:
  - active error,
  - candidate list,
  - closure condition,
  - one host and one VM evidence line (or explicit unreachable reason).

This preserves speed by avoiding context thrash.

---

## Non-redundancy rules (do not waste cycles)

The following methods are **not** to be re-run unless fresh checkpoint evidence re-activates them:

1. Historical blockers already closed in current PHASE3 path (for example: earlier zero-payload transport class, earlier startup deadlock class) are not primary by default.
2. Known rejected toggles are not to be re-tested without new causal evidence:
   - graph re-enable path that regressed to hard failure,
   - batched-CUBLAS setting that caused severe cold-start regression.
3. Do not restart broad E1/E6/E7-era tracing unless current Checkpoint C or fresh run proves those signatures are earliest again.
4. No long blind load/generate runs before passing short gates and pre-integration suite.

---

## Strict principles for rapid closure

### 1) Gate first, integrate second

Before every integration run:

- pass short health gates (service, discovery, host correlation),
- pass the `Plan A` pre-integration suite (accuracy micro-cases + speed sanity + residency checks),
- only then run deeper `Plan B` or broader integrated verification.

### 2) Pre-reviewed test content is mandatory

Do not use ad-hoc prompts during integration debugging. Use a fixed, versioned suite:

- deterministic exact-string case,
- deterministic arithmetic case,
- deterministic structured JSON case,
- bounded latency probes (cold and warm),
- keep_alive residency checks.

### 3) Code hygiene is part of debugging

For touched components (guest shim, stub, mediator, runner-adjacent code):

- remove obsolete debug branches and redundant dead code before integration test,
- remove stale warning-producing fragments in touched code paths,
- keep logs focused on causal evidence (call_id/seq/rc/status/timestamp), not noise.

Unplanned warnings and redundant branches are treated as tracking risk, not cosmetic debt.

### 4) Smallest-possible-change policy

- one hypothesis -> one minimal patch -> one bounded verification,
- if no causal movement, revert the hypothesis path and move to next candidate,
- avoid multi-variable edits that hide cause/effect.

### 4a) Serial gate isolation

- do not run `Plan A`, `Plan B`, and `Plan C` gates concurrently,
- after any `Plan C` proof or any cross-model experiment, unload resident models and confirm `/api/ps` is empty before preservation re-checks,
- when preserving the milestone baseline after `Plan C` work, re-run `Plan A` first and `Plan B` second.

### 5) Timebox and escalation

- each candidate gets a short timebox and a closure/disproof criterion before work starts,
- if criterion is not met in timebox, mark result and advance by queue rules.

### 6) Compliance additions required for the final goal

Every substantive Phase 1 update must also record:

1. **Lane under discussion:** `Plan A` or `Plan B`.
2. **Current `Plan A` state:** `pass`, `fail`, or `unverified`.
3. **Live artifact identity:** deployed host/guest/runtime path actually under test.
4. **Last proven checkpoint:** exact ladder position before the current conclusion.
5. **Bounded repro definition:** exact command, timeout, model, and trace set.
6. **Regression verdict:** whether the current result is earlier than the last proven checkpoint.
7. **Milestone-scope verdict:** whether the result proves transport health only, `Plan A` only, or true `Plan B` advancement.

Without all seven items, the update is incomplete.

### 7) Model-switching discipline

- A passing alternate model may prove that the repaired path is healthy, but it does **not** automatically close a failing target model.
- Do not silently convert a passing alternate model into milestone closure.
- If the active branch changes model or suite scope, explicitly record whether that change is:
  - a preserved canary (`Plan A`),
  - a target milestone (`Plan B`),
  - or a diagnostic side branch only.
- If `Plan A` regresses during `Plan B` work, stop and restore `Plan A` before interpreting `Plan B`.

---

## Phase 1 acceptance criteria (fast-track)

### Plan A acceptance

All of the following must pass on the preserved canary baseline:

1. **Accuracy gate**
   - deterministic suite pass rate: 100% on required cases.
2. **Speed gate**
   - cold p95 within current target budget,
   - warm bounded latency within target budget,
   - no outlier explained only by avoidable logging/config noise.
3. **Residency gate**
   - with `keep_alive=-1`, model remains visible in `/api/ps` and can serve warm request without reload behavior,
   - with `keep_alive=0`, unload behavior is confirmed.
4. **Stability guard**
   - no terminating runner abort in gate window.

### Plan B acceptance

`Plan B` must be evaluated on a `Plan A`-passing baseline and must pass the approved Tiny gate:

- contract: `phase3/PHASE1_PLAN_B_TINY_GATE.md`
- runner: `phase3/phase1_plan_b_tiny_gate.py`

Approved binding `Plan B` cases:

1. `B1_cold_residency_pin`
2. `B2_warm_arithmetic_strict`
3. `B3_warm_json_strict`
4. `B4_force_unload`

Current interpretation of that gate:

- strict JSON-only output is now achievable with written request-side stop controls,
- arithmetic proof is now bound to the structured arithmetic JSON form in the approved `B2` case,
- exact-token behavior is no longer the binding closure criterion unless the user explicitly restores it.

Until the approved Tiny gate passes, Phase 1 remains open if `Plan B` remains a required target.

### Plan C acceptance

`Plan C` must be evaluated on the already-proven service path with a dedicated client model and must pass the approved client gate:

- contract: `phase3/PHASE1_PLAN_C_CLIENT_GATE.md`
- runner: `phase3/phase1_plan_c_client_gate.py`

Approved binding `Plan C` cases:

1. `C1_small_arithmetic_cli_style`
2. `C2_large_arithmetic_cli_style`
3. `C3_second_large_arithmetic_cli_style`
4. `C4_reference_arithmetic_cli_style`
5. `C5_force_unload`

Current interpretation of that gate:

- `Plan C` exists to close normal one-shot user-facing arithmetic on the live service,
- it is intentionally a separate model lane so `Plan A` and `Plan B` stay unchanged,
- preservation proof after `Plan C` requires a forced-clean handoff and serial re-check of `Plan A` then `Plan B`.

---

## Immediate execution order (start now)

1. Freeze baseline settings that are currently known-safe.
2. Run and record the `Plan A` pre-integration suite from `phase1_milestone_test_suite.json`.
3. If `Plan A` fails, repair `Plan A` before any deeper work.
4. If `Plan A` passes, promote one `Plan B` or `Plan C` active error based on the explicitly chosen lane and earliest failed checkpoint or gate family.
5. Execute one minimal fix cycle for that active error only.
6. After risky changes, force-clean resident models and re-run the full `Plan A` gate before interpreting downstream lane results.
7. Re-run the selected downstream lane gate serially and update queue state.

---

## Current queue reset for this fast-track

- **Plan A active error:** none while the preserved canary remains green.
- **Plan B active error:** none while the approved revised Tiny gate remains green.
- **Plan C active error:** none while the approved client-facing `qwen2.5:3b` gate remains green.
- **Plan B candidates:** exact-token Tiny behavior is non-gating unless explicitly restored; residual `0x00bc` and any earlier checkpoint regression remain candidate-only unless they become earliest again on a fresh bounded repro.
- **Plan C candidates:** prompt-wrapper-only fixes are candidate-only unless they close the plain user-facing arithmetic path without mutating `Plan A` / `Plan B`.
- **Closure condition for the next downstream active error:** if the revised Tiny gate or the client-facing gate regresses, identify the earliest failing binding case or earlier checkpoint on the preserved baseline and record the required queue fields.

This queue definition replaces the older single-lane `P1-A` starting state and reflects the current passing `Plan A` + revised `Plan B` + client-facing `Plan C` closure baseline.
