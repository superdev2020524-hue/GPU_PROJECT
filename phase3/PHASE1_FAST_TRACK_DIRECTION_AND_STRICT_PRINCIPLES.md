# Phase 1 fast-track direction and strict principles

*Created: 2026-04-05*

This document is the execution contract for finishing the Phase 1 milestone quickly, without repeating already-verified work.

---

## Final direction (what must be true to close Phase 1)

Phase 1 is closed only when all three outcomes are simultaneously true on the active baseline:

1. **Response accuracy:** bounded deterministic prompts return correct and parseable outputs.
2. **Response speed:** bounded runs meet cold and warm latency targets with repeatable evidence.
3. **Weight residency behavior:** loaded model state remains resident when requested (and unloads when requested), consistent with normal Ollama GPU behavior.

If any one of these fails, Phase 1 remains open.

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
- pass pre-integration test suite (accuracy micro-cases + speed sanity + residency checks),
- only then run deeper integrated verification.

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

### 5) Timebox and escalation

- each candidate gets a short timebox and a closure/disproof criterion before work starts,
- if criterion is not met in timebox, mark result and advance by queue rules.

---

## Phase 1 acceptance criteria (fast-track)

All criteria must pass on the same baseline window:

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

---

## Immediate execution order (start now)

1. Freeze baseline settings that are currently known-safe.
2. Run and record pre-integration suite from `phase1_milestone_test_suite.json`.
3. Promote one active error based on earliest failed gate among accuracy/speed/residency.
4. Execute one minimal fix cycle for that active error only.
5. Re-run full gate suite and update queue state.

---

## Initial queue state for this fast-track

- **Active error:** `P1-A` (milestone acceptance not yet proven in one converged window).
- **Candidates:** `P1-B` speed jitter inflation, `P1-C` residency inconsistency, residual non-terminating probe noise.
- **Closure condition for `P1-A`:** all three gate families (accuracy, speed, residency) pass in one bounded evidence bundle.

This queue state remains until first gated bundle is completed and evaluated.
