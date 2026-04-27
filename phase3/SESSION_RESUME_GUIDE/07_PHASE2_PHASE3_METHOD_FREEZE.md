# Phase 2 / Phase 3 Method Freeze

This document is the continuity contract for all work after the `Plan A` canary closure and while `Plan B` or later Phase 3 goals may still remain open.

Its purpose is simple:

- keep the current method intact,
- prevent regression to ad-hoc debugging,
- preserve the exact evidence discipline that worked,
- and make sure Phase 2 and the final Phase 3 objective are completed using the same operating model.

This document is not Phase 1-only. It is the rulebook for how to continue from here without losing `Plan A` while pushing `Plan B` or later milestones.

---

## 1. Current starting point

As of the current validated baseline:

1. The repaired GPU/vGPU-backed path is working.
2. The checked-in default `Plan A` canary gate passes.
3. The passing canary baseline uses `qwen2.5:0.5b`.
4. The current canary suite lives in `phase3/phase1_milestone_test_suite.json`.
5. The current canary runner lives in `phase3/phase1_milestone_gate.py`.
6. The canary was re-verified after a clean `ollama` restart.
7. `Plan B` remains open if `tinyllama:latest` is still a required Phase 1 target.

This matters because all later work must preserve this as the known-good canary path unless there is explicit proof that the active error requires changing it. A passing canary is not, by itself, permission to erase an open `Plan B` target.

---

## 2. Non-negotiable method freeze

The following rules are now frozen and apply to Phase 2 and Phase 3 work unless the user explicitly overrides them for one bounded action.

### 2.1 One active error only

- Keep exactly one active error.
- Record all other observations as candidates.
- Do not replace the active error just because a new symptom appears.
- Promote a candidate only when the active error is closed, disproved, or clearly superseded by earlier proof in the same causal chain.

### 2.2 Gate first, integrate second

Before deeper experiments:

1. verify service health,
2. verify GPU mode,
3. verify the current canary gate,
4. only then start larger milestone work.

### 2.3 One hypothesis, one minimal change, one bounded verification

- No multi-layer patch storms.
- No host + guest + Ollama + config changes in the same cycle unless the user explicitly requests a coordinated deployment.
- Each cycle must have a declared pass/fail condition before it starts.

### 2.4 Write closure evidence every time

For every substantive cycle, record:

- lane under discussion (`Plan A` or `Plan B`),
- current `Plan A` state (`pass`, `fail`, or `unverified`),
- active error,
- candidate list,
- closure condition,
- last proven checkpoint,
- live artifact proof,
- bounded repro command / timeout,
- evidence,
- why the active error remains active or why it is closed,
- next single step.

### 2.5 Do not discard proven references

Do not replace working documents with vague summaries.

The written materials are not optional notes. They are part of the implementation method.

---

## 3. The fixed artifact set that must be preserved

These are the reference materials that must remain available and must continue to be treated as canonical.

### 3.1 Rule and contract documents

- `phase3/PHASE1_FAST_TRACK_DIRECTION_AND_STRICT_PRINCIPLES.md`
- `phase3/STAGE1_NO_REGRESSION_FAST_PATH.md`
- `phase3/SYSTEMATIC_ERROR_TRACKING_PLAN.md`

These define the discipline and anti-regression method.

### 3.2 Live status and evidence documents

- `phase3/PHASE1_FAST_TRACK_STATUS.md`
- `phase3/ERROR_TRACKING_STATUS.md`
- `phase3/HOST_VM_CHANGE_LOG.md`

These preserve the actual state history and change evidence.

### 3.3 Gate and baseline documents

- `phase3/phase1_milestone_gate.py`
- `phase3/phase1_milestone_test_suite.json`

These are the executable canary.

### 3.4 Explanation and onboarding documents

- `phase3/PHASE1_IMPLEMENTATION_MANUAL.md`
- `phase3/SESSION_RESUME_GUIDE/README.md`
- `phase3/SESSION_RESUME_GUIDE/01_GOAL_ROLE_AND_GUARDRAILS.md`
- `phase3/SESSION_RESUME_GUIDE/07_PHASE2_PHASE3_METHOD_FREEZE.md`

These explain the process to future engineers or future sessions.

### 3.5 GPU-mode reference documents

- `phase3/GPU_MODE_DO_NOT_BREAK.md`
- the GPU verification section in `phase3/PHASE1_IMPLEMENTATION_MANUAL.md`

These exist specifically to prevent false success caused by accidental CPU fallback.

---

## 4. Mandatory preflight for every future Phase 2 / Phase 3 work session

Before any serious milestone work, perform these checks.

### 4.1 Confirm service health

Inside the VM:

```bash
systemctl is-active ollama
curl -s http://127.0.0.1:11434/api/tags
```

### 4.2 Confirm GPU mode

Run the GPU verification sequence from `PHASE1_IMPLEMENTATION_MANUAL.md`:

1. service configuration proof,
2. runner library proof,
3. live transport-traffic proof.

If GPU mode is not proven, Phase 2 / Phase 3 milestone work must not begin.

### 4.3 Confirm the canary gate

From the host workspace:

```bash
cd /home/david/Downloads/gpu/phase3
python3 phase1_milestone_gate.py \
  --base-url http://10.25.33.110:11434 \
  --timeout-sec 240 \
  --output /tmp/phase1_milestone_gate_report.json
```

If the canary gate fails:

- do not start new milestone exploration,
- first classify whether the failure is a regression or an environment drift issue.

### 4.3b Confirm whether the session is preserving `Plan A` or advancing `Plan B`

Before deeper work, explicitly record:

1. whether the session goal is `Plan A` preservation or `Plan B` advancement,
2. whether `tinyllama:latest` is still a required Phase 1 target for this session,
3. what exact `Plan B` gate or checkpoint will be used,
4. and what result would count only as canary health rather than `Plan B` closure.

### 4.4 Confirm the artifact baseline if code changed

If any deployment-facing component changed, also verify:

- live service config,
- deployed guest shim paths,
- host mediator/stub state,
- relevant logs are clean enough for a fresh comparison.

---

## 5. Required execution loop for Phase 2 and Phase 3

Use this loop exactly.

### Step 1: define the milestone gate

For each future milestone, write down:

1. what success means,
2. what quick gate proves it,
3. what canary must continue to pass,
4. what must not regress.

For Phase 1 specifically, also write down whether the work is:

1. `Plan A` preservation,
2. `Plan B` advancement,
3. or broader Phase 3 work that is only allowed because `Plan A` is still green.

Do not start with "investigate generally."

### Step 2: define the active error

State:

- active error,
- candidate list,
- closure condition.

This must be written before the main experiment.

### Step 3: run one bounded experiment

Each experiment must:

- test only one main hypothesis,
- be time-bounded,
- produce interpretable host/VM evidence,
- and have a known comparison point.

### Step 4: update the written record

After the experiment:

- update the status file,
- record what changed,
- record why the active error stayed active or closed,
- identify the next single step.

### Step 5: rerun the canary after risky changes

If you changed anything that could affect:

- transport,
- runtime configuration,
- deployed shims,
- the host replay path,
- model-serving behavior,

rerun the default Phase 1 canary gate.

This is how we make forward progress without losing the already-proven path.

---

## 6. How future milestone documentation should be structured

For Phase 2 and Phase 3 work, do not invent a new documentation style. Reuse the same package shape.

For each future milestone, create or maintain:

1. a direction-and-rules document,
2. a status/evidence log,
3. a gate runner or command bundle,
4. a fixed suite or checklist,
5. an implementation/explanation manual if the work becomes large enough.

Recommended naming pattern:

- `PHASE2_*`
- `PHASE3_*`

Examples:

- `PHASE2_MILESTONE_DIRECTION_AND_STRICT_PRINCIPLES.md`
- `PHASE2_MILESTONE_STATUS.md`
- `phase2_milestone_test_suite.json`
- `phase2_milestone_gate.py`

The important point is not the file names. The important point is to keep the same artifact roles.

---

## 7. The Phase 1 gate is now a permanent canary

This is one of the most important continuity rules.

Even when the main work moves to `Plan B`, Phase 2, or the final Phase 3 goal, the validated `Plan A` gate must remain the first regression detector.

Why:

- it is fast,
- it is already proven,
- it exercises the repaired GPU/vGPU-backed serving path,
- and it catches environment drift before deeper milestone work wastes time.

Therefore:

- do not delete it,
- do not casually change it,
- do not stop running it after runtime-affecting changes.

---

## 8. What counts as acceptable future change

A change is acceptable if:

1. it advances the current active error,
2. it is bounded and explainable,
3. it does not silently invalidate the canary baseline,
4. and its effect is recorded in the written evidence trail.

A change is not acceptable if:

- it mixes multiple causal layers without need,
- it changes the baseline without recording why,
- it removes evidence or reference material,
- or it makes later sessions unable to reconstruct what happened.

---

## 9. Template for every future Phase 2 / Phase 3 status update

Use this exact structure.

### Active error

State the single active error in one sentence.

### Candidates

List the candidate branches and explicitly say they are candidates.

### Closure condition

State what evidence would close the active error.

### Evidence

Include:

- one VM evidence line,
- one host evidence line,
- and the gate impact.

### Why the active error remains active or was closed

Say why, explicitly.

### Next single step

Only one next step.

This template is how we prevent drift.

---

## 10. Recommended reading order for future sessions

If a future engineer or assistant must resume work after context loss, read in this order:

1. `phase3/SESSION_RESUME_GUIDE/01_GOAL_ROLE_AND_GUARDRAILS.md`
2. `phase3/SESSION_RESUME_GUIDE/07_PHASE2_PHASE3_METHOD_FREEZE.md`
3. `phase3/PHASE1_FAST_TRACK_DIRECTION_AND_STRICT_PRINCIPLES.md`
4. `phase3/PHASE1_FAST_TRACK_STATUS.md`
5. `phase3/phase1_milestone_test_suite.json`
6. `phase3/phase1_milestone_gate.py`
7. `phase3/PHASE1_IMPLEMENTATION_MANUAL.md`

Only after that should the engineer branch into older historical notes.

---

## 11. Final instruction

From this point onward, the correct mindset is:

- preserve the canary,
- preserve the queue discipline,
- preserve the written evidence,
- and scale the same method into Phase 2 and the final Phase 3 objective.

Do not try to become faster by dropping the method.

The method is what made the system finally converge.
