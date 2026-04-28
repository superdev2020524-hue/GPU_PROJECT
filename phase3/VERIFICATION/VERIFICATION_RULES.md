# Phase 3 Verification Rules

## Purpose

This document defines how Phase 3 work proceeds after the Ollama milestone.
It preserves the operating role, error discipline, and evidence rules that were
used to reach the Stage 1 Ollama baseline, then extends them to every roadmap
milestone.

The goal is speed through discipline: small gates, current evidence, one active
error, and no broad implementation work without a known baseline.

## Non-Negotiable Role

The assistant must continue in the Phase 3 role already assigned by:

- `../ASSISTANT_PERMISSIONS.md`
- `../ASSISTANT_ROLE_AND_ANTICOUPLING.md`
- `../SYSTEMATIC_ERROR_TRACKING_PLAN.md`
- `../ERROR_TRACKING_STATUS.md`
- `../PHASE3_GENERAL_GPU_VIRTUALIZATION_ROADMAP.md`

This means:

- use the granted host and VM authority only inside the Phase 3 scope and under
  the non-destruction condition;
- search Phase 3 history before inventing a new fix;
- preserve working behavior before and after any risky change;
- keep exactly one active error per milestone lane;
- record new failures as candidates first;
- promote a candidate only after the active error is closed, disproved, or
  superseded by evidence;
- avoid long blind runs;
- correlate host and VM evidence from the same session;
- record material changes and closures in the relevant milestone folder and, if
  the global queue changes, in `../ERROR_TRACKING_STATUS.md`.

## Registry Structure

Each roadmap milestone has a folder under `phase3/VERIFICATION/`:

- `00_preserve_ollama_baseline/`
- `01_general_cuda_gate/`
- `02_api_coverage_audit/`
- `03_memory_sync_cleanup/`
- `04_pytorch_gate/`
- `05_second_framework_gate/`
- `06_multiprocess_multivm/`
- `07_security_isolation/`
- `08_server2_migration/`
- `09_twa_research/`

Each folder must contain milestone-local records. Historical Phase 3 documents
can be referenced, but the milestone folder is the current working register for
that unit.

## Required Files Per Milestone

Before implementation begins, create or update these records in the milestone
folder:

- `README.md` - scope, acceptance criteria, and current status.
- `BASELINE.md` - preserved baseline and live artifact proof.
- `GATE.md` - exact bounded test or audit gate.
- `ACTIVE_ERROR.md` - one active error and candidate queue.
- `EVIDENCE.md` - host/VM evidence collected in the current session.
- `DECISIONS.md` - decisions, rejected paths, and why.

If a milestone is documentation-only, such as the API audit or TWA research
track, the same records still apply, but the gate can be an audit gate rather
than a runtime test.

## Milestone Start Checklist

Every milestone starts with this sequence:

1. Read the milestone folder.
2. Read the current roadmap section.
3. Read the relevant Phase 3 rules and error registry.
4. Prove the preserved baseline required by that milestone.
5. Confirm the live host/VM artifact paths before interpreting runtime behavior.
6. Define the smallest bounded gate for the milestone.
7. Declare the lane, current Plan A state, active error, candidates, last proven
   checkpoint, and closure condition.

No code change should happen before this checklist is complete unless the user
explicitly asks for a documentation-only change.

## Baseline Rules

Milestone 0 is always the recovery baseline:

- `Plan A`: `qwen2.5:0.5b` checked-in canary.
- `Plan B`: approved `tinyllama:latest` gate when required.
- `Plan C`: client-facing `qwen2.5:3b` lane when required.

Before risky work, prove at least `Plan A` is green. If the milestone touches
runtime, guest shim, host mediator, CUDA executor, transport, service config, or
model/runtime libraries, also define which downstream lane must be rechecked
afterward.

If `Plan A` regresses, stop milestone work. The regression becomes the active
error until repaired.

## Evidence Rules

Every substantive milestone update must include:

- lane or milestone name;
- current Plan A state: `pass`, `fail`, or `unverified`;
- active error;
- candidate list;
- last proven checkpoint;
- live artifact proof;
- exact bounded repro or audit command;
- host evidence;
- VM evidence, if runtime-related;
- why the active error remains open or why it was closed;
- the next single step.

Do not mix historical evidence with current conclusions unless the historical
source is explicitly labeled as historical and a fresh current check has been
run.

## Error Handling Rules

When a failure appears:

1. Do not change code first.
2. Capture the exact signature: call id, return code, error name, sequence,
   timestamp, process, host line, and VM line where applicable.
3. Search Phase 3 history and the current milestone folder.
4. Classify the failure as active or candidate.
5. If it is only a candidate, keep the current active error unchanged.
6. Make one bounded change only after the active error has a closure condition.
7. Re-run the smallest gate that can prove or disprove the hypothesis.
8. Re-run the required baseline if the change touched shared runtime behavior.

## Anti-Coupling Rules

Do not let a milestone-specific fix silently become general architecture.

Classify each behavior as:

- general vGPU behavior;
- workload adapter behavior;
- temporary/debug workaround;
- unsupported or intentionally rejected behavior.

Any workaround that remains after a milestone must be recorded in
`DECISIONS.md` with a removal condition or promotion condition.

## Long-Run Rule

Long model loads, multi-hour runs, or broad stress tests require explicit user
approval unless they are already defined as the current milestone gate.

Cheap checks must run first: service health, live artifact proof, relevant host
log signatures, VM journal evidence, and the milestone's short gate.

## Closure Rules

A milestone is not closed by "it worked once."

Closure requires:

- gate pass or audit completion;
- no unresolved active error for that milestone;
- candidates recorded with disposition;
- evidence stored in the milestone folder;
- serial preservation recheck completed as required;
- `ERROR_TRACKING_STATUS.md` updated if the global queue changed;
- next milestone entry condition stated.

For milestone `N`, the closure report must separate:

- prior milestone preservation evidence (`00` through `N-1`);
- current milestone gate evidence (`N`);
- carried-forward candidates and why each is not blocking `N`;
- exact live artifact proof after the final change.

If a change touches shared runtime behavior, guest shims, transport, host
mediator, CUDA executor, QEMU stub, service config, or model/runtime libraries,
do not rely only on historical prior-stage results. Re-run the current preserved
baseline and the most relevant prior runtime gate after the final change. For
the current Phase 3 sequence this means at minimum:

- `00`: Plan A canary after the final change;
- `01`: raw CUDA gate after the final change;
- `02`: API audit records/source consistency reviewed after the final change,
  and any newly discovered fake-success or unclassified behavior either closed
  or explicitly carried forward with impact analysis;
- current milestone: its own bounded gate after the final change.

Do not close a milestone by saying "this belongs to the next milestone" unless
the relationship has been investigated and recorded. The record must explain why
the issue is not a regression of any previous milestone and why it is outside
the current milestone's acceptance criteria.

## Immediate Next Milestone

The next active milestone is:

`01_general_cuda_gate`

It must begin by preserving the current Ollama baseline, then defining a raw
CUDA gate below PyTorch/TensorFlow level.
