# Goal, Role, and Guardrails

## 1. Real goal

Phase 3 is a general-purpose GPU remoting project, not an Ollama-only workaround.

Target architecture:

- guest app
- guest CUDA / cuBLAS shims
- VGPU-STUB
- host mediator
- host CUDA on physical H100
- results back to the VM

Phase 1 is only the first proof milestone:

- Ollama must run in GPU mode in the VM
- model load must proceed over the mediated GPU path
- inference must complete
- the VM must receive a real HTTP 200 response

Until one full generate succeeds, Phase 1 is not done.

## 2. Assistant role

The assistant is required to work in a way that preserves the real goal and avoids random drift.

Mandatory behavior:

- Search `phase3` history first before inventing a new explanation.
- Re-check previously working behavior after each fix.
- Use Checkpoints A-C before recommending a long load.
- Treat `SYSTEMATIC_ERROR_TRACKING_PLAN.md` as the operational procedure.
- Treat long-duration model loading as operator-approved work only.
- Pursue both active tracks together:
  - eliminate the current model-load / runner blocker
  - investigate and reduce transmission / model-load delay
- Always verify whether the live run is using `shmem` or `BAR1`; do not assume the fast path is active.
- Treat unacceptable load speed as a Phase 3 engineering issue, not a cosmetic issue to defer until after correctness.

## 3. Current authority

Current permissions are broader than some older notes imply.

VM:

- full authority
- commands, edits, deploy, build, install, restart, logs

Host / dom0:

- read logs and files
- edit Phase 3 sources and configs in agreed paths
- build and install mediator-side binaries
- restart mediator-related services
- all of this is allowed only under the non-destruction rule

Non-destruction rule:

- no destructive wipes
- no reckless changes to system trees
- if a change is unusually risky, stop and ask

## 4. What not to do

Do not do these unless there is a specific reason:

- do not re-read the entire `phase3` tree just to re-establish the basics
- do not run another blind 2-4 hour load first
- do not assume E1 is still the main live blocker without checking current logs
- do not describe the current situation as "pipeline still fundamentally broken"
- do not report "shared memory exists" as if that means the current run is fast
- do not treat 30+ minute model load as acceptable just because it eventually reaches HtoD progress

## 5. The shortest correct summary

As of this resume pack:

- the project is in late-stage narrowing, not early-stage uncertainty
- mediated GPU basics are working
- the current best next hypothesis is MMQ / graph-reserve failure in full model init
- the transmission path is still a real blocker because recent runs fell back to `BAR1` and remain serialized
- work must continue on both: targeted load-failure correction and transport/load-performance investigation

## 6. Source docs that remain authoritative

These are the main upstream docs behind this folder:

- `PHASE3_PURPOSE_AND_GOALS.md`
- `ASSISTANT_PERMISSIONS.md`
- `ASSISTANT_ROLE_AND_ANTICOUPLING.md`
- `SYSTEMATIC_ERROR_TRACKING_PLAN.md`
- `PHASE_NO_LOAD_FIRST_THEN_LONG_RUN.md`
- `ERROR_TRACKING_STATUS.md`
