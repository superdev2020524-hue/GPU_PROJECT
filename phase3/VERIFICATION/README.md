# Phase 3 Verification Registry

This registry controls the next Phase 3 work after the Ollama milestone.

The purpose is to keep the roadmap executable, traceable, and safe. Each
milestone has its own folder. Work must move through those folders in order,
with explicit baselines, evidence, active-error tracking, and closure criteria.

## Authoritative Documents

- `VERIFICATION_RULES.md` - binding procedure for milestone work.
- `MILESTONE_INDEX.md` - roadmap-to-folder mapping.
- `templates/` - reusable records for each milestone.
- `../PHASE3_GENERAL_GPU_VIRTUALIZATION_ROADMAP.md` - internal roadmap.
- `../SYSTEMATIC_ERROR_TRACKING_PLAN.md` - active-error and evidence discipline.
- `../ASSISTANT_ROLE_AND_ANTICOUPLING.md` - assistant role and anti-coupling duties.
- `../ASSISTANT_PERMISSIONS.md` - host and VM permissions.
- `../ERROR_TRACKING_STATUS.md` - rolling error registry and session history.

## Rule

Do not treat a milestone as started until its folder contains a current
baseline, scope, gate definition, evidence plan, and active-error record.

Do not treat a milestone as closed until its gate passes, the required host and
VM evidence is recorded, and the preserved Phase 1 baseline has been rechecked
where required.
