# Baseline - Milestone 07 Security And Isolation

Milestone 07 starts after expanded Milestone 06 closure.

## Preserved Entry State

- Phase 1 Plan A:
  `/tmp/m06_final_phase1_planA.json` -> `overall_pass=True`.
- Phase 1 Plan B:
  `/tmp/m06_final_phase1_planB.json` -> `overall_pass=True`.
- Phase 1 Plan C:
  Test-10 `/tmp/m06_final_phase1_planC.json` -> `overall_pass=True`.
- Milestone 01:
  `/tmp/m06_final_m01_general_cuda_gate.json` -> `overall_pass=True`,
  Driver API 5/5 and Runtime API 5/5.
- Milestone 02:
  `/tmp/m06_final_m02_api_audit.json` -> `overall_pass=True`,
  `protocol_ids_excluding_sentinel=87`, no missing executor mentions, and no
  missing required matrix sections.
- Milestone 03:
  `/tmp/m06_final_m03_async_preservation/summary.json` ->
  `overall_pass=True`, three 4 MiB async stream/event runs passed.
- Milestone 04:
  Test-10 PyTorch preservation 3/3 pass in
  `/tmp/m06_final_framework_preservation/summary.json`.
- Milestone 05:
  Test-10 CuPy preservation 3/3 pass in
  `/tmp/m06_final_framework_preservation/summary.json`.
- Milestone 06:
  Test-6 clean CuPy tail
  `/tmp/m06_final_test6_cupy_clean/summary.json` -> `overall_pass=True`,
  3/3 pass.

## Live M06 Baseline

- One host mediator serves Test-10 and Test-6.
- Test-10: `vm_id=10`, Pool A.
- Test-6: `vm_id=6`, Pool B, `priority=low`.
- Final mediator health after M06:
  `sync FAILED=0`, `CUDA_ERROR_ILLEGAL_ADDRESS=0`,
  `Unsupported CUDA protocol call=0`, `invalid handle=0`.
- Final mediator stats after M06:
  `Total processed=7429`, `Pool A processed=5059`, `Pool B processed=2370`,
  `WFQ queue depth=0`, `CUDA busy=no`.

## M07 Entry Rule

Before any risky M07 probe or code change, preserve the M06 baseline. If a
malformed-input probe regresses Plan A or the framework gates, stop and repair
the regression before drawing deeper security conclusions.
