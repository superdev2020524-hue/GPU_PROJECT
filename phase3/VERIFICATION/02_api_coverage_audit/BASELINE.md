# Baseline - Milestone 02 API Coverage Audit

## Scope

- Milestone: `02_api_coverage_audit`
- Folder: `phase3/VERIFICATION/02_api_coverage_audit/`
- Roadmap reference: `PHASE3_GENERAL_GPU_VIRTUALIZATION_ROADMAP.md`, Milestone 2
- Work type: audit / documentation plus bounded fail-closed hardening
- Owner role retained: Phase 3 assistant role, non-destruction rule, one-active-error discipline

## Preserved Baseline

Milestone 02 starts from the closed Milestone 01 baseline.

- Plan A state: pass
- Required preserved lane: Plan A
- Latest Plan A report:
  `/tmp/phase1_milestone_gate_after_m02_e3_live_binary_restart.json`
- Latest Milestone 01 gate:
  `/tmp/phase3_general_cuda_gate_after_m02_e3_live_binary_restart.json`
- Active Milestone 01 error: none
- Active Milestone 02 error: none

## Baseline Meaning

Milestone 02 is an audit milestone. It should not change runtime behavior unless
a later user-approved follow-up fix is explicitly split from the audit.

If any shared runtime file is changed during Milestone 02, stop and rerun Plan A
before making further conclusions.

This rule was applied for:

- `M02-E1`: Plan A passed before and after the deployed Runtime/cuBLASLt
  fail-closed change, and the Milestone 01 raw CUDA gate also passed afterward.
- `M02-E2`: Plan A passed before and after the deployed cuBLAS fail-closed
  change, and the Milestone 01 raw CUDA gate also passed afterward.
- `M02-E3`: Plan A passed before and after the deployed host executor
  protocol-disposition change, the live mediator executable path was verified,
  and the Milestone 01 raw CUDA gate also passed afterward.

## Known Candidates Carried Forward

- Residual non-terminating `0x00bc` / `cuFuncGetParamInfo` status noise.
- BAR1 fallback after shmem GPA resolution fails with `pfn_hidden`.
- `/usr/lib64/libcudart.so.12` resolves to the vGPU Runtime shim and is a
  deployment-scope risk for future non-Ollama applications.
- Ollama CUDA v12 runtime should eventually be replaced with a clean matching
  CUDA 12 artifact.
