# Baseline - Milestone 04 PyTorch Gate

## Entry Condition

Milestone 04 starts only after Milestone 03 memory/sync/cleanup closure and the
post-closure serial preservation recheck.

## Preserved Prior Milestones

- `00_preserve_ollama_baseline`
  - Plan A after Plan C gate fix:
    `/tmp/phase1_milestone_gate_serial_00_after_planc_fix.json` ->
    `overall_pass=True`.
  - Plan B after Plan C gate fix:
    `/tmp/phase1_plan_b_serial_00_after_planc_fix.json` ->
    `overall_pass=True`.
  - Plan C after gate invocation fix:
    `/tmp/phase1_plan_c_serial_00_after_m03_fixed_clean.json` ->
    `overall_pass=True`.
- `01_general_cuda_gate`
  - Serial raw CUDA after Milestone 03:
    `/tmp/phase3_general_cuda_gate_serial_01_after_m03.json` ->
    `overall_pass=True`, Driver API 5/5 and Runtime API 5/5.
- `02_api_coverage_audit`
  - Serial audit consistency after Milestone 03:
    `/tmp/phase3_api_audit_serial_02_after_m03.json` ->
    `overall_pass=True`.
- `03_memory_sync_cleanup`
  - 4 MiB async/mixed stream-event gate:
    `/tmp/async_stream_event_probe_repeat.json` -> `overall_pass=True`,
    `pass_count=5`, `runs=5`, `bytes_per_run=4194304`.

## Current Plan A State

Pass at Milestone 04 entry.

## Live Artifact Proof To Refresh

Before interpreting PyTorch behavior, capture:

- VM `ollama` state and `/api/ps`;
- VM vGPU PCI device;
- guest shim paths under `/opt/vgpu/lib`;
- host mediator PID and executable path;
- host `/tmp/mediator.log` health counters.
