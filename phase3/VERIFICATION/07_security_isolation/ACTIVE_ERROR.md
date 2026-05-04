# Active Error - Milestone 07 Security And Isolation

## Current Lane

Milestone 07: Security And Isolation

## Current Plan A State

Pass after final M07 refresh and TensorFlow shim correction:

- `/tmp/m07_final_after_tf_3param_planA.json` -> `overall_pass=True`.

## Active Error

None. M07 gate surface is closed after the 2026-04-30 refresh, and required
preservation passed after the final guest-shim TensorFlow launch-layout fix.

## Closed Errors

- `M07-E1`: malformed request handling at the mediator socket/CUDA-header
  boundary was unproven. Closed after adding mediator pre-allocation payload
  bounds and CUDA-header validation, then proving malformed socket probes fail
  safely and both VM gates still pass.
- `M07-E2`: quarantine/rate-limit rejection behavior and guest BAR/MMIO
  permission assumptions were unproven. Closed after fixing DB-to-watchdog
  quarantine sync, proving targeted Test-6 quarantine/recovery, proving Test-6
  rate-limit rejection/recovery, and documenting current BAR `0666` exposure as
  an explicit engineering trust assumption and production-hardening candidate.
- `M07-E3`: final M07 preservation found TensorFlow `TensorAssignOp` +
  `scalar_const_op` three-parameter `EigenMetaKernel` still using a bad raw
  launch layout. The earlier `8+8+8` layout passed one launch but corrupted the
  scalar/count fields on a repeat launch and produced
  `CUDA_ERROR_ILLEGAL_ADDRESS`, followed by primary-context recovery and
  `CUDA_ERROR_CONTEXT_IS_DESTROYED`. Closed by changing the guest-shim layout to
  compact `8+4+4` bytes (`device pointer`, scalar float, element count) and
  rerunning TensorFlow plus final preservation.

## Candidate Queue

- Guest root may have more direct BAR/MMIO write capability than a production
  tenant policy should allow.
- Guest unprivileged users currently have write access to BAR0/BAR1
  (`resource0`/`resource1` mode `0666`) so the current baseline does not provide
  in-guest user isolation from the vGPU BAR interface.
- Strong IOMMU and hypervisor-level isolation assumptions are documented only at
  the roadmap level.

## Last Proven Checkpoint

Final M07 refresh on the live mediator:

- malformed socket probe: `/tmp/m07_final_after_tf_3param_malformed.json` ->
  `overall_pass=True`;
- Plan A: `/tmp/m07_final_after_tf_3param_planA.json` -> `overall_pass=True`;
- M01 raw CUDA: `/tmp/m07_final_after_tf_3param_m01.json` ->
  `overall_pass=True`;
- M03 async stream/event, M04 PyTorch, M05 CuPy, M05 TensorFlow, and M06
  two-process CuPy all passed on the corrected guest shim;
- mediator PID `1846882` remained alive and no primary-context recovery was
  logged after the final malformed probe.

## Closure Condition

M07 is closed for the defined gate. Future production-hardening candidates
remain, especially replacing guest BAR `0666` access with narrower policy.

## Preservation Notes

- Plan A had an immediate force-unload polling false negative. Delayed `/api/ps`
  showed the model absent, the gate was fixed to wait for absence, and Plan A
  then passed.
- Plan B had the same immediate force-unload polling false negative. The gate
  was fixed to wait for absence, and Plan B then passed.
- These were gate timing repairs, not vGPU/mediator regressions.
