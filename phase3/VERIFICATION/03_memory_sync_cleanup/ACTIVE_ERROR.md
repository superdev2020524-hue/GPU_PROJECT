# Active Error - Milestone 03 Memory, Synchronization, And Cleanup

## Current Lane

Milestone 03: Memory, Synchronization, And Cleanup

## Current Plan A State

Pass.

Evidence:

- Before Milestone 03 work:
  `/tmp/phase1_milestone_gate_before_m03.json` -> `overall_pass=True`.
- Final after Milestone 03:
  `/tmp/phase1_milestone_gate_m03_final_after_chunking.json` ->
  `overall_pass=True`.

## Active Error

None. Milestone 03 active errors are closed at this checkpoint.

Closed errors:

- `M03-E1`: normal CUDA process exit now emits process-scoped cleanup.
- `M03-E2`: forced `SIGKILL` stale-owner cleanup is handled by the next CUDA
  process.
- `M03-E3`: Driver event and stream-wait paths now use host RPC.
- `M03-E4`: large BAR1 copy single-call stall is closed by 256 KiB copy
  chunking.

`M03-E2` forced-kill closure facts:

- A CUDA process allocated 64 MiB and was killed with `SIGKILL`.
- The killed process PID `3948` remained in `/tmp/vgpu_cuda_owner_pids`.
- The next CUDA process detected the stale owner and sent cleanup for owner
  `167776108`.
- Host log shows `CUDA process cleanup vm_id=10 owner=167776108`.
- Host log shows `Cleaned up VM 167776108`.
- Raw CUDA immediately after the kill passed.
- Final Plan A passed:
  `/tmp/phase1_milestone_gate_after_m03_e2_stale_owner_sweep.json` ->
  `overall_pass=True`.
- Final raw CUDA gate passed:
  `/tmp/phase3_general_cuda_gate_after_m03_e2_stale_owner_sweep.json` ->
  `overall_pass=True`.

Current active evidence:

`M03-E3` closure:

- `cuEventCreate`, `cuEventDestroy`, `cuEventRecord`, `cuEventSynchronize`,
  `cuEventQuery`, and `cuStreamWaitEvent` now use host RPC paths.
- Bounded async/mixed stream-event probe passed:
  `/tmp/async_stream_event_probe_repeat.json` -> `overall_pass=True`,
  `pass_count=5`, `runs=5`, `bytes_per_run=4194304`.
- Final Plan A passed:
  `/tmp/phase1_milestone_gate_m03_final_after_chunking.json` ->
  `overall_pass=True`.
- Final raw CUDA gate passed:
  `/tmp/phase3_general_cuda_gate_m03_final_after_chunking.json` ->
  `overall_pass=True`.
- Host log after the repeated gate and final regressions shows no fresh
  `sync FAILED`, `CUDA_ERROR_ILLEGAL_ADDRESS`, `invalid handle`, or unsupported
  CUDA protocol calls.
- `M03-E4` large 4 MiB async HtoD over BAR1 initially stalled as one single
  transfer. It is closed by chunking BAR1 copy calls to 256 KiB pieces in
  `guest-shim/cuda_transport.c`.

Source evidence:

- `CUDACallHeader` carried `vm_id` but no guest PID/process identity at
  milestone start.
- `cuda_executor_cleanup_vm()` frees VM memory, streams, events, modules,
  libraries, cuBLAS handles, and pending async HtoD buffers, but it is only
  called during executor destruction.
- A mediator-only connection cleanup attempt was insufficient: the persistent
  mediator fd is owned by the QEMU vGPU stub, not by each guest CUDA process.

Original impact:

A killed or exited CUDA process can leave host-side allocations and handles alive
until mediator restart. Cleaning by whole VM is unsafe because Ollama or another
process in the same VM may still own live GPU state.

Closure:

- use BAR0 scratch register `0x03c` to pass guest PID from transport to stub;
- derive an internal executor owner id from `(vm_id, pid)`;
- add internal `CUDA_CALL_PROCESS_CLEANUP`;
- emit process cleanup from the Driver shim unload path.

Live proof:

- QEMU RPM installed and `vgpu-cuda` is present in device help.
- Test-10 rebooted onto the rebuilt QEMU stub.
- Mediator PID `331347` is running from `/root/phase3/mediator_phase3`.
- Host log shows `CUDA process cleanup: count=12`.
- Host log shows `Cleaned up VM: count=12`.
- Host log shows no `Unsupported CUDA protocol call: vgpuProcessCleanup`.
- Host log shows no `sync FAILED` or `CUDA_ERROR_ILLEGAL_ADDRESS`.
- Plan A passed:
  `/tmp/phase1_milestone_gate_after_m03_e1_process_cleanup.json` ->
  `overall_pass=True`.
- Raw CUDA gate passed:
  `/tmp/phase3_general_cuda_gate_after_m03_e1_process_cleanup.json` ->
  `overall_pass=True`.

## Candidate List

- BAR1 fallback after shmem GPA resolution fails with `pfn_hidden`.
- Residual `cuFuncGetParamInfo(0x00bc)` unsupported/invalid-value noise.

## Last Proven Checkpoint

Milestone 02 closed with:

- Plan A pass after verified live mediator restart;
- Milestone 01 raw CUDA gate pass after the same restart;
- protocol coverage comparison `protocol_ids=86`, `executor_case_ids=86`,
  `missing_cases=[]`.

## Closure Condition

Milestone 03 closes when:

- the memory/sync/cleanup gate is implemented or explicitly defined as an audit
  gate with evidence;
- repeated safe runs pass;
- forced-kill recovery is proven or a precise active error is closed;
- host/VM evidence shows cleanup behavior is understood;
- Plan A remains pass after any runtime behavior changes;
- candidates are recorded for later milestones.

## Next Single Step

Proceed to Milestone 04 only after the user confirms or asks to continue.
