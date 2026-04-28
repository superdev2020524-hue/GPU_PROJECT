# Evidence - Milestone 03 Memory, Synchronization, And Cleanup

## Baseline Evidence

- Plan A before Milestone 03:
  `/tmp/phase1_milestone_gate_before_m03.json` -> `overall_pass=True`.
- Milestone 02 final raw CUDA regression:
  `/tmp/phase3_general_cuda_gate_after_m02_e3_live_binary_restart.json` ->
  `overall_pass=True`.
- Milestone 02 final live mediator proof:
  `/root/phase3/mediator_phase3`
  `8f306df61150071553a5dc7c9b8cba257658111acf6a71331aaa2fb7ebebe796`;
  `/root/phase3/src/cuda_executor.c`
  `0f2b66f05b9c633592f44cdb4fe4b1596b63d2733eaedccd3af2518d35cfd21c`.

## Evidence To Collect

- Source-level memory and lifetime behavior from:
  - `src/cuda_executor.c`
  - `guest-shim/libvgpu_cuda.c`
  - `guest-shim/libvgpu_cudart.c`
  - transport/protocol files as needed
- Host mediator evidence from `/tmp/mediator.log`.
- VM probe output for memory/copy/sync integrity.
- Explicit proof for any forced process termination test.

## Current Session Findings

Milestone 03 has started. `M03-E1` and `M03-E2` are closed.

### Initial Facts

- Plan A passed before Milestone 03 work.
- Milestone 03 folder existed with `README.md`; required registry files were
  added before implementation.
- The first work item was an audit/gate-definition pass, not a code change.

### Source Audit Evidence For `M03-E1`

- `include/cuda_protocol.h` defines `CUDACallHeader` with `vm_id` but no guest
  PID/process identity.
- `src/cuda_executor.c` stores memory, module, library, function, stream, event,
  cuBLAS, and pending async HtoD state in `vm_state_t`, keyed by `vm_id`.
- `cuda_executor_cleanup_vm()` frees the relevant executor-owned resources, but
  source search found it called only from `cuda_executor_destroy()`.
- `src/mediator_phase3.c` removes and closes persistent CUDA fds when
  `handle_persistent_message()` returns false, including peer EOF/read error,
  but does not notify the CUDA executor.

### Active Error

None at this checkpoint.

### Closure Proof Required

- Deploy a process-owner protocol across guest transport, QEMU vGPU stub,
  mediator, and executor.
- Emit `CUDA_CALL_PROCESS_CLEANUP` when a CUDA-using guest process unloads the
  Driver shim.
- Prove the host receives process cleanup for the `(vm_id,pid)` owner id.
- Prove cleanup does not tear down other live processes in the same VM.
- Re-run Plan A and the raw CUDA gate after deployment.

### Mediator-Only Attempt Rejected As Closure

The mediator was changed and deployed once to clean executor state on persistent
connection close. Regression passed:

- Plan A:
  `/tmp/phase1_milestone_gate_after_m03_e1_connection_scoped_cleanup.json` ->
  `overall_pass=True`.
- Raw CUDA:
  `/tmp/phase3_general_cuda_gate_after_m03_e1_connection_scoped_cleanup.json` ->
  `overall_pass=True`.

However, `/tmp/mediator.log` showed only the persistent fd registration and no
cleanup firing after the short probes:

- `registered for persistent: count=1`
- `cleaning executor connection state: count=0`
- `Cleaned up VM: count=0`

Conclusion: the persistent fd is VM/QEMU-stub level, not guest-process level.
This attempt is useful regression evidence but does not close `M03-E1`.

### Local Cross-Layer Fix Drafted

Local source now contains the bounded process-owner direction:

- `include/cuda_protocol.h`: adds internal `CUDA_CALL_PROCESS_CLEANUP`.
- `guest-shim/cuda_transport.c`: writes `getpid()` to BAR0 scratch before CUDA
  doorbell and adds `cuda_transport_process_cleanup()`.
- `guest-shim/libvgpu_cuda.c`: emits cleanup from the Driver shim destructor
  when a transport exists.
- `src/vgpu-stub-enhanced.c`: derives executor owner id from `(vm_id,pid)`.
- `src/mediator_phase3.c`: intercepts `CUDA_CALL_PROCESS_CLEANUP` and calls
  `cuda_executor_cleanup_vm()` for that owner.
- `src/cuda_executor.c`: classifies the new protocol id explicitly for the API
  coverage discipline.

### Live Deployment Proof For `M03-E1`

Host files copied and built:

- `/root/phase3/mediator_phase3`
  `f1384d50c9252210ee7e8c5f936a2909044a8f69c0f052d504055eff2add9308`
- `/root/phase3/src/mediator_phase3.c`
  `92a3764fa3f63ee664893c60c454390f20d4c3829d2c7320590b852457fe71e6`
- `/root/phase3/src/cuda_executor.c`
  `2cdb6948526143dde07d79177cd9a0d72978bf1380c776682b82a3eccb17c89f`
- `/root/phase3/src/vgpu-stub-enhanced.c`
  `b2841444ccae7e981cc3ef754049ee09785e29b9f513a1db3402c0aaadc3536a`
- `/root/phase3/include/cuda_protocol.h`
  `861e23e007dca15303144d81908248c5aaf9e18134fdf5e69652f65b5b4906ad`

QEMU proof:

- RPM built:
  `/root/vgpu-build/rpmbuild/RPMS/x86_64/qemu-4.2.1-5.2.15.2.xcpng8.3.x86_64.rpm`
  `ad9a5fbe348dec3194cf271fb61e9ac2438a601b32339de1f7f69533a7643d51`
- Installed QEMU binary:
  `/usr/lib64/xen/bin/qemu-system-i386`
  `ca2275145da5bcfc1d0a48f501c6feb6b3c707e478155b5c2007692dd648eab3`
- `qemu-system-i386 -device help` includes `vgpu-cuda`.
- Test-10 was stopped, QEMU RPM installed, mediator restarted, and Test-10
  started again.

Guest proof:

- `/opt/vgpu/lib/libvgpu-cuda.so.1`
  `b19496ae3f1ecc0c4610df1d8df39f54c6494f8bfa326413959b3b4b192dfcff`
- `/opt/vgpu/lib/libvgpu-cudart.so`
  `66b10c345acd084164b115df5fc7b9b8851fe18583610e7c3d12ac90f17149cc`
- `/home/test-10/phase3/guest-shim/cuda_transport.c`
  `196b75f7fc7a7376aa7b1da5e0d35abe048f512678c1ce20d4ba707a355c01c6`
- `/home/test-10/phase3/include/cuda_protocol.h`
  `861e23e007dca15303144d81908248c5aaf9e18134fdf5e69652f65b5b4906ad`
- VM sees vGPU PCI device `00:05.0 10de:2331`.
- Ollama service is active after restart.

Cleanup proof:

- Host log: `CUDA process cleanup: count=12`.
- Host log: `Cleaned up VM: count=12`.
- Latest owner cleanup examples:
  - `owner=167774650 request_id=184`
  - `owner=167774689 request_id=198`
  - `owner=167774728 request_id=212`
  - `owner=167774767 request_id=226`
  - `owner=167774806 request_id=240`
- Host log: `Unsupported CUDA protocol call: vgpuProcessCleanup: count=0`.
- Host log: `sync FAILED: count=0`.
- Host log: `CUDA_ERROR_ILLEGAL_ADDRESS: count=0`.

Regression proof:

- Plan A:
  `/tmp/phase1_milestone_gate_after_m03_e1_process_cleanup.json` ->
  `overall_pass=True`.
- Raw CUDA:
  `/tmp/phase3_general_cuda_gate_after_m03_e1_process_cleanup.json` ->
  `overall_pass=True`, Driver API 5/5 and Runtime API 5/5.

### Next Single Step

Promote and test the forced process kill lane.

## `M03-E2` Evidence Plan

Bounded repro:

- Run one CUDA process that calls `cuInit`, allocates device memory, prints its
  PID, then sleeps.
- Kill that process with `SIGKILL`.
- Run a fresh raw CUDA safe probe.
- Check host mediator log for:
  - whether `CUDA_CALL_PROCESS_CLEANUP` fired for the killed PID;
  - whether the next safe probe passes;
  - whether host log shows `sync FAILED`, `CUDA_ERROR_ILLEGAL_ADDRESS`, or stale
    handle errors.

Closure/disposition:

- If the next safe probe fails, `M03-E2` remains active as forced-kill poisoning.
- If the next safe probe passes but the killed owner is not cleaned, close only
  "next-run poisoning" and carry forward a bounded leak candidate.

## `M03-E2` Closure Evidence

Initial forced-kill classification:

- First forced-kill probe allocated 64 MiB at owner `167775312` and was killed.
- Fresh raw CUDA probe passed:
  `/tmp/phase3_general_cuda_gate_after_m03_e2_forced_kill.json` ->
  `overall_pass=True`.
- Host log showed no cleanup for owner `167775312`, proving the need for stale
  owner cleanup beyond destructor cleanup.

Fix:

- `guest-shim/cuda_transport.c` now maintains
  `/tmp/vgpu_cuda_owner_pids`.
- On transport init, a CUDA process scans that registry and sends
  `CUDA_CALL_PROCESS_CLEANUP` for registered PIDs that no longer exist in
  `/proc`.
- On normal cleanup, the process unregisters itself.
- For cleanup calls, BAR0 scratch uses the target PID from cleanup args instead
  of the current process PID.

Live proof:

- Deployed guest transport:
  `/home/test-10/phase3/guest-shim/cuda_transport.c`
  `0ff906eb2f20c166de9d7d9335a51ce206543408836f514e7000dda2df135e91`.
- Deployed Driver shim:
  `/opt/vgpu/lib/libvgpu-cuda.so.1`
  `476d7ff3b2fecc7e42f789ba258f79a55b90291d458ae423ec7b1dc988b28bcd`.
- Forced-kill process actual CUDA PID: `3948`.
- Registry before and after kill contained `3948`.
- Expected owner id: `167776108`.
- Host log: `CUDA process cleanup vm_id=10 owner=167776108 request_id=287`.
- Host log: `Cleaned up VM 167776108`.
- Host log after final regression:
  - `CUDA process cleanup: count=27`
  - `Cleaned up VM: count=27`
  - `sync FAILED: count=0`
  - `CUDA_ERROR_ILLEGAL_ADDRESS: count=0`
  - `Unsupported CUDA protocol call: vgpuProcessCleanup: count=0`

Regression proof:

- Plan A:
  `/tmp/phase1_milestone_gate_after_m03_e2_stale_owner_sweep.json` ->
  `overall_pass=True`.
- Raw CUDA:
  `/tmp/phase3_general_cuda_gate_after_m03_e2_stale_owner_sweep.json` ->
  `overall_pass=True`, Driver API 5/5 and Runtime API 5/5.

Conclusion:

`M03-E2` is closed. Forced kill no longer poisons the next CUDA process and the
next CUDA process cleans the stale owner.

## `M03-E3` Closure Evidence

Active error:

- Driver API events used fake guest-local handles and success returns.
- `cuStreamWaitEvent` returned success locally without proving host stream/event
  ordering.
- Host executor already had real stream and event call handlers, so the missing
  coverage was in `guest-shim/libvgpu_cuda.c`.

Fix:

- `cuEventCreate`, `cuEventDestroy`, `cuEventRecord`, `cuEventSynchronize`,
  `cuEventQuery`, and `cuStreamWaitEvent` now call the host via RPC.
- New bounded probe:
  `phase3/tests/memory_sync_cleanup/async_stream_event_probe.c`.

Live artifact proof:

- Guest `/opt/vgpu/lib/libvgpu-cuda.so.1`:
  `05f3cc5dc992db4eea974b98df1057ad8db1358c5487436c1f038d3dd7c32739`.
- Guest `/home/test-10/phase3/guest-shim/libvgpu_cuda.c`:
  `73bcd11330383ecdc2b36d1ef1f7d1a4685453e120d14c4eb4009ad1ce007bcf`.
- Guest `/home/test-10/phase3/guest-shim/cuda_transport.c`:
  `8ec007496ef8cc5d702000717ccb2f963d2cc3e75aa4282781cd09952a96f2bc`.
- VM async probe binary `/tmp/async_stream_event_probe`:
  `444aec611165834da963eeb9f1c5bb44b4f3285474cc67916958aee0ddffd6a5`.
- VM async probe source `/tmp/async_stream_event_probe.c`:
  `38aff40075042b20c083eb081c2d94b8b87930fa90aed88ff05312d95e8471e5`.

Gate proof:

- One direct async/mixed probe initially passed with 4 KiB byte verification.
- A later 4 MiB run exposed `M03-E4`: single-call BAR1 async HtoD could stall.
- `M03-E4` fix: cap BAR1 copy chunks to 256 KiB in
  `guest-shim/cuda_transport.c`.
- Repeated fresh-process async/mixed probe passed:
  `/tmp/async_stream_event_probe_repeat.json` -> `overall_pass=True`,
  `pass_count=5`, `runs=5`, `bytes_per_run=4194304`.
- Host mediator proof after the repeated gate:
  - `call_id=0x32` HtoD traffic: `count=56`;
  - `call_id=0x33` HtoDAsync traffic: `count=320`;
  - `call_id=0x34` DtoH traffic: `count=15`;
  - `call_id=0x35` DtoD traffic: `count=15`;
  - `call_id=0x71` event-create traffic: `count=62`;
  - `call_id=0x73` event-record traffic: `count=61`;
  - `call_id=0x74` event-synchronize traffic: `count=45`;
  - `call_id=0x75` event-query traffic: `count=15`;
  - `call_id=0x66` stream-wait-event traffic: `count=15`;
  - `call_id=0x72` event-destroy traffic: `count=60`;
  - `CUDA process cleanup: count=73`;
  - `Cleaned up VM: count=73`;
  - `sync FAILED: count=0`;
  - `CUDA_ERROR_ILLEGAL_ADDRESS: count=0`;
  - `invalid handle: count=0`;
  - `Unsupported CUDA protocol call: count=0`.

Final regression proof:

- Plan A:
  `/tmp/phase1_milestone_gate_m03_final_after_chunking.json` ->
  `overall_pass=True`.
- Raw CUDA:
  `/tmp/phase3_general_cuda_gate_m03_final_after_chunking.json` ->
  `overall_pass=True`, Driver API 5/5 and Runtime API 5/5.

## Milestone 03 Closure

Milestone 03 is complete at this checkpoint. Closed active errors:

- `M03-E1`: normal CUDA process-exit cleanup.
- `M03-E2`: forced-kill stale-owner cleanup.
- `M03-E3`: RPC-backed Driver event and stream-wait behavior.
- `M03-E4`: large BAR1 copy single-call stall, closed by 256 KiB copy chunking.

Carry-forward observations:

- BAR1 remains the live fallback when shmem GPA resolution reports
  `pfn_hidden`; this is not a Milestone 03 blocker because the bounded gates
  pass through BAR1.
- Residual `cuFuncGetParamInfo(0x00bc)` noise is carried forward as known
  compatibility noise, not the active Milestone 03 blocker.

## Post-Closure Serial Preservation Recheck

Reason:

- User correctly raised that milestone closure must preserve previous stages,
  not only pass the newest gate.
- This recheck was run after the final Milestone 03 transport and event changes.

Serial result:

- `00_preserve_ollama_baseline`: pass.
  `/tmp/phase1_milestone_gate_serial_00_after_m03.json` ->
  `overall_pass=True`.
- Optional `00` Plan B Tiny lane: pass.
  `/tmp/phase1_plan_b_serial_00_after_m03.json` -> `overall_pass=True`.
- Optional `00` Plan C client lane: initially not proven, then closed after
  bounded investigation.
  - HTTP `/api/generate` for `qwen2.5:3b` returned `12` in 5.83s.
  - `ollama run MODEL PROMPT_AS_ARG` timed out for both `qwen2.5:0.5b` and
    `qwen2.5:3b`.
  - `printf PROMPT | ollama run MODEL` succeeded for both models.
  - Root cause: Plan C gate invocation method, not Milestone 03 runtime
    memory/sync behavior.
  - Fix: `phase1_plan_c_client_gate.py` now sends prompts through stdin and
    normalizes CLI terminal-control output before comparing exact answers.
  - Fixed clean-state Plan C report:
    `/tmp/phase1_plan_c_serial_00_after_m03_fixed_clean.json` ->
    `overall_pass=True`, all C1-C5 cases pass, final `/api/ps` is empty.
- `01_general_cuda_gate`: pass.
  `/tmp/phase3_general_cuda_gate_serial_01_after_m03.json` ->
  `overall_pass=True`, Driver API 5/5 and Runtime API 5/5.
- `02_api_coverage_audit`: pass.
  `/tmp/phase3_api_audit_serial_02_after_m03.json` ->
  `overall_pass=True`, `protocol_ids_excluding_sentinel=87`,
  `missing_executor_name_mentions=[]`, `missing_matrix_terms=[]`,
  `missing_gap_terms=[]`.
- `03_memory_sync_cleanup`: pass.
  `/tmp/async_stream_event_probe_repeat.json` -> `overall_pass=True`,
  `pass_count=5`, `runs=5`, `bytes_per_run=4194304`.
- Final post-Plan-C-fix preservation:
  - Plan A:
    `/tmp/phase1_milestone_gate_serial_00_after_planc_fix.json` ->
    `overall_pass=True`.
  - Plan B:
    `/tmp/phase1_plan_b_serial_00_after_planc_fix.json` ->
    `overall_pass=True`.
  - Final `/api/ps`: `{"models":[]}`.

Important qualification:

- Milestone 02 is an audit milestone, so its preservation recheck is a source
  and registry consistency check, not a runtime workload gate.
- `CUDA_CALL_MAX` is a sentinel macro and was explicitly excluded from the
  protocol/executor consistency count.
- The strict serial preservation claim now covers `00` Plan A, optional
  `00` Plan B, optional `00` Plan C, `01`, `02`, and `03` after Milestone 03.
