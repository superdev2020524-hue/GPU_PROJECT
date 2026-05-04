# Evidence - Milestone 07 Security And Isolation

## Baseline Evidence

M07 begins from the expanded M06 closure state documented in
`../06_multiprocess_multivm/` and summarized in `BASELINE.md`.

## Evidence To Collect

- Source audit of request-bound checks.
- Live mediator/socket proof before malformed probes.
- Bounded malformed request reports.
- Mediator health before and after each malformed probe.
- Known-good Test-10 and Test-6 gate results after malformed probes.
- Classification of any observed failure as active error or candidate.

## Current Session Findings

M07 is active. `M07-E1` is closed and `M07-E2` is active.

## Source Audit - Request Boundary

- `include/vgpu_protocol.h` defines the socket header and the intended
  `VGPU_CUDA_SOCKET_MAX_PAYLOAD` as 8 MiB.
- Before M07, `src/mediator_phase3.c` read `hdr->payload_len` and allocated that
  amount for CUDA calls without enforcing the 8 MiB CUDA payload ceiling before
  `malloc`.
- The vGPU stub bounded BAR1/shmem payload selection, but that did not protect
  the mediator socket directly from a malformed socket header.
- M07 patch added:
  - pre-allocation socket payload ceiling:
    `sizeof(CUDACallHeader) + VGPU_CUDA_SOCKET_MAX_PAYLOAD` for CUDA calls;
  - CUDA payload magic validation;
  - `num_args <= CUDA_MAX_INLINE_ARGS` validation;
  - declared `cuda_hdr.data_len` equals actual bulk length validation.

## M07-E1 Malformed Socket Probe

- Probe:
  `phase3/tests/security_isolation/malformed_socket_probe.py`.
- Build/deploy:
  host `/root/phase3/mediator_phase3` rebuilt successfully after the M07 bounds
  patch and restarted.
- Live artifact:
  host `/root/phase3/mediator_phase3`
  `a203b3e8b20a7476eb5fbbb5cea5312a1997f2ce7573b2444dcfff26d5afa618`;
  host `/root/phase3/src/mediator_phase3.c`
  `b98b08fb015a4179596b2d3cb0ac942db1ea417667a7d2d26a8029e64d8c18d7`.
- Restart proof:
  mediator PID `946699` started from `./mediator_phase3`, initialized CUDA on
  `NVIDIA H100 PCIe`, and discovered two VM sockets:
  `/var/xen/qemu/root-2/tmp/vgpu-mediator.sock` and
  `/var/xen/qemu/root-6/tmp/vgpu-mediator.sock`.
- Expanded malformed probe:
  dom0 `/tmp/m07_malformed_socket_probe_after_bounds.json` ->
  `overall_pass=True`.
- Cases closed safely:
  - invalid socket magic;
  - truncated socket header;
  - unknown message type;
  - CUDA payload shorter than `CUDACallHeader`;
  - oversized declared CUDA payload;
  - invalid CUDA payload magic;
  - CUDA `num_args` overflow;
  - CUDA declared/actual data length mismatch;
  - control ping still returned a PONG.
- Mediator log attribution:
  the malformed cases were logged on the Test-6 socket path with messages
  including `Invalid magic`, `Incomplete header`, `CUDA call payload too short`,
  `Oversized payload`, `Invalid CUDA payload magic`,
  `CUDA num_args out of range`, and `CUDA payload length mismatch`.
- Post-probe mediator health:
  `sync FAILED=0`, `CUDA_ERROR_ILLEGAL_ADDRESS=0`,
  `Unsupported CUDA protocol call=0`, `invalid handle=0`, heartbeat alive with
  two sockets.
- Post-probe known-good gates:
  Test-6 `/tmp/m07_post_bounds_test6_cupy.json` -> `overall_pass=True`;
  Test-10 `/tmp/m07_post_bounds_test10_cupy.json` -> `overall_pass=True`.

## Active Error After M07-E1

`M07-E2`: quarantine/rate-limit rejection behavior and guest BAR/MMIO permission
assumptions are not yet proven for the current live baseline.

## M07-E2 Quarantine, Rate Limit, And BAR Policy

### BAR/MMIO Permission Baseline

- Live Test-6 `/sys/bus/pci/devices/0000:00:05.0/resource0`:
  mode `0666`, readable, writable, size `4096`.
- Live Test-6 `/sys/bus/pci/devices/0000:00:05.0/resource1`:
  mode `0666`, readable, writable, size `16777216`.
- Live Test-10 showed the same `resource0`/`resource1` `0666` policy.
- Source cause:
  `guest-shim/install.sh` intentionally grants `0666` to BAR resources in
  `check_vgpu_device`, udev rules, and `vgpu-devices.service` so non-root
  CUDA/Ollama processes can open BAR0/BAR1.
- M07 classification:
  this is not hidden. Current baseline trusts in-guest users enough to expose
  the vGPU BAR transport interface. Production tenant hardening should replace
  this with a narrower group/device-policy model or hypervisor mediation, but
  it does not block M07 because the current gate explicitly defers full
  IOMMU/hypervisor hardening unless it causes a reachable host crash or
  cross-VM corruption.

### Quarantine Sync Fix

- Gap found:
  `vgpu-admin quarantine-vm` updated the DB and asked the mediator to reload
  config, but `src/mediator_phase3.c` only reloaded rate-limit settings. The
  DB quarantine field was left as a comment and was not synced into the live
  watchdog state.
- Fix:
  added `wd_set_quarantine(watchdog_t *, vm_id, quarantined)` to
  `include/watchdog.h` and `src/watchdog.c`, then called it during mediator
  config reload for every VM.
- Host rebuild:
  `make mediator_phase3` succeeded.
- Live artifact after quarantine fix:
  host `/root/phase3/mediator_phase3`
  `78944e853c2557e40d545fe22ccd3c9e1f04dc0411bcb3298ade9c2f8e233355`;
  host `/root/phase3/src/mediator_phase3.c`
  `d374806149c24447f76f6a037aafa75ac5bf7cb0c68df6f4f0993363e709915e`;
  host `/root/phase3/src/watchdog.c`
  `d378b088b8a4c3f876bddd661d6ee6ceff4ab6e1a4a5b445badb7cd881ff1f4c`;
  host `/root/phase3/include/watchdog.h`
  `1c8d2705d551bb3ea01c8313f346730ad58429b9ebfe68750bd58a9bd67ce7a4`.
- Restart proof:
  mediator PID `950225` initialized CUDA, discovered the Test-10/Test-6
  sockets, and continued heartbeat.

### Quarantine Probe

- Command path:
  `/root/phase3/vgpu-admin quarantine-vm --vm-uuid=1d11c550-01bb-d9f0-b33b-e9cdc1d7e8e1`.
- Mediator log:
  `[WATCHDOG] Quarantine set: vm=6`.
- Cross-VM isolation:
  while Test-6 was quarantined, Test-10 CuPy probe
  `/tmp/m07_quarantine_test10_cupy.json` returned `overall_pass=True`.
- Targeted rejection:
  Test-6 CuPy failed as expected with mediator/shim errors:
  `VM_QUARANTINED` for `cuInit`, `cuGetGpuInfo`, `cuMemAlloc_v2`,
  `cuDevicePrimaryCtxRelease`, and cleanup.
- Recovery:
  `/root/phase3/vgpu-admin clear-quarantine --vm-uuid=1d11c550-01bb-d9f0-b33b-e9cdc1d7e8e1`;
  DB row became `vm_id=6 quarantined=0 error_count=0`;
  mediator log showed `[WATCHDOG] Quarantine cleared: vm=6`;
  Test-6 CuPy `/tmp/m07_post_quarantine_clear_test6_cupy.json` returned
  `overall_pass=True`.

### Rate-Limit Probe

- Command path:
  `/root/phase3/vgpu-admin set-rate-limit --vm-uuid=1d11c550-01bb-d9f0-b33b-e9cdc1d7e8e1 --rate=1 --queue-depth=0`.
- DB row:
  `vm_id=6 max_jobs_per_sec=1 max_queue_depth=0 quarantined=0`.
- Mediator log:
  `[RATE-LIMIT] vm=6: rate=1 jobs/sec, max_queue=0`.
- Targeted rejection:
  Test-6 CuPy failed as expected with `RATE_LIMITED` for CUDA calls including
  `cuGetGpuInfo`, `cuMemAlloc_v2`, `cuDevicePrimaryCtxRelease`, and cleanup.
  Mediator log included `vm=6 rate-limited (CUDA call, code=1)`.
- Recovery:
  `/root/phase3/vgpu-admin set-rate-limit --vm-uuid=1d11c550-01bb-d9f0-b33b-e9cdc1d7e8e1 --rate=0 --queue-depth=0`;
  DB row became `vm_id=6 max_jobs_per_sec=0 max_queue_depth=0 quarantined=0`;
  Test-6 CuPy `/tmp/m07_post_rate_restore_test6_cupy.json` returned
  `overall_pass=True`.

### M07-E2 Closure

`M07-E2` is closed for the M07 gate: quarantine and rate-limit rejection paths
are proven, the target VM recovers after policy restoration, another VM remains
healthy while the bad VM is restricted, and the current BAR `0666` exposure is
documented as a current engineering trust assumption rather than an unexamined
gap.

## Final Malformed Probe On Quarantine-Sync Binary

- Final live mediator artifact:
  `/root/phase3/mediator_phase3`
  `78944e853c2557e40d545fe22ccd3c9e1f04dc0411bcb3298ade9c2f8e233355`,
  PID `950225`.
- Final rerun:
  dom0 `/tmp/m07_malformed_socket_probe_final.json` ->
  `overall_pass=True`.
- Covered cases remained green after the second mediator rebuild:
  invalid socket magic, truncated header, unknown message type, short CUDA
  payload, oversized declared payload, invalid CUDA magic, `num_args` overflow,
  CUDA data length mismatch, and control ping.

## `00 -> 06` Serial Preservation After M07

Status: in progress.

- Baseline policy before preservation:
  Test-6 and Test-10 DB rows both had `rate=0`, `queue_depth=0`,
  `quarantined=0`, `error_count=0`; mediator PID `950225` was running the
  final M07 binary.
- Plan A first run showed accuracy and speed pass but immediate
  `force_unload` `/api/ps` check still saw `qwen2.5:0.5b`. A delayed `/api/ps`
  check immediately afterward showed the model absent. Classified as gate timing
  rather than model residency failure.
- Plan A gate fixed by waiting briefly for model absence after force unload.
  Rerun `/tmp/m07_preserve_plan_a_fixed.json` -> `overall_pass=True`;
  accuracy, speed, and residency all passed; `force_unload.absent_after_wait=True`.
- Plan B first run hit the same immediate `B4_force_unload` timing issue.
  Plan B gate fixed with the same wait-for-absence behavior. Rerun
  `/tmp/m07_preserve_plan_b_fixed.json` -> `overall_pass=True`;
  B1/B2/B3/B4 all passed; `B4_force_unload.absent_after_wait=True`.
- Plan C `/tmp/m07_preserve_plan_c.json` -> `overall_pass=True`.
- M01 raw CUDA `/tmp/m07_preserve_m01_general_cuda.json` ->
  `overall_pass=True`, 15 cases passed.
- M02 API coverage source audit
  `/tmp/m07_preserve_m02_api_coverage_source_audit.json` ->
  `overall_pass=True`, `protocol_ids=87`, `executor_case_ids=87`,
  `missing_cases=[]`, sentinel `CUDA_CALL_MAX` ignored.
- M03 async stream/event probe passed:
  `ASYNC_STREAM_EVENT_PROBE PASS bytes=4194304`.
- M03 forced-kill lane passed:
  `forced_kill_alloc_probe` printed `READY pid=42845 ptr=0x7f4b94000000`,
  then was killed with `SIGKILL`; post-kill raw CUDA
  `/tmp/m07_preserve_m03_post_kill_raw_cuda.json` ->
  `overall_pass=True`, 11 cases passed.
- M04 PyTorch `/tmp/m07_preserve_pytorch.json` -> `overall_pass=True`;
  `cuda_available`, `device_count`, `tensor_htod_dtoh`, `elementwise_add`,
  `matmul`, `small_nn_inference`, and `repeated_warm_execution` all passed.
- M05 CuPy `/tmp/m07_preserve_cupy.json` -> `overall_pass=True`;
  `import_cupy`, `device_count`, `device_name`, `tensor_htod_dtoh`,
  `elementwise_add`, `matmul`, and `repeated_warm_execution` all passed.
- M06 same-VM two-process CuPy
  `/tmp/m07_preserve_m06_two_process.json` -> `overall_pass=True`;
  both children passed.
- M06 same-VM mixed framework
  `/tmp/m07_preserve_m06_mixed.json` -> `overall_pass=True`;
  PyTorch child passed and CuPy child passed.
- M06 cross-VM concurrent CuPy
  `/tmp/m07_preserve_m06_cross_vm_cupy.json` -> `overall_pass=True`;
  Test-10 CuPy passed and Test-6 CuPy passed.

## M07 Closure Status

Milestone 07 is closed for the defined gate:

- malformed mediator socket/CUDA-header inputs fail closed;
- mediator stays alive and attributes bad requests;
- targeted quarantine rejects one VM while another VM stays healthy;
- clearing quarantine restores the VM;
- rate limiting rejects the target VM and restoring unlimited mode recovers it;
- BAR0/BAR1 `0666` exposure is explicitly documented as a current engineering
  trust assumption and production-hardening candidate;
- serial preservation from Plan A/Plan B/Plan C through M01-M06 is green on the
  final M07 mediator/watchdog baseline.

## 2026-04-30 Reopen/Closure Refresh After TensorFlow Scope Challenge

Milestone 07 was treated as open again because the latest user-facing summary
did not carry forward the full M07 closure chain after TensorFlow entered the
M05 evidence set. The current run re-proved the M07 gate instead of relying only
on historical closure.

### Live Baseline Proof

- Live mediator:
  `/root/phase3/mediator_phase3`
  `cb9c99626fa8bd6f06a2191cfa5780227c5926f0e738227867653c69a1ff6de8`,
  PID `1045424` at entry, later restarted cleanly as PID `1846882` after a
  TensorFlow illegal-address fault during preservation.
- Live M07 source artifacts:
  - `/root/phase3/src/mediator_phase3.c`
    `d374806149c24447f76f6a037aafa75ac5bf7cb0c68df6f4f0993363e709915e`
  - `/root/phase3/src/watchdog.c`
    `d378b088b8a4c3f876bddd661d6ee6ceff4ab6e1a4a5b445badb7cd881ff1f4c`
  - `/root/phase3/include/watchdog.h`
    `1c8d2705d551bb3ea01c8313f346730ad58429b9ebfe68750bd58a9bd67ce7a4`
- Live sockets:
  `/var/xen/qemu/root-2/tmp/vgpu-mediator.sock`,
  `/var/xen/qemu/root-6/tmp/vgpu-mediator.sock`,
  `/var/xen/qemu/root-7/tmp/vgpu-mediator.sock`.
- VM-10 and VM-6 BAR policy:
  `resource0` and `resource1` remained `0666`, matching the documented
  engineering trust assumption.

### Current M07 Gate Rerun

- Malformed socket probe:
  `/tmp/m07_current_malformed_socket_probe.json` -> `overall_pass=True`.
- Covered cases:
  invalid socket magic, truncated header, unknown message type, short CUDA
  payload, oversized declared CUDA payload, invalid CUDA magic, `num_args`
  overflow, CUDA data length mismatch, and control ping.
- Mediator remained alive and attributed bad requests to VM-6 in the log
  (`Invalid magic`, `Incomplete header`, `Oversized payload`, invalid CUDA
  payload magic, `CUDA num_args out of range`, and CUDA payload length mismatch).

### Quarantine And Rate-Limit Rerun

- VM-6 baseline raw CUDA probe:
  `/tmp/m07_vm6_raw_cuda_baseline.log` -> `OVERALL PASS`.
- Quarantine:
  `/root/phase3/vgpu-admin quarantine-vm --vm-uuid=1d11c550-01bb-d9f0-b33b-e9cdc1d7e8e1`;
  mediator logged `[WATCHDOG] Quarantine set: vm=6`.
- Cross-VM non-poisoning while VM-6 was quarantined:
  VM-10 CuPy `/tmp/m07_quarantine_test10_cupy_current.log` ->
  `overall_pass=True`.
- Targeted rejection:
  VM-6 raw CUDA `/tmp/m07_quarantine_vm6_raw_current.log` failed as expected
  with `VM_QUARANTINED` on CUDA calls.
- Recovery:
  `clear-quarantine` restored VM-6, and
  `/tmp/m07_post_quarantine_clear_vm6_raw_current.log` -> `OVERALL PASS`.
- Rate limit:
  `set-rate-limit --rate=1 --queue-depth=0` for VM-6 logged
  `[RATE-LIMIT] vm=6: rate=1 jobs/sec, max_queue=0`.
- Targeted rejection:
  VM-6 raw CUDA `/tmp/m07_rate_limited_vm6_raw_current.log` failed as expected
  with `RATE_LIMITED`.
- Recovery:
  `set-rate-limit --rate=0 --queue-depth=0` restored unlimited mode, and
  `/tmp/m07_post_rate_restore_vm6_raw_current.log` -> `OVERALL PASS`.

### Active Preservation Blocker Found And Closed

- During required preservation, TensorFlow failed after the M07 probes:
  `/tmp/m07_current_preserve_tensorflow.json` -> `overall_pass=False`.
- Earliest failing checkpoint:
  TensorFlow `TensorAssignOp + scalar_const_op` three-parameter
  `EigenMetaKernel` launched twice. The first launch succeeded; the second
  launch hit `CUDA_ERROR_ILLEGAL_ADDRESS`, which triggered primary-context
  recovery and caused follow-on `CUDA_ERROR_CONTEXT_IS_DESTROYED`.
- Cause:
  the previous guest-shim TensorFlow heuristic used a raw `8+8+8` layout for
  this kernel shape. The actual compact ABI is
  `(device pointer, scalar float, element count)` -> `8+4+4` bytes.
- Fix:
  `guest-shim/libvgpu_cuda.c` changed the three-parameter
  `TensorAssignOp + scalar_const_op` layout to offsets `0/8/12`, sizes
  `8/4/4`, total `16`.
- Live guest artifact after fix:
  VM-10 `/opt/vgpu/lib/libvgpu-cuda.so.1`
  `fea57a74687e289ebd9b62f4716e7ff3abe255ff77579a28c5ce622c3f51998f`.
- Closure proof:
  `/tmp/m07_current_preserve_tensorflow_after_3param_fix.json` ->
  `overall_pass=True`, `used_gpu_for_training=True`, `train_elapsed_sec=100.731`.

### Final Preservation After Last Shim Change

After the `8+4+4` TensorFlow fix and clean mediator restart:

- Plan A:
  `/tmp/m07_final_after_tf_3param_planA.json` -> `overall_pass=True`.
- Milestone 01 raw CUDA:
  `/tmp/m07_final_after_tf_3param_m01.json` -> `overall_pass=True`,
  3/3 Driver and 3/3 Runtime repetitions.
- Milestone 03 async/mixed stream-event:
  `/tmp/m07_final_after_tf_3param_m03_async.log` ->
  `ASYNC_STREAM_EVENT_PROBE PASS bytes=4194304`.
- Milestone 04 PyTorch:
  `/tmp/m07_final_after_tf_3param_pytorch.json` -> `overall_pass=True`.
- Milestone 05 CuPy:
  `/tmp/m07_final_after_tf_3param_cupy.json` -> `overall_pass=True`.
- Milestone 05 TensorFlow:
  `/tmp/m07_current_preserve_tensorflow_after_3param_fix.json` ->
  `overall_pass=True`, `used_gpu_for_training=True`.
- Milestone 06 two-process CuPy:
  `/tmp/m07_final_after_tf_3param_m06_two_cupy/child_1.log` and `child_2.log`
  both reported `overall_pass=True`.
- Milestone 07 malformed socket final:
  `/tmp/m07_final_after_tf_3param_malformed.json` -> `overall_pass=True`;
  mediator PID `1846882` remained alive and
  `grep -c "Recovering primary context" /tmp/mediator.log` returned `0`.

### Refreshed Closure

M07 is closed for the defined gate after the refresh:

- malformed mediator socket/CUDA-header inputs fail closed;
- mediator remains alive and logs the offending socket/VM;
- VM-6 quarantine and rate-limit rejection paths fail closed and recover;
- VM-10 remains healthy during targeted VM-6 quarantine;
- BAR `0666` exposure remains an explicit engineering trust assumption and
  production-hardening candidate;
- final preservation after the last guest-shim change is green for Plan A,
  raw CUDA, memory/sync, PyTorch, CuPy, TensorFlow, two-process CuPy, and M07
  malformed input.

## M07 Follow-Up: Optional Group BAR Policy

The next M07 hardening step has been prepared but not live-deployed:

- File changed: `guest-shim/install.sh`.
- New compatibility default:
  `VGPU_BAR_ACCESS_MODE=world` keeps existing behavior and sets
  `resource0`/`resource1` to `0666`.
- New hardening mode:
  `VGPU_BAR_ACCESS_MODE=group` uses `VGPU_BAR_GROUP` (default `vgpu`) and sets
  `resource0`/`resource1` to `root:${VGPU_BAR_GROUP}` with mode `0660`.
- Optional user enrollment:
  `VGPU_BAR_USERS="ollama test-10"` adds selected users to the BAR access group
  if they exist.
- The check-time permission application, generated udev rules, and generated
  `vgpu-devices.service` now all honor the selected BAR access mode.
- Local syntax proof:
  `bash -n guest-shim/install.sh` passed.
- Deployment status:
  not enabled on VM-10 or VM-6. This avoids accidentally regressing the
  preserved baseline before a dedicated reversible test.

Required proof before adopting group mode:

- choose a non-baseline VM or a maintenance window;
- install with `VGPU_BAR_ACCESS_MODE=group` and correct `VGPU_BAR_USERS`;
- prove `resource0`/`resource1` are `0660` and group-owned;
- prove unauthorized guest user cannot open BAR resources;
- prove authorized CUDA/Ollama user still passes a raw CUDA or CuPy gate;
- rerun Plan A if VM-10 is ever changed.

### Test-4 Group BAR Pilot

- Non-baseline target selected: Test-4, UUID
  `ba77526f-8955-acf8-2616-76f56a81ae8a`, mediator `vm_id=9`.
- Reason for target: VM-10 and VM-6 are preserved baselines; Test-4 was halted,
  already configured with `-device vgpu-cuda,pool_id=A,priority=high,vm_id=9`,
  and reachable at `test-4@10.25.33.12` after start.
- Applied live on Test-4 only:
  - created/used group `vgpu`;
  - added `test-4` and `ollama` to `vgpu`;
  - set `/sys/bus/pci/devices/0000:00:05.0/resource0` to `root:vgpu 0660`;
  - set `/sys/bus/pci/devices/0000:00:05.0/resource1` to `root:vgpu 0660`.
- Unauthorized proof:
  `sudo -u nobody` could not open either BAR file:
  both returned `PermissionError: [Errno 13] Permission denied`.
- Authorized proof:
  - `test-4` group membership included `vgpu`;
  - `ollama` group membership included `vgpu`;
  - `sudo -u ollama` could open both BAR files;
  - `test-4` could open both BAR files.
- CUDA transport proof:
  a Test-4 Driver API probe under group BAR mode reached the mediator as
  `vm_id=9`, connected over BAR1, completed device discovery, context creation,
  device allocation, HtoD copy, DtoH copy, stream creation, and event creation.
  The probe's full kernel section failed on Test-4's older CUDA/kernel baseline,
  so this pilot is recorded as BAR permission/transport proof, not a full raw
  CUDA milestone pass for Test-4.
- Baseline non-poisoning proof after the Test-4 pilot:
  Test-10 CuPy `/tmp/m07_post_test4_group_policy_test10_cupy.json` ->
  `overall_pass=True`.
- Mediator health after the Test-4 pilot:
  PID `950225` remained alive with three sockets
  (`root-2`, `root-6`, `root-7`), `sync FAILED=0`,
  `CUDA_ERROR_ILLEGAL_ADDRESS=0`, and `invalid handle=0`.
