# Evidence - Milestone 06 Multi-Process And Multi-VM

## Baseline Evidence

Milestone 06 begins after Milestone 05 closure:

- M03 preservation after M05:
  `/tmp/m03_preservation_after_m05_300/run_1.txt` through `run_3.txt` ->
  3/3 pass.
- M04 PyTorch preservation after M05:
  `/tmp/m04_pytorch_preservation_after_m05/run_1.json` through `run_3.json` ->
  3/3 pass.
- M05 CuPy preservation:
  `/tmp/m05_cupy_preservation_after_m05/run_1.json` through `run_3.json` ->
  3/3 pass.

## Evidence To Collect

- Orchestrator report for two concurrent same-VM framework processes.
- Per-child stdout/stderr and parsed JSON report.
- Follow-up known-good single-process probe after concurrency.
- Mediator health and ownership evidence after concurrent workload.
- Explicit classification of any timeout or failed child as the active M06 error.

## Current Session Findings

Milestone 06 is active. The first bounded action is a two-process CuPy concurrency
probe in Test-10 using the isolated `/mnt/m04-pytorch/venv` environment.

## Initial Same-VM Concurrency Evidence

- Probe script:
  `phase3/tests/multiprocess_multivm/two_process_cupy_probe.py`, deployed to
  `/mnt/m04-pytorch/two_process_cupy_probe.py`.
- Two-process CuPy probe:
  `/tmp/m06_two_process_cupy_summary.json` -> `overall_pass=True`.
- Child reports:
  - `/tmp/m06_two_process_cupy/child_1.json` -> `overall_pass=True`.
  - `/tmp/m06_two_process_cupy/child_2.json` -> `overall_pass=True`.
- Both children independently passed CuPy import, device count/name, HtoD/DtoH,
  elementwise add, matmul, and repeated warm execution value checks.
- Follow-up single-process CuPy probe:
  `/tmp/m06_followup_single_cupy.json` -> `overall_pass=True`.

## Failed-Process Recovery Evidence

- A CUDA allocation process from
  `phase3/tests/memory_sync_cleanup/forced_kill_alloc_probe.c` reached `READY`,
  then was killed with `SIGKILL`.
- Concurrent CuPy during the killed allocation process:
  `/tmp/m06_failed_process_recovery/concurrent_cupy.json` ->
  `overall_pass=True`, elapsed `5.681s`.
- Follow-up CuPy after the kill:
  `/tmp/m06_failed_process_recovery/followup_cupy.json` ->
  `overall_pass=True`, elapsed `5.321s`.
- Summary:
  `/tmp/m06_failed_process_recovery/summary.json` ->
  `overall_pass=True`, killed process return code `-9`.

## Mediator Health Evidence

After the initial M06 probes, dom0 `/tmp/mediator.log` reported:

- `sync FAILED: 0`.
- `CUDA_ERROR_ILLEGAL_ADDRESS: 0`.
- `Unsupported CUDA protocol call: 0`.
- `invalid handle: 0`.
- `CUDA process cleanup: 17`.
- `Cleaned up VM: 16`.
- `vm_id=10: 2181`.
- `cuLaunchKernel SUCCESS: 107`.
- `WFQ queue depth: 0`, `CUDA busy: no`.

Current interpretation: no M06 active error is promoted from the initial
same-VM CuPy concurrency or killed-process recovery probes.

## Mixed Framework Concurrency Evidence

- Probe script:
  `phase3/tests/multiprocess_multivm/mixed_pytorch_cupy_probe.py`, deployed to
  `/mnt/m04-pytorch/mixed_pytorch_cupy_probe.py`.
- Mixed PyTorch plus CuPy same-VM probe:
  `/tmp/m06_mixed_pytorch_cupy_summary.json` -> `overall_pass=True`.
- Child reports:
  - `/tmp/m06_mixed_pytorch_cupy/pytorch.json` -> `overall_pass=True`.
  - `/tmp/m06_mixed_pytorch_cupy/cupy.json` -> `overall_pass=True`.
- Elapsed times:
  - PyTorch child completed within the mixed probe bound.
  - CuPy child elapsed `73.610s` while PyTorch was active, and still passed all
    value checks.

## Post-Mixed Mediator Health Evidence

After the mixed PyTorch/CuPy probe, dom0 `/tmp/mediator.log` reported:

- `sync FAILED: 0`.
- `CUDA_ERROR_ILLEGAL_ADDRESS: 0`.
- `Unsupported CUDA protocol call: 0`.
- `invalid handle: 0`.
- `CUDA process cleanup: 19`.
- `Cleaned up VM: 18`.
- `vm_id=10: 2422`.
- `cuLaunchKernel SUCCESS: 121`.
- `WFQ queue depth: 0`, `CUDA busy: no`.

Current interpretation: same-VM framework concurrency is green for the scoped
CuPy+CuPy, killed-process recovery, and PyTorch+CuPy probes. Priority/fairness,
Ollama plus framework concurrency, and multi-VM behavior remain open M06 scope.

## Existing VM Inventory And Selection

- VM creation instructions exist under `vm_create/`, including
  `create_vm.sh`, `create_test3_vm.sh`, and `post_install_vm.sh`.
- Those scripts map `Test-N` to static IP `10.25.33.(10+N)`, but the historical
  session log records ISO SR/network reliability problems during fresh VM
  creation.
- Host inventory showed Test-5 and Test-8 were the cleanest halted candidates
  initially because both retained 4 vCPUs and a bootable disk.
- Test-8 started but did not answer on expected IP `10.25.33.18`.
- Test-5 started and answered ping on `10.25.33.15`, but SSH port 22 refused
  connections.
- Test-6 was selected after restoring `VCPUs-at-startup=4`; it answered on
  `10.25.33.16` and SSH succeeded as `test-6` with the known password using an
  isolated known-hosts file.

## Test-6 Second Mediated VM Bring-Up

- Test-6 host UUID:
  `eb93375a-b277-cf0d-12e7-c0e7ccfcdd7f`.
- Test-6 vGPU args:
  `-device vgpu-cuda,pool_id=B,priority=low,vm_id=6`.
- Guest proof:
  `lspci -nn` shows `00:05.0 3D controller [0302]: NVIDIA Corporation Device
  [10de:2331] (rev a1)`.
- Guest shim install method:
  copied the known-good `/opt/vgpu/lib` shim set from Test-10 into Test-6 and
  set BAR resource permissions for the vGPU PCI device.
- Test-6 live shim hashes:
  - `/opt/vgpu/lib/libvgpu-cuda.so.1`
    `51c47c7104a45d62773ff11481af0e33fb92fc2c7a1795bc29c6d71c6eb3a1ba`.
  - `/opt/vgpu/lib/libvgpu-cudart.so`
    `364832a3aad55ffe297a0b96858d45d6fa3e51fd1badbeef04382f24e04611a4`.
  - `/opt/vgpu/lib/libvgpu-cublas.so.12`
    `5202e665bac786aa648b44ce1e3ee68ea48ecbaa5fd7e6425a68f428ef160f73`.
- Test-6 mediated allocation/free smoke:
  `cuInit`, `cuDeviceGet`, `cuDevicePrimaryCtxRetain`, `cuCtxSetCurrent`,
  `cuMemAlloc_v2`, and `cuMemFree_v2` all returned `0`.
- Test-6 transport proof:
  guest log reported `Connected (vm_id=6) data_path=BAR1
  status_from=BAR1-status-mirror`.

## Single Mediator Multi-VM Evidence

- Dom0 mediator PID:
  `830120 ./mediator_phase3`.
- The same mediator reported two live server sockets:
  - `/var/xen/qemu/root-2/tmp/vgpu-mediator.sock` for Test-10.
  - `/var/xen/qemu/root-6/tmp/vgpu-mediator.sock` for Test-6.
- Host mediator log for Test-6:
  `CUDA result sent vm_id=6` for `cuInit`, `cuDeviceGet`, `cuMemAlloc`, and
  `cuMemFree`, plus `CUDA process cleanup vm_id=6`.
- Host mediator stats after Test-6 smoke included Pool A and Pool B traffic:
  `Pool A processed: 2402`, `Pool B processed: 5`.

## True Multi-VM Concurrency Evidence

- Orchestrator report:
  `/tmp/m06_true_multivm/summary.json` -> `overall_pass=True`.
- Test-10 child:
  `/tmp/m06_true_multivm/test10_cupy.out` -> CuPy gate `overall_pass=True`,
  elapsed `11.631s`.
- Test-6 child:
  `/tmp/m06_true_multivm/test6_alloc.out` -> mediated allocation/free
  `overall_pass=True`, elapsed `11.630s`.
- Post-run mediator health:
  `vm_id=6: 13`, `vm_id=10: 2540`, `root-6: 26`, `root-2: 453`,
  `CUDA process cleanup vm_id=6: 2`, `CUDA process cleanup vm_id=10: 20`,
  `sync FAILED: 0`, `CUDA_ERROR_ILLEGAL_ADDRESS: 0`,
  `Unsupported CUDA protocol call: 0`, `invalid handle: 0`,
  `cuLaunchKernel SUCCESS: 134`.

Current interpretation: the single-mediator multi-VM design is now proven at the
initial bounded level. Full M06 closure still requires deciding how far to push
Ollama/framework mix, priority/fairness observability, and heavier multi-VM
workload coverage.

## Expanded Gate Evidence

The expanded M06 gate was broadened after the user clarified that no foreseeable
M06 verification should be skipped before closure.

### Test-6 Framework Readiness

- Test-6 initially had Python but no `pip`, no `ensurepip`, and no framework
  runtime.
- `unattended-upgrade` on Test-6 held the package-management lock while
  consuming CPU for more than 16 minutes; it was stopped after it proved stuck,
  then `dpkg --configure -a` and `apt-get -f install -y` completed cleanly.
- Installed Test-6 packaging/build prerequisites with `apt-get install -y
  python3-pip python3-venv`.
- Created isolated framework environment:
  `/opt/m06-cupy/venv`.
- Installed:
  `cupy-cuda12x==14.0.1`, `nvidia-cuda-nvrtc-cu12==12.1.105`, and
  `nvidia-cuda-runtime-cu12==12.1.105`.
- Corrected Test-6 Runtime shim symlink:
  `/opt/vgpu/lib/libcudart.so.12 -> /opt/vgpu/lib/libvgpu-cudart.so`.
- First Test-6 CuPy attempt failed only after import/transfer because NVRTC was
  missing:
  `/tmp/m06_test6_cupy_probe.json` -> `overall_pass=False`,
  `DynamicLibNotFoundError: libnvrtc.so`.
- After installing NVRTC/runtime:
  `/tmp/m06_test6_cupy_after_nvrtc.json` -> `overall_pass=True`, including
  import, device count/name, HtoD/DtoH, elementwise add, matmul, and repeated
  warm execution.

### True Multi-VM Framework Gates

- Test-10 CuPy plus Test-6 CuPy:
  `/tmp/m06_multivm_framework/summary.json` -> `overall_pass=True`.
  - Test-10 child elapsed `9.690s`, `overall_pass=True`.
  - Test-6 child elapsed `14.883s`, `overall_pass=True`.
- Test-10 PyTorch plus Test-6 CuPy:
  `/tmp/m06_multivm_mixed_framework/summary.json` -> `overall_pass=True`.
  - Test-10 PyTorch child elapsed `84.175s`, `overall_pass=True`.
  - Test-6 CuPy child elapsed `84.174s`, `overall_pass=True`.

### Cross-VM Failure Isolation

- Test-6 allocation process from
  `phase3/tests/memory_sync_cleanup/forced_kill_alloc_probe.c` was compiled on
  Test-6 and run through the mediated CUDA path.
- It reached `READY` and was killed while Test-10 ran a CuPy gate.
- Summary:
  `/tmp/m06_crossvm_failure/summary.json` -> `overall_pass=True`.
- Test-10 CuPy during Test-6 kill:
  `/tmp/m06_crossvm_failure/test10_cupy_during_test6_kill.out` ->
  `overall_pass=True`.
- Follow-up Test-10 CuPy:
  `/tmp/m06_crossvm_failure/test10_cupy_followup.out` -> `overall_pass=True`.
- Follow-up Test-6 CuPy:
  `/tmp/m06_crossvm_failure/test6_cupy_followup.out` -> `overall_pass=True`.

### Ollama Plus Second-VM Framework Concurrency

- Test-10 Plan A gate ran while Test-6 executed seven consecutive CuPy gates.
- Summary:
  `/tmp/m06_ollama_test6_concurrency/summary.json` -> `overall_pass=True`.
- Plan A report:
  `/tmp/m06_planA_with_test6_cupy.json` -> `overall_pass=True`.
- Test-6 loop:
  `/tmp/m06_test6_cupy_loop_during_planA/run_1.json` through `run_7.json` ->
  all pass.

### Fairness And Ownership Observability

- Explicit fairness delta:
  `/tmp/m06_fairness_delta/summary.json` -> `overall_pass=True`.
- Workload: three Test-10 CuPy gates and three Test-6 CuPy gates ran
  concurrently.
- Mediator log slice for that bounded window:
  - `vm_id=6: 345`.
  - `vm_id=10: 426`.
  - `CUDA process cleanup vm_id=6: 3`.
  - `CUDA process cleanup vm_id=10: 3`.
  - `cuLaunchKernel SUCCESS: 78`.
  - `sync FAILED: 0`.
  - `CUDA_ERROR_ILLEGAL_ADDRESS: 0`.
  - `Unsupported CUDA protocol call: 0`.
  - `invalid handle: 0`.
- Final dom0 mediator health after all expanded probes:
  - `sync FAILED: 0`.
  - `CUDA_ERROR_ILLEGAL_ADDRESS: 0`.
  - `Unsupported CUDA protocol call: 0`.
  - `invalid handle: 0`.
  - `vm_id=6: 2395`.
  - `vm_id=10: 5587`.
  - `CUDA process cleanup vm_id=6: 24`.
  - `CUDA process cleanup vm_id=10: 47`.
  - `cuLaunchKernel SUCCESS: 538`.
  - Last stats: `Total processed: 7429`, `Pool A processed: 5059`,
    `Pool B processed: 2370`, `WFQ queue depth: 0`, `Context switches: 0`,
    `CUDA busy: no`.

### Final Preservation After M06 Expansion

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
  missing required matrix sections. Historical risk terms were classified audit
  evidence, not current failures.
- Milestone 03:
  `/tmp/m06_final_m03_async_preservation/summary.json` ->
  `overall_pass=True`, three 4 MiB async stream/event runs passed with elapsed
  times `139.469s`, `119.200s`, and `149.674s`.
- Milestone 04:
  `/tmp/m06_final_framework_preservation/summary.json` shows Test-10 PyTorch
  3/3 pass.
- Milestone 05:
  `/tmp/m06_final_framework_preservation/summary.json` shows Test-10 CuPy 3/3
  pass.
- Milestone 06 Test-6 framework tail:
  `/tmp/m06_final_test6_cupy_clean/summary.json` -> `overall_pass=True`,
  Test-6 CuPy 3/3 pass with stdout and debug logs separated.
- A wrapper parser issue initially misclassified two Test-6 CuPy preservation
  runs because vGPU debug logs appeared after the JSON object; direct reparse
  showed `overall_pass=True`, and the clean rerun confirmed 3/3 pass.

Current interpretation: expanded Milestone 06 is closed. No active M06 error
remains.
