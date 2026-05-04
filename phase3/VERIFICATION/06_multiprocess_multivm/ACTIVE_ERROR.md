# Active Error - Milestone 06 Multi-Process And Multi-VM

## Current Lane

Milestone 06: Multi-Process And Multi-VM

## Current Plan A State

Pass at entry from the preserved M05 entry chain:

- `/tmp/phase1_milestone_gate_serial_00_after_m04_64k_final.json` ->
  `overall_pass=True`.

## Active Error

None after the expanded same-VM, true multi-VM, framework-on-both-VMs,
cross-VM failure, Ollama-plus-second-VM, fairness-observability, and final
preservation probes.

Evidence:

- Two concurrent CuPy M05 probes passed:
  `/tmp/m06_two_process_cupy_summary.json` -> `overall_pass=True`, both child
  reports `overall_pass=True`.
- A killed CUDA allocation process did not poison concurrent or follow-up work:
  `/tmp/m06_failed_process_recovery` summary -> `overall_pass=True`,
  killed process return code `-9`, concurrent CuPy pass, follow-up CuPy pass.
- Mixed PyTorch plus CuPy same-VM concurrency passed:
  `/tmp/m06_mixed_pytorch_cupy_summary.json` -> `overall_pass=True`, both child
  reports `overall_pass=True`.
- Reused Test-6 as the second mediated VM:
  `test-6@10.25.33.16`, vGPU args
  `-device vgpu-cuda,pool_id=B,priority=low,vm_id=6`, guest PCI device
  `00:05.0 10de:2331`.
- Test-6 mediated allocation smoke passed:
  `cuInit`, context, `cuMemAlloc_v2`, and `cuMemFree_v2` all returned success
  while transport connected as `vm_id=6`.
- True multi-VM concurrency passed:
  `/tmp/m06_true_multivm/summary.json` -> `overall_pass=True`, with Test-10
  CuPy pass and Test-6 allocation/free pass running concurrently.
- Test-6 CuPy framework readiness passed after installing the missing NVRTC
  runtime package:
  `/tmp/m06_test6_cupy_after_nvrtc.json` -> `overall_pass=True`.
- True multi-VM framework gates passed:
  `/tmp/m06_multivm_framework/summary.json` -> `overall_pass=True` for
  Test-10 CuPy plus Test-6 CuPy, and
  `/tmp/m06_multivm_mixed_framework/summary.json` -> `overall_pass=True` for
  Test-10 PyTorch plus Test-6 CuPy.
- Cross-VM failure isolation passed:
  `/tmp/m06_crossvm_failure/summary.json` -> `overall_pass=True`.
- Ollama plus second-VM framework concurrency passed:
  `/tmp/m06_ollama_test6_concurrency/summary.json` -> `overall_pass=True`.
- Fairness/ownership delta passed:
  `/tmp/m06_fairness_delta/summary.json` -> `overall_pass=True`, with mediator
  slice counts for both `vm_id=6` and `vm_id=10`, clean per-VM cleanup, and no
  sync/illegal/unsupported/invalid-handle errors.
- Final preservation passed:
  Phase 1 Plan A/B/C, Milestone 01, Milestone 02, Milestone 03, Milestone 04,
  Milestone 05, and Test-6 CuPy all remained green.
- Mediator health after these probes:
  `sync FAILED=0`, `CUDA_ERROR_ILLEGAL_ADDRESS=0`,
  `Unsupported CUDA protocol call=0`, `invalid handle=0`.

## Candidate Queue

- No active M06 blocker remains.
- Residual candidate for later milestones: the mediator exposes pool counters
  and per-VM ownership, but it does not emit per-request priority labels in the
  current log format. M06 records priority observability through VM
  configuration and Pool A/Pool B counters; stronger scheduling-policy
  enforcement belongs in a later QoS-specific milestone if required.

## Last Proven Checkpoint

M06 closure checkpoint: one host mediator serving Test-10 (`vm_id=10`) and
Test-6 (`vm_id=6`) concurrently, with same-VM CuPy/CuPy, same-VM
PyTorch/CuPy, same-VM killed-process recovery, true multi-VM CuPy/CuPy,
true multi-VM PyTorch/CuPy, cross-VM killed-process recovery, Ollama Plan A
plus Test-6 CuPy, fairness/ownership logging, and final serial preservation all
passing.

## Closure Condition

Close Milestone 06 only after bounded concurrency gates pass, ownership/health
evidence is recorded, and the relevant prior single-process preservation gates
remain green.

Status: met.
