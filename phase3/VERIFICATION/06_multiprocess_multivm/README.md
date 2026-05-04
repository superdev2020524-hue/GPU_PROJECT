# 06 - Multi-Process And Multi-VM

## Purpose

Move from single-workload success to virtualization behavior.

## Scope

- two processes in one VM;
- multiple VMs when available;
- priority scheduling;
- fairness policy;
- memory pressure;
- long-running plus short interactive workload;
- cancellation and cleanup;
- mediator health under load.

## Closure Criteria

- no cross-process data leakage;
- no stale status from one process affects another;
- priority policy is observable;
- one failed workload does not poison the mediator;
- metrics identify which VM/process owns work.

## Current Status

Milestone 06 is complete for the expanded multi-process and multi-VM gate.

Scope correction after TensorFlow test:

- This closure covers the surfaces actually tested: Ollama, raw CUDA, PyTorch,
  and CuPy across the documented single-VM and multi-VM concurrency cases.
- It does not cover TensorFlow concurrency or TensorFlow GPU registration.
- TensorFlow is blocked earlier in `05_second_framework_gate` as `M05-E5`.

Completed probes:

- Two concurrent CuPy framework probes passed with both children reporting
  `overall_pass=True`.
- A killed CUDA allocation process did not poison a concurrent or follow-up CuPy
  probe.
- Mixed PyTorch plus CuPy framework probes passed concurrently in the same VM.
- A second existing VM, Test-6, was recovered and attached to the same single
  host mediator as `vm_id=6`.
- True multi-VM concurrency passed: Test-10 ran the CuPy gate while Test-6 ran a
  mediated CUDA allocation/free smoke probe.
- Test-6 was expanded from raw CUDA smoke to full CuPy framework readiness.
- True multi-VM framework concurrency passed for Test-10 CuPy plus Test-6 CuPy.
- Mixed true multi-VM framework concurrency passed for Test-10 PyTorch plus
  Test-6 CuPy.
- A killed Test-6 CUDA process did not poison concurrent or follow-up Test-10
  and Test-6 CuPy work.
- Ollama Plan A passed while Test-6 ran seven consecutive CuPy gates.
- Pool/priority/fairness observability was recorded from mediator ownership
  logs and stats: Pool A and Pool B counters advanced, per-VM ownership was
  visible as `vm_id=10` and `vm_id=6`, and final queue depth was zero.
- Final preservation passed for Phase 1 Plan A/B/C, Milestone 01, Milestone 02,
  Milestone 03, Milestone 04, Milestone 05, and the new Test-6 CuPy gate.
