# Gate - Milestone 01 General CUDA Compatibility

## Gate Name

`phase3_general_cuda_gate`

## Purpose

Prove basic non-Ollama CUDA behavior through the mediated vGPU path before
moving to PyTorch, TensorFlow, or other large frameworks.

## Gate Strategy

Use small standalone CUDA programs and one runner that produces JSON. Each case
must be small enough to fail quickly and clearly.

The first version should avoid framework dependencies and cover only core CUDA
behavior.

## Required Cases

1. Driver API device discovery
   - `cuDeviceGetCount`
   - `cuDeviceGetName`
   - `cuDeviceTotalMem_v2`
2. Runtime API device discovery
   - `cudaGetDeviceCount`
   - `cudaGetDeviceProperties`
3. Driver and Runtime API memory allocation and free
   - `cuMemAlloc` / `cuMemFree`
   - `cudaMalloc`
   - `cudaFree`
4. Host-to-device copy
   - known input buffer copied to device
5. Device-to-host copy
   - device result copied back and compared
6. Simple Driver API kernel launch
   - vector add or increment kernel
7. Stream synchronization
   - create stream, async copy, launch, synchronize
8. Event synchronization
   - record event and wait
9. Repeated process cleanup
   - run the same small binary multiple times and verify no stale state

## Current Gate Files

- `tests/general_cuda_gate/driver_api_probe.c`
- `tests/general_cuda_gate/runtime_api_probe.c`
- `tests/general_cuda_gate/run_general_cuda_gate.py`

The current runner deploys both probes to the VM, builds them with `gcc`, runs
each probe twice, and stores one JSON report.

## Output

The runner must write JSON with:

- gate name;
- timestamp;
- VM target;
- each case name;
- command;
- exit code;
- stdout preview;
- stderr preview;
- pass/fail;
- host evidence summary;
- VM evidence summary;
- overall pass.

Default output path:

`/tmp/phase3_general_cuda_gate_report.json`

Latest passing expanded report:

`/tmp/phase3_general_cuda_gate_runtime_expansion_after_planA_recovery.json`

Latest passing hidden-risk sweep:

`/tmp/phase3_general_cuda_gate_hidden_risk_sweep_rerun1.json`

Latest Plan A preservation proof after hidden-risk sweep:

`/tmp/phase1_milestone_gate_after_m01_hidden_risk_sweep.json`

## Pass Criteria

- all gate cases return success;
- copied data matches expected values;
- host mediator shows physical GPU execution for the session;
- no active stale payload, stale status, sync failure, or illegal-address
  signature appears;
- repeated runs do not poison later runs.

## Hidden-Risk Sweep Before Closure

Milestone 01 must not close from a single happy path. Before closure, run and
record these additional checks:

- path isolation: verify vGPU shims remain under `/opt/vgpu/lib` and do not
  overwrite Ollama's real CUDA runtime files;
- exported-symbol coverage: verify all symbols used by the gate are present in
  the deployed Driver and Runtime shims;
- second kernel shape: run another simple Driver API kernel with a different
  parameter layout and write pattern;
- repeated process cleanup: run both probes more than twice in a clean sequence;
- preservation: rerun Plan A after any shim or deployment change;
- candidate discipline: keep residual `0x00bc` and BAR1 fallback as candidates
  unless they become correctness-breaking.

## Fail Criteria

- any gate binary exits non-zero;
- output data mismatch;
- host mediator shows a correctness-breaking CUDA failure;
- VM shows a new `STATUS_ERROR` that terminates the test;
- repeated process run fails after an earlier pass.

## Baseline Recheck Rule

If implementing this gate changes guest shim, host mediator, CUDA executor,
transport, service config, or deployment layout, rerun `Plan A` after the change.
