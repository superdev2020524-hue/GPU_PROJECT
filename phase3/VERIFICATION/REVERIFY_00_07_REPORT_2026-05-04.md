# Phase 3 Re-Verification Report: Milestones 00-07

Date: 2026-05-04

Result: **PASS**

This was a fresh preservation run from Milestone 00 through Milestone 07. I did
not treat old records as proof. Each gate below was run again on the live system,
and the result files are listed so the run can be checked later.

## Plain Summary

The current mediated GPU path is healthy for the verified scope:

- Ollama Plan A, Plan B, and Plan C all passed on VM-10.
- Raw CUDA Driver and Runtime API probes passed.
- The API coverage audit still matches the executor source: every protocol ID has
  a handler.
- Memory, stream, event, and post-kill recovery probes passed.
- PyTorch, CuPy, and TensorFlow all used the mediated GPU path successfully.
- Same-VM and cross-VM concurrency passed.
- Malformed mediator socket traffic was rejected safely, and both VMs still ran
  known-good CuPy probes afterward.

No active error is open for Milestones 00 through 07 after this run.

## Milestone Results

### 00 - Preserve Ollama Baseline

Status: **PASS**

Evidence:

- Plan A, `qwen2.5:0.5b`:
  `VM10:/tmp/20260504_reverify_m00_planA.json`
- Plan B, `tinyllama:latest`:
  `VM10:/tmp/20260504_reverify_m00_planB.json`
- Plan C, `qwen2.5:3b` CLI lane:
  `VM10:/tmp/20260504_reverify_m00_planC.json`

What this proves:

Ollama still answers the controlled prompts, meets the speed limits, pins models
when asked, and unloads models cleanly after the run.

### 01 - General CUDA Gate

Status: **PASS**

Evidence:

- `workstation:/tmp/20260504_reverify_m01_general_cuda.json`

What this proves:

The raw CUDA path still works below any framework. The runner completed 5 Driver
API repetitions and 5 Runtime API repetitions, including device discovery,
allocation/free, copies, streams, events, module loading, kernel launch, and
cleanup.

### 02 - API Coverage Audit

Status: **PASS**

Evidence:

- `workstation:/tmp/20260504_reverify_m02_api_coverage_source_audit.json`

Key result:

- Protocol IDs found: `87`
- Executor cases found: `87`
- Missing executor cases: `[]`

What this proves:

The checked-in CUDA protocol and the host executor are still in sync. The API
matrix and gap list are present, and Milestone 02 has no active error.

### 03 - Memory, Synchronization, And Cleanup

Status: **PASS**

Evidence:

- Async stream/event probe:
  `VM10:/tmp/20260504_reverify_m03_async_stream_event.log`
- Forced-kill child proof:
  `VM10:/tmp/20260504_reverify_m03_forced_kill_child.log`
- Post-kill recovery probe:
  `VM10:/tmp/20260504_reverify_m03_post_kill_async.log`

What this proves:

The 4 MiB async copy path, stream/event synchronization, DtoD copy, memset, and
data verification passed. A GPU-using process was then killed with `SIGKILL`, and
the next known-good memory/sync probe still passed. That means the killed process
did not poison the next run.

### 04 - PyTorch Gate

Status: **PASS**

Evidence:

- `VM10:/tmp/20260504_reverify_m04_pytorch.json`

What this proves:

PyTorch sees CUDA through the mediated path and completes value-checked tensor
copy, elementwise operation, matrix multiply, small neural-network inference, and
repeated warm execution.

### 05 - Second Framework Gate

Status: **PASS**

Evidence:

- CuPy:
  `VM10:/tmp/20260504_reverify_m05_cupy.json`
- TensorFlow:
  `VM10:/tmp/20260504_reverify_m05_tensorflow.json`

What this proves:

CuPy passed import, device discovery, HtoD/DtoH, elementwise add, matmul, and
repeated execution.

TensorFlow registered a logical GPU and completed the bounded GPU training probe
with `used_gpu_for_training=True`. This is the corrected TensorFlow lane that was
previously the main framework blocker.

### 06 - Multi-Process And Multi-VM

Status: **PASS**

Evidence:

- Same-VM CuPy/CuPy:
  `VM10:/tmp/20260504_reverify_m06_two_process_cupy.json`
- Same-VM PyTorch/CuPy:
  `VM10:/tmp/20260504_reverify_m06_mixed_pytorch_cupy.json`
- Test-6 CuPy single-VM readiness:
  `VM6:/tmp/20260504_reverify_m06_test6_cupy_single.json`
- Cross-VM concurrent CuPy:
  `workstation:/tmp/20260504_reverify_m06_cross_vm_cupy.json`

Operational note:

Test-6 was reachable, but its earlier `/mnt/m04-pytorch` framework directory was
not present in this boot. I rebuilt an isolated venv at
`/home/test-6/m06-cupy-venv` and used that for the Test-6 and cross-VM CuPy
checks. This was an environment refresh, not a runtime code change.

What this proves:

The mediator handled two framework processes in the same VM, mixed framework
load in the same VM, Test-6 CuPy work, and concurrent CuPy work from Test-10 and
Test-6.

### 07 - Security And Isolation

Status: **PASS**

Evidence:

- Malformed socket probe:
  `host:/tmp/20260504_reverify_m07_malformed_socket.json`
- Mediator health summary:
  `host:/tmp/20260504_reverify_m07_mediator_summary.json`
- Post-probe Test-10 known-good CuPy:
  `VM10:/tmp/20260504_reverify_m07_post_test10_cupy.json`
- Post-probe Test-6 known-good CuPy:
  `VM6:/tmp/20260504_reverify_m07_post_test6_cupy.json`

Key mediator health result:

- `sync FAILED`: `0`
- `CUDA_ERROR_ILLEGAL_ADDRESS`: `0`
- `Recovering primary context`: `0`

What this proves:

The mediator rejected malformed socket messages safely, stayed alive, logged the
bad inputs, and still served known-good CUDA framework work from both VMs
afterward.

## Final Judgment

Milestones 00 through 07 are verified on the live system for the documented gate
scope. The strongest evidence is that the run starts with Ollama preservation,
then moves through raw CUDA, API consistency, memory/sync behavior, PyTorch,
CuPy, TensorFlow, same-VM concurrency, cross-VM concurrency, malformed security
traffic, and finally post-security known-good framework checks.

The remaining caution is the same engineering caution already recorded in the
milestone files: this is not a claim that every CUDA application or every model
shape will work. It is a clean pass for the explicit gates that define
Milestones 00 through 07.
