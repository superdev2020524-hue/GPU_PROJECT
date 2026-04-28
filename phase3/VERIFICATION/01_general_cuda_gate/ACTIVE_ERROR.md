# Active Error - Milestone 01 General CUDA Gate

## Current Lane

Milestone 01: General CUDA Compatibility Gate

## Current Plan A State

Pass.

Evidence:

- Before Milestone 01 implementation:
  `/tmp/phase1_milestone_gate_before_m01.json` -> `overall_pass=True`.
- After guest-shim fix:
  `/tmp/phase1_milestone_gate_after_m01_e1_fix.json` -> `overall_pass=True`.
- After Runtime API expansion and deployment correction:
  `/tmp/phase1_milestone_gate_after_runtime_restore_attempt.json` -> `overall_pass=True`.
- After hidden-risk sweep:
  `/tmp/phase1_milestone_gate_after_m01_hidden_risk_sweep.json` -> `overall_pass=True`.

## Active Error

None.

Closed: `M01-E1` generic raw CUDA kernel launch parameter serialization
failure.

Closed: `M01-E2` Runtime API expansion deployment regression.

Closed: `M01-H1` second-kernel expectation mismatch in the test probe.

The first raw Driver API probe originally reached device discovery, context
setup, allocation, HtoD, DtoH, stream creation, event creation, module load, and
function lookup, then failed at `cuLaunchKernel`. The fallback fix closed that
failure.

## Candidate List

- Residual non-terminating `0x00bc` / `cuFuncGetParamInfo` status noise from the
  preserved Ollama baseline. In Milestone 01 it still appears when the generic
  `add_one` function returns `cuFuncGetParamInfo -> 801`, but it is no longer
  correctness-breaking because the fallback path launches successfully.
- Shared-memory registration still falls back to BAR1 (`pfn_hidden`) during this
  small gate. Candidate-only for Milestone 01 because HtoD and DtoH round-trip
  passed and the gate completed successfully.
- Ollama CUDA runtime file restoration used the existing CUDA 13 runtime as a
  local recovery source after the CUDA 12 runtime path was accidentally
  overwritten during Runtime shim deployment. Candidate-only for follow-up
  hardening because Plan A recovered and the original bad overwritten file was
  preserved as `.bad-vgpu-cudart-20260427-1751`.
- `/usr/lib64/libcudart.so.12` still resolves to the vGPU Runtime shim. This is
  acceptable for standalone probe coverage but remains a deployment-scope
  candidate because Ollama must keep its real CUDA Runtime path separate.

## Closure Condition

`M01-E1` closes when the raw Driver API probe launches `add_one` successfully,
copies the incremented buffer back, verifies all values, and the host evidence
shows `cuLaunchKernel SUCCESS` for the generic non-Ollama kernel without
regressing Plan A.

Closure achieved:

- Raw CUDA gate report: `/tmp/phase3_general_cuda_gate_after_m01_e1_fix.json`
  -> `overall_pass=True`.
- The gate ran the probe twice; both runs passed.
- VM output includes `CASE kernel OP cuLaunchKernel RC 0`,
  `KERNEL_RESULT_OK n=64`, and `OVERALL PASS`.
- Host evidence includes two generic non-Ollama kernel successes:
  `cuLaunchKernel SUCCESS ... name=add_one ... vm=10`.
- Plan A recheck report:
  `/tmp/phase1_milestone_gate_after_m01_e1_fix.json` -> `overall_pass=True`.

`M01-E2` closure achieved:

- Failure report:
  `/tmp/phase1_milestone_gate_after_m01_runtime_expansion.json` ->
  `overall_pass=False`.
- Cause: Runtime shim deployment copied through Ollama's
  `/usr/local/lib/ollama/cuda_v12/libcudart.so.12` symlink and overwrote
  `libcudart.so.12.8.90` with the vGPU Runtime shim.
- Recovery: preserved the overwritten file as
  `libcudart.so.12.8.90.bad-vgpu-cudart-20260427-1751`, restored the runtime
  file from the existing CUDA 13 runtime copy, restarted `ollama`, and reran
  Plan A.
- Recovery proof:
  `/tmp/phase1_milestone_gate_after_runtime_restore_attempt.json` ->
  `overall_pass=True`.
- Expanded Milestone 01 proof after recovery:
  `/tmp/phase3_general_cuda_gate_runtime_expansion_after_planA_recovery.json`
  -> `overall_pass=True`.

`M01-H1` closure achieved:

- Hidden-risk sweep report:
  `/tmp/phase3_general_cuda_gate_hidden_risk_sweep.json` ->
  `overall_pass=False`.
- Cause: test expectation did not account for the first `add_one` kernel
  mutating the source device buffer before the second `scale_add` kernel.
- Fix: corrected expected value to `(input[i] + 1) * scale + i`.
- Closure proof:
  `/tmp/phase3_general_cuda_gate_hidden_risk_sweep_rerun1.json` ->
  `overall_pass=True`, with Driver API probe 5/5 pass and Runtime API probe
  5/5 pass.

## Promotion Rule

The first failing Milestone 01 gate case becomes active only after the exact
bounded repro, host evidence, VM evidence, and failure signature are recorded.

Do not promote framework failures here. PyTorch and TensorFlow belong to later
milestones.

## Last Proven Checkpoint

Current: Plan A preservation gate passed before Milestone 01 implementation.

Current Milestone 01 checkpoint:
`Driver API device discovery -> context -> alloc/free -> HtoD -> DtoH -> stream -> event -> module load -> function lookup -> add_one kernel -> scale_add kernel -> DtoH verify -> cleanup`, plus `Runtime API device discovery -> cudaMalloc/cudaFree -> cudaMemcpy -> cudaMemcpyAsync -> stream/event sync -> cudaDeviceSynchronize`, repeated five times per probe.

Initial failure evidence:

- VM report: `/tmp/phase3_general_cuda_gate_report_rerun1.json`.
- VM stdout: `CASE kernel OP cuLaunchKernel RC 1`, followed by
  `KERNEL_MISMATCH idx=0 got=0 expected=1`.
- VM stderr: `STATUS_ERROR: call=cuFuncGetParamInfo(0x00bc) ... host_status=0x00000321`,
  then `STATUS_ERROR: call=cuLaunchKernel(0x0050) ... host_status=0x00000001`.
- Host: `cuLaunchKernel launch FAILED: rc=1 ... name=add_one ... params=0 vm=10`.
- Host: `call FAILED: vm=10 call=cuLaunchKernel(0x0050) rc=1(CUDA_ERROR_INVALID_VALUE)`.

## Next Single Step

Promote the current expanded hidden-risk gate to Milestone 01 closure if the
user accepts this coverage as the endpoint, or continue with a broader API audit
under Milestone 02.
