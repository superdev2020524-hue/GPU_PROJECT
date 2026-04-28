# Active Error - Milestone 02 API Coverage Audit

## Current Lane

Milestone 02: API Coverage Audit

## Current Plan A State

Pass.

Evidence:

- Before `M02-E1` runtime changes:
  `/tmp/phase1_milestone_gate_before_m02_e1.json` -> `overall_pass=True`.
- After `M02-E1` deployment:
  `/tmp/phase1_milestone_gate_after_m02_e1_fail_closed.json` ->
  `overall_pass=True`.
- Before `M02-E2` cuBLAS change:
  `/tmp/phase1_milestone_gate_before_m02_e2.json` -> `overall_pass=True`.
- After `M02-E2` deployment:
  `/tmp/phase1_milestone_gate_after_m02_e2_cublas_fail_closed.json` ->
  `overall_pass=True`.
- Before `M02-E3` executor change:
  `/tmp/phase1_milestone_gate_before_m02_e3.json` -> `overall_pass=True`.
- After `M02-E3` final live-binary restart:
  `/tmp/phase1_milestone_gate_after_m02_e3_live_binary_restart.json` ->
  `overall_pass=True`.

## Active Error

None. Milestone 02 active errors are closed.

## Closed Error

`M02-E1`: silent-success API stubs in high-risk Runtime and cuBLASLt paths.

Closure achieved by fail-closed changes and live deployment:

- Runtime `cudaLaunchKernel` now returns `cudaErrorNotSupported`.
- Runtime graph APIs now return `cudaErrorNotSupported`.
- Runtime `cudaMallocManaged`, `cudaHostRegister`, `cudaHostUnregister`,
  peer-access enable/disable, 2D/3D/peer async copies, and graph capture now
  fail closed instead of reporting fake success.
- Runtime `cudaDeviceSynchronize` and `cudaStreamSynchronize` now preserve
  Driver synchronization failure.
- cuBLASLt descriptor/layout/preference/heuristic/matmul calls now return
  `CUBLAS_STATUS_NOT_SUPPORTED`.
- Live artifact proof:
  `/opt/vgpu/lib/libvgpu-cudart.so`
  `66b10c345acd084164b115df5fc7b9b8851fe18583610e7c3d12ac90f17149cc`,
  `/opt/vgpu/lib/libvgpu-cublasLt.so.12`
  `28907f065f14b2e00686e1620057588dbf008e4b0da7d1091278eea4841bd3da`.
- Regression proof:
  `/tmp/phase1_milestone_gate_after_m02_e1_fail_closed.json` ->
  `overall_pass=True`;
  `/tmp/phase3_general_cuda_gate_after_m02_e1_fail_closed.json` ->
  `overall_pass=True`.

`M02-E2`: cuBLAS stub-handle fallback could return success without host compute
if transport was unavailable.

Closure achieved by fail-closed cuBLAS behavior and live deployment:

- `cublasCreate_v2` no longer allocates a stub handle when there is no real
  transport/context. It returns `CUBLAS_STATUS_NOT_INITIALIZED` and writes
  `VGPU_CUBLAS_MODE=NO_TRANSPORT`.
- Historical stub-handle paths in `cublasSetStream_v2`, `cublasGetStream_v2`,
  `cublasSetMathMode`, `cublasGetMathMode`, `cublasSgemm_v2`,
  `cublasGemmEx`, `cublasGemmStridedBatchedEx`, and `cublasGemmBatchedEx`
  now return `CUBLAS_STATUS_NOT_INITIALIZED`.
- Live artifact proof:
  `/opt/vgpu/lib/libvgpu-cublas.so.12`
  `80659ffeb12467a8df36ff225aee5a22629eab947a6ad137abb33113f3773a5b`,
  size `113192`.
- Regression proof:
  `/tmp/phase1_milestone_gate_after_m02_e2_cublas_fail_closed.json` ->
  `overall_pass=True`;
  `/tmp/phase3_general_cuda_gate_after_m02_e2_cublas_fail_closed.json` ->
  `overall_pass=True`.

`M02-E3`: protocol IDs existed without explicit executor handling.

Closure achieved by explicit executor dispositions and live deployment:

- `src/cuda_executor.c` now names and explicitly fail-closes all protocol IDs
  that do not have real executor behavior.
- Final source-level comparison:
  `protocol_ids=86`, `executor_case_ids=86`, `missing_cases=[]`.
- Explicit unsupported protocol cases include context push/pop protocol IDs,
  protocol-only async DtoH/DtoD IDs, D16 memset, managed/host allocation,
  cooperative launch, texture objects, occupancy helpers, cuBLASLt protocol
  IDs, and error-name/string protocol IDs.
- Live artifact proof:
  `/root/phase3/mediator_phase3`
  `8f306df61150071553a5dc7c9b8cba257658111acf6a71331aaa2fb7ebebe796`;
  `/root/phase3/src/cuda_executor.c`
  `0f2b66f05b9c633592f44cdb4fe4b1596b63d2733eaedccd3af2518d35cfd21c`.
- Live process proof: mediator PID `295063` was running from
  `/root/phase3/mediator_phase3`, not a deleted pre-rebuild executable.
- Regression proof:
  `/tmp/phase1_milestone_gate_after_m02_e3_live_binary_restart.json` ->
  `overall_pass=True`;
  `/tmp/phase3_general_cuda_gate_after_m02_e3_live_binary_restart.json` ->
  `overall_pass=True`.

## Candidate List

- Residual `cuFuncGetParamInfo(0x00bc)` unsupported/invalid-value noise.
- BAR1 fallback after shmem GPA resolution fails with `pfn_hidden`.
- Runtime shim deployment scope: `/usr/lib64/libcudart.so.12` points to vGPU
  Runtime shim.
- Remaining lower-priority silent/synthetic behavior in Runtime status queries,
  Driver optional exports, and NVML.
- Ollama-shaped implementations that may not generalize to PyTorch/TensorFlow.

## Last Proven Checkpoint

Milestone 01 hidden-risk gate passed:

`Driver API two-kernel path + Runtime API copy/stream/event path + five process
repetitions + Plan A preservation`.

After `M02-E1`, Milestone 01 raw CUDA gate also passed:

`/tmp/phase3_general_cuda_gate_after_m02_e1_fail_closed.json` ->
`overall_pass=True`.

After `M02-E2`, Milestone 01 raw CUDA gate also passed:

`/tmp/phase3_general_cuda_gate_after_m02_e2_cublas_fail_closed.json` ->
`overall_pass=True`.

After `M02-E3`, Milestone 01 raw CUDA gate also passed:

`/tmp/phase3_general_cuda_gate_after_m02_e3_live_binary_restart.json` ->
`overall_pass=True`.

## Closure Condition

Milestone 02 closes when:

- `API_COVERAGE_MATRIX.md` exists and classifies the audited APIs;
- `GAP_LIST.md` prioritizes the gaps for later milestones;
- `M02-E1` has a disposition: closed for the highest-risk paths listed above;
- `M02-E2` has a disposition: closed by fail-closed cuBLAS behavior;
- `M02-E3` has a disposition for protocol IDs without executor cases;
- Plan A remains pass if any runtime behavior is changed.

All listed closure conditions are satisfied.

## Next Single Step

Milestone 02 is complete. Proceed to Milestone 03 when instructed.
