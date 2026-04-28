# Evidence - Milestone 02 API Coverage Audit

## Baseline Evidence

- Plan A: `/tmp/phase1_milestone_gate_after_m01_hidden_risk_sweep.json` ->
  `overall_pass=True`.
- Milestone 01 hidden-risk gate:
  `/tmp/phase3_general_cuda_gate_hidden_risk_sweep_rerun1.json` ->
  `overall_pass=True`.

## Audit Evidence To Collect

- Source-level API definitions from:
  - `guest-shim/libvgpu_cuda.c`
  - `guest-shim/libvgpu_cudart.c`
  - `guest-shim/libvgpu_cublas.c`
  - `guest-shim/libvgpu_cublasLt.c`
  - `guest-shim/libvgpu_nvml.c`
- Protocol coverage from:
  - `include/cuda_protocol.h`
  - `src/cuda_executor.c`
- Current deployed symbol evidence from VM where useful.

## Current Session Findings

### Source Audit

- Command method: direct source inspection of the audit-scope files.
- Files inspected:
  - `phase3/guest-shim/libvgpu_cuda.c`
  - `phase3/guest-shim/libvgpu_cudart.c`
  - `phase3/guest-shim/libvgpu_cublas.c`
  - `phase3/guest-shim/libvgpu_cublasLt.c`
  - `phase3/guest-shim/libvgpu_nvml.c`
  - `phase3/include/cuda_protocol.h`
  - `phase3/src/cuda_executor.c`

### Key Evidence

- Driver API core path is mediated through protocol calls for init, context,
  memory, copy, module load, kernel launch, stream, event, function metadata,
  and selected cuBLAS calls.
- Driver API unsupported areas mostly return clear `CUDA_ERROR_NOT_SUPPORTED`
  through explicit unsupported helpers or stubs.
- Driver API still has unsafe fallbacks: `cuFuncGetAttribute` synthesizes
  defaults after host metadata failure, `cuFuncSetAttribute` is no-op success,
  stream attribute helpers return success, and host registration/address-range
  helpers do not provide full CUDA semantics.
- Runtime API includes real paths for device discovery, allocation, copy,
  streams, and events, but also includes success-returning stubs for
  `cudaLaunchKernel`, graph APIs, managed memory, peer access enable/disable,
  several copy variants, host registration, and some synchronization paths.
- cuBLAS has mediated RPC support for create/destroy, stream association,
  `cublasSgemm_v2`, `cublasGemmEx`, `cublasGemmStridedBatchedEx`, and
  `cublasGemmBatchedEx`. The original audit found that stub handle fallback
  could still return success without host compute; `M02-E2` closed that path.
- cuBLASLt currently returns success for all listed calls and does not perform
  real descriptor/layout/heuristic/matmul work.
- NVML is largely discovery-oriented: it returns one H100-like GPU and many
  synthetic/default telemetry values. `nvmlDeviceGetBAR1MemoryInfo` clearly
  returns unsupported, but several monitoring calls return plausible success.
- The initial protocol audit found IDs without executor cases, including P2P
  attributes, D16 memset, host allocation/free, cooperative launch, texture
  object calls, context push/pop, async DtoH/DtoD, and cuBLASLt calls.
  `M02-E3` closed this by adding explicit executor dispositions.

### Output Artifacts

- `API_COVERAGE_MATRIX.md`
- `GAP_LIST.md`

### Current Active Error

None. Milestone 02 active errors are closed.

### `M02-E1` Correction Evidence

- Before change, preserved Plan A passed:
  `/tmp/phase1_milestone_gate_before_m02_e1.json` -> `overall_pass=True`.
- Local IDE lint check showed no diagnostics for:
  - `phase3/guest-shim/libvgpu_cudart.c`
  - `phase3/guest-shim/libvgpu_cublasLt.c`
- Local `gcc -fsyntax-only` could not run because local `gcc` is not installed.
- VM build/deploy succeeded for:
  - `/opt/vgpu/lib/libvgpu-cudart.so`
  - `/opt/vgpu/lib/libvgpu-cublasLt.so.12`
- Live artifact proof:
  - `/opt/vgpu/lib/libvgpu-cudart.so`
    `66b10c345acd084164b115df5fc7b9b8851fe18583610e7c3d12ac90f17149cc`
    size `45416`
  - `/opt/vgpu/lib/libvgpu-cublasLt.so.12`
    `28907f065f14b2e00686e1620057588dbf008e4b0da7d1091278eea4841bd3da`
    size `16160`
  - `systemctl is-active ollama` -> `active`
- After deployment, Plan A passed:
  `/tmp/phase1_milestone_gate_after_m02_e1_fail_closed.json` ->
  `overall_pass=True`.
- After deployment, Milestone 01 raw CUDA regression gate passed:
  `/tmp/phase3_general_cuda_gate_after_m02_e1_fail_closed.json` ->
  `overall_pass=True`, including 5/5 Driver API probe runs and 5/5 Runtime API
  probe runs.

### `M02-E1` Behavior Changed

- Runtime `cudaLaunchKernel` now returns `cudaErrorNotSupported`.
- Runtime graph APIs now return `cudaErrorNotSupported`.
- Runtime `cudaMallocManaged`, `cudaHostRegister`, `cudaHostUnregister`,
  peer-access enable/disable, 2D/3D/peer async copies, and graph capture now
  fail closed.
- Runtime `cudaDeviceSynchronize` and `cudaStreamSynchronize` now forward to
  Driver synchronization and preserve failure.
- Runtime event/wait wrappers no longer return success when required Driver
  symbols are absent.
- cuBLASLt descriptor/layout/preference/heuristic/matmul calls now return
  `CUBLAS_STATUS_NOT_SUPPORTED` instead of generic success.

### Next Single Step

### `M02-E2` Correction Evidence

- Before change, preserved Plan A passed:
  `/tmp/phase1_milestone_gate_before_m02_e2.json` -> `overall_pass=True`.
- VM build/deploy succeeded for `/opt/vgpu/lib/libvgpu-cublas.so.12`.
- Live artifact proof:
  - `/opt/vgpu/lib/libvgpu-cublas.so.12`
    `80659ffeb12467a8df36ff225aee5a22629eab947a6ad137abb33113f3773a5b`
    size `113192`
  - `systemctl is-active ollama` -> `active`
- After deployment, Plan A passed:
  `/tmp/phase1_milestone_gate_after_m02_e2_cublas_fail_closed.json` ->
  `overall_pass=True`.
- After deployment, Milestone 01 raw CUDA regression gate passed:
  `/tmp/phase3_general_cuda_gate_after_m02_e2_cublas_fail_closed.json` ->
  `overall_pass=True`, including 5/5 Driver API probe runs and 5/5 Runtime API
  probe runs.

### `M02-E2` Behavior Changed

- `cublasCreate_v2` no longer creates a local stub handle when no real
  transport/context exists.
- No-transport cuBLAS creation now returns `CUBLAS_STATUS_NOT_INITIALIZED` and
  records `VGPU_CUBLAS_MODE=NO_TRANSPORT`.
- Historical stub-handle compute/control paths now fail closed with
  `CUBLAS_STATUS_NOT_INITIALIZED`.

### `M02-E3` Correction Evidence

- Before change, preserved Plan A passed:
  `/tmp/phase1_milestone_gate_before_m02_e3.json` -> `overall_pass=True`.
- Initial executor update built successfully, but live-process verification
  found the mediator was still running from
  `/root/phase3/mediator_phase3 (deleted)`. This evidence was rejected as
  insufficient.
- Final source-level comparison after the corrected patch:
  `protocol_ids=86`, `executor_case_ids=86`, `missing_cases=[]`.
- Final host build and live restart proof:
  - `/root/phase3/mediator_phase3`
    `8f306df61150071553a5dc7c9b8cba257658111acf6a71331aaa2fb7ebebe796`
  - `/root/phase3/src/cuda_executor.c`
    `0f2b66f05b9c633592f44cdb4fe4b1596b63d2733eaedccd3af2518d35cfd21c`
  - mediator PID `295063`, executable `/root/phase3/mediator_phase3`
- After the final live-binary restart, Plan A passed:
  `/tmp/phase1_milestone_gate_after_m02_e3_live_binary_restart.json` ->
  `overall_pass=True`.
- After the final live-binary restart, Milestone 01 raw CUDA regression gate
  passed:
  `/tmp/phase3_general_cuda_gate_after_m02_e3_live_binary_restart.json` ->
  `overall_pass=True`, including 5/5 Driver API probe runs and 5/5 Runtime API
  probe runs.

### `M02-E3` Behavior Changed

- Protocol IDs without real executor behavior now have explicit executor cases
  returning `CUDA_ERROR_NOT_SUPPORTED` instead of only falling through the
  generic default.
- `executor_call_id_to_name()` now names the previously ambiguous protocol IDs,
  improving failure logs for future framework gates.

### Milestone 02 Closure Evidence

- `API_COVERAGE_MATRIX.md` exists and classifies audited Driver, Runtime,
  cuBLAS, cuBLASLt, NVML, and protocol/executor behavior.
- `GAP_LIST.md` exists and carries remaining P1/P2 risks forward.
- `M02-E1`, `M02-E2`, and `M02-E3` are closed with Plan A and raw CUDA
  regression evidence.
- Remaining synthetic or Ollama-shaped behavior is classified as P1/P2 follow-up
  work rather than an unresolved Milestone 02 blocker.
