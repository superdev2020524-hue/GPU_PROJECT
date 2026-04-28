# Gap List - Milestone 02 API Coverage Audit

Date: 2026-04-28

## P0 - Must Resolve Before Framework Gates

1. Runtime `cudaLaunchKernel` was a success-returning stub. **Closed by
   `M02-E1` fail-closed change.**
   - Previous behavior: returned `cudaSuccess` without launching.
   - Current behavior: returns `cudaErrorNotSupported`.
   - Risk: Runtime-compiled kernels can appear to run while producing no work.
   - Required direction: either implement a Driver-backed launch path or return a
     clear unsupported error for unsupported Runtime kernel launches.

2. cuBLASLt was success-stubbed for descriptor/layout/preference/heuristic and
   matmul calls. **Closed by `M02-E1` fail-closed change for those calls.**
   - Previous behavior: descriptor/layout/preference/heuristic/matmul calls all
     returned success without real state or compute.
   - Current behavior: those calls return `CUBLAS_STATUS_NOT_SUPPORTED`.
   - Risk: PyTorch and other frameworks may route GEMM through cuBLASLt and get
     silent wrong results.
   - Required direction: replace `cublasLtMatmul` with real mediated execution
     or clear unsupported errors until implemented.

3. Runtime graph APIs were success-returning no-ops. **Closed by `M02-E1`
   fail-closed change.**
   - Previous behavior: graph instantiate/launch/update/destroy returned success
     with synthetic handles.
   - Current behavior: graph APIs return `cudaErrorNotSupported`.
   - Risk: framework graph capture/replay can silently skip work.
   - Required direction: return unsupported until a real graph path exists.

4. Runtime stream/device/event failure paths hid errors. **Partially closed by
   `M02-E1` fail-closed change.**
   - Previous behavior: `cudaStreamSynchronize`, some event wrappers, and
     `cudaStreamWaitEvent` could return success after missing symbols or Driver
     errors.
   - Current behavior: `cudaStreamSynchronize` and `cudaDeviceSynchronize`
     propagate Driver failure, and missing event/wait symbols return
     initialization error.
   - Risk: asynchronous errors can be hidden and corrupt later results.
   - Required direction: propagate Driver errors in framework gates.

5. Runtime host registration and managed memory were fake-success paths.
   **Closed by `M02-E1` fail-closed change.**
   - Previous behavior: `cudaHostRegister`, `cudaHostUnregister`, and
     `cudaMallocManaged` did not provide real CUDA semantics but returned
     success.
   - Current behavior: those calls return `cudaErrorNotSupported`.
   - Risk: frameworks may assume pinned or unified memory behavior.
   - Required direction: return unsupported or implement a constrained mediated
     model.

## P1 - High Priority Hardening

1. Driver `cuFuncGetAttribute` and Runtime function-attribute helpers synthesize
   defaults when host metadata is unavailable.
   - Risk: occupancy and launch configuration can be wrong for real kernels.
   - Direction: separate "safe defaults for Ollama" from general framework
     metadata behavior.

2. Driver `cuMemGetAddressRange`, host registration, and host-device pointer
   paths return plausible success without real ownership/bounds tracking.
   - Risk: pointer-introspection code may make unsafe decisions.
   - Direction: add allocation tracking or return clear unsupported status.

3. cuBLAS stub handle fallback could return success without host compute.
   **Closed by `M02-E2` fail-closed change.**
   - Previous risk: GEMM calls could be silently skipped if transport was
     unavailable.
   - Current behavior: `cublasCreate_v2` fails closed when no real
     transport/context exists, and historical stub-handle compute paths return
     `CUBLAS_STATUS_NOT_INITIALIZED`.

4. cuBLAS GEMM type and batched paths need independent correctness tests.
   - Risk: current proof is Ollama-shaped; Milestone 01 did not verify GEMM
     numerical correctness.
   - Direction: add focused SGEMM/GemmEx/strided/batched correctness probes.

5. Protocol IDs existed without executor cases. **Closed by `M02-E3` explicit
   executor disposition.**
   - Previous examples: `CUDA_CALL_DEVICE_GET_P2P_ATTRIBUTE`,
     `CUDA_CALL_MEMSET_D16`, `CUDA_CALL_MEM_ALLOC_HOST`,
     `CUDA_CALL_MEM_FREE_HOST`, `CUDA_CALL_LAUNCH_COOPERATIVE_KERNEL`,
     `CUDA_CALL_TEX_CREATE`, `CUDA_CALL_TEX_DESTROY`,
     `CUDA_CALL_CUBLASLT_CREATE`, `CUDA_CALL_CUBLASLT_DESTROY`,
     `CUDA_CALL_CUBLASLT_MATMUL`, plus protocol-only context push/pop and
     async DtoH/DtoD IDs.
   - Current behavior: final comparison reports `protocol_ids=86`,
     `executor_case_ids=86`, `missing_cases=[]`; unsupported protocol-only IDs
     now return `CUDA_ERROR_NOT_SUPPORTED` through explicit executor cases.

## P2 - Telemetry And Compatibility Cleanup

1. NVML returns mostly synthetic/default telemetry.
   - Risk: monitoring tools can display plausible but fake values.
   - Direction: document as discovery-only or connect selected telemetry to host
     NVML.

2. NVML comment/behavior mismatch.
   - Current behavior: encoder and decoder utilization are under a "return
     NOT_SUPPORTED" comment but return zero/success.
   - Direction: align behavior with the documented failure mode.

3. Driver advanced interop, texture, array, IPC, external memory/semaphore, and
   stream value operations are unsupported.
   - Current behavior: most return `CUDA_ERROR_NOT_SUPPORTED`, which is correct
     for now.
   - Direction: keep unsupported unless a later workload requires them.

4. Device and property reporting is strongly H100/Ollama-shaped.
   - Risk: future multi-GPU or non-H100 environments will inherit fixed values.
   - Direction: source properties from host state where feasible.

## Recommended Next Single Step

Milestone 02 is complete. Carry the remaining P1/P2 items into Milestone 03 and
later framework gates instead of treating them as unresolved Milestone 02
blockers.
