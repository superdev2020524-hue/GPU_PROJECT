# API Coverage Matrix - Milestone 02

Date: 2026-04-28

Scope: current source-level behavior in the Phase 3 vGPU layer.

## Status Key

- `implemented`: mediated to host or locally correct for the current vGPU contract.
- `implemented but Ollama-shaped`: works for the preserved Ollama path but is not
  general enough to trust for frameworks.
- `partial`: some behavior works, but semantics are incomplete.
- `stubbed`: exported and returns success or fixed data without real semantics.
- `missing`: no current implementation found in the audited source.
- `unsafe fallback`: hides failure or reports success in a way that can mislead a
  general workload.
- `unsupported by design`: returns a clear unsupported error.
- `not required for current milestone`: intentionally not part of this audit gate.

## CUDA Driver API

| Entry | Status | Evidence | Risk / Notes |
| --- | --- | --- | --- |
| `cuInit` | implemented | `guest-shim/libvgpu_cuda.c`, `CUDA_CALL_INIT` | Lazy init and transport discovery are in place. |
| `cuDriverGetVersion` | implemented | `CUDA_CALL_DRIVER_GET_VERSION` | Host version is returned through executor. |
| `cuGetProcAddress`, `cuGetProcAddress_v2` | partial | `guest-shim/libvgpu_cuda.c` | Broad symbol routing exists, but correctness depends on each exported symbol. |
| `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceGetAttribute`, `cuDeviceTotalMem`, `cuDeviceComputeCapability`, `cuDeviceGetPCIBusId`, `cuDeviceGetUuid` | implemented but Ollama-shaped | guest-side cached/default GPU info and executor query cases | Good for discovery; must be verified against framework expectations before treating as general. |
| `cuCtxCreate`, `cuCtxCreate_v2`, `cuCtxCreate_v3`, `cuCtxDestroy`, `cuCtxSetCurrent`, `cuCtxGetCurrent`, `cuCtxSynchronize`, `cuCtxGetDevice`, `cuCtxGetApiVersion`, `cuCtxPushCurrent_v2`, `cuCtxPopCurrent_v2`, `cuCtxGetFlags` | partial | `CUDA_CALL_CTX_*` cases | Context lifecycle is mediator-managed; destroy/release paths are no-op or retained for stability. |
| `cuDevicePrimaryCtxRetain`, `cuDevicePrimaryCtxRelease`, `cuDevicePrimaryCtxRelease_v2`, `cuDevicePrimaryCtxReset`, `cuDevicePrimaryCtxReset_v2`, `cuDevicePrimaryCtxSetFlags`, `cuDevicePrimaryCtxSetFlags_v2`, `cuDevicePrimaryCtxGetState` | partial | `CUDA_CALL_DEVICE_PRIMARY_CTX_*` cases | Enough for current baseline; release/reset semantics are simplified. |
| `cuMemAlloc`, `cuMemAlloc_v2`, `cuMemFree`, `cuMemFree_v2` | implemented | `CUDA_CALL_MEM_ALLOC`, `CUDA_CALL_MEM_FREE` | Proven by Milestone 01 Driver and Runtime probes. |
| `cuMemAllocManaged` | partial | delegates to `cuMemAlloc_v2` | Unified memory semantics are not implemented. |
| `cuMemAllocPitch`, `cuMemAllocPitch_v2` | partial | allocates linear memory and returns pitch as width | No real pitch alignment semantics. |
| `cuMemGetAddressRange`, `cuMemGetAddressRange_v2` | unsafe fallback | returns base as input pointer and size `0` | Can mislead code that relies on allocation bounds. |
| `cuMemHostAlloc`, `cuMemFreeHost` | partial | local aligned host allocation/free | Host pinning/device mapping semantics are incomplete. |
| `cuMemHostRegister`, `cuMemHostUnregister` | unsafe fallback | returns success without real host registration | General workloads may assume pinned/registered memory. |
| `cuMemHostGetDevicePointer` | unsafe fallback | returns host pointer as device pointer | Not safe as a general CUDA contract. |
| `cuMemcpyHtoD`, `cuMemcpyHtoD_v2`, `cuMemcpyDtoH`, `cuMemcpyDtoH_v2`, `cuMemcpyDtoD`, `cuMemcpyDtoD_v2`, `cuMemcpy`, `cuMemcpyAsync`, `cuMemcpyHtoDAsync`, `cuMemcpyHtoDAsync_v2`, `cuMemcpyDtoHAsync`, `cuMemcpyDtoHAsync_v2`, `cuMemcpyDtoDAsync`, `cuMemcpyDtoDAsync_v2` | implemented / partial | `CUDA_CALL_MEMCPY_*` cases | 1D copies are mediated; async behavior often uses sync fallback or managed pending staging. |
| `cuMemcpy2DUnaligned`, `cuMemcpy2DAsync`, `cuMemcpy2DAsync_v2`, `cuMemcpy3D`, `cuMemcpy3DAsync`, `cuMemcpy3DPeer`, `cuMemcpy3DPeerAsync`, `cuMemcpyBatchAsync`, `cuMemcpy3DBatchAsync` | unsupported by design | `log_not_supported_copy_path()` | Good failure shape: clear `CUDA_ERROR_NOT_SUPPORTED`. |
| `cuMemcpyPeer`, `cuMemcpyPeerAsync` | partial | delegates to DtoD while ignoring peer contexts | Single-VM/single-GPU only. |
| `cuMemsetD8`, `cuMemsetD8_v2`, `cuMemsetD8Async`, `cuMemsetD32`, `cuMemsetD32_v2` | implemented / partial | `CUDA_CALL_MEMSET_D8`, `CUDA_CALL_MEMSET_D32` | 1D memset path exists; async stream semantics are simplified. |
| `cuMemsetD2D8`, `cuMemsetD2D8Async` | unsupported by design | `log_not_supported_copy_path()` | Clear unsupported error. |
| `cuMemGetInfo`, `cuMemGetInfo_v2` | implemented | `CUDA_CALL_MEM_GET_INFO` | Host memory info is queried by executor. |
| `cuModuleLoadData`, `cuModuleLoadDataEx`, `cuModuleLoadFatBinary`, `cuModuleUnload`, `cuModuleGetFunction`, `cuModuleGetGlobal`, `cuModuleGetGlobal_v2` | implemented / partial | `CUDA_CALL_MODULE_*` cases | PTX/fatbin loading and function lookup are proven for Milestone 01; global lookup is present but not gate-proven. |
| `cuModuleLoad`, `cuModuleGetTexRef`, `cuModuleGetSurfRef` | unsupported by design | `DEFINE_CUDA_NOT_SUPPORTED_STUB` | Clear unsupported error. |
| `cuLibraryLoadData`, `cuLibraryUnload`, `cuLibraryGetModule` | partial | `CUDA_CALL_LIBRARY_*` cases | Used for CUDA library initialization/Ollama path; not fully general. |
| `cuLibraryLoadFromFile`, `cuLibraryGetGlobal`, `cuLibraryGetManaged`, `cuLibraryGetUnifiedFunction`, `cuLibraryGetKernelCount`, `cuLibraryEnumerateKernels` | unsupported by design | `DEFINE_CUDA_NOT_SUPPORTED_STUB` | Clear unsupported error. |
| `cuLibraryGetKernel`, `cuKernelGetFunction` | partial | maps library kernel to module function handle | Useful bridge, not complete CUDA library semantics. |
| `cuKernelGetAttribute`, `cuKernelSetAttribute`, `cuKernelSetCacheConfig`, `cuKernelGetName`, `cuKernelGetParamInfo` | unsupported by design | `DEFINE_CUDA_NOT_SUPPORTED_STUB` | Clear unsupported error. |
| `cuLaunchKernel`, `cuLaunchKernel_ptsz`, `cuLaunchKernelEx`, `cuLaunchKernelEx_ptsz` | implemented / partial | `CUDA_CALL_LAUNCH_KERNEL` | Proven for Milestone 01 raw kernels. Parameter layout fallback remains a candidate risk. |
| `cuFuncGetParamInfo` | partial | `CUDA_CALL_FUNC_GET_PARAM_INFO` | Host can return `801`; launch fallback preserves generic PTX path. |
| `cuFuncGetAttribute` | unsafe fallback | returns synthetic non-zero defaults if RPC fails | Keeps Ollama alive but can hide unsupported attributes. |
| `cuFuncSetCacheConfig` | partial | `CUDA_CALL_FUNC_SET_CACHE_CONFIG` | Mediated call exists. |
| `cuFuncSetAttribute` | unsafe fallback | logs no-op success | Must not be treated as real support. |
| `cuFuncSetSharedMemConfig`, `cuFuncGetName` | unsupported by design | `log_not_supported_runtime_feature()` | Clear unsupported error. |
| `cuStreamCreate`, `cuStreamCreateWithFlags`, `cuStreamCreateWithPriority`, `cuStreamDestroy`, `cuStreamSynchronize`, `cuStreamQuery`, `cuStreamWaitEvent`, `cuStreamGetFlags`, `cuStreamGetPriority`, `cuStreamGetDevice`, `cuStreamGetCtx` | implemented / partial | `CUDA_CALL_STREAM_*` cases and local wrappers | Basic stream path exists; priority/query/device semantics need framework testing. |
| `cuStreamCopyAttributes`, `cuStreamGetAttribute`, `cuStreamSetAttribute` and `_ptsz` variants | unsafe fallback | return `CUDA_SUCCESS` without meaningful attributes | Likely to mislead frameworks using stream attributes. |
| `cuStreamWaitValue32`, `cuStreamWriteValue32`, `cuStreamWaitValue64`, `cuStreamWriteValue64`, `cuStreamBatchMemOp`, callback/attach/PTSZ variants | unsupported by design | explicit unsupported stubs | Clear unsupported error for most advanced stream ops. |
| `cuEventCreate`, `cuEventCreateWithFlags`, `cuEventDestroy`, `cuEventRecord`, `cuEventRecordWithFlags`, `cuEventSynchronize`, `cuEventQuery`, `cuEventElapsedTime` | implemented / partial | `CUDA_CALL_EVENT_*` cases | Basic path proven by Milestone 01; timing semantics should be verified later. |
| Array, mipmapped array, texture, surface APIs | unsupported by design | `log_not_supported_object_path()` | Clear unsupported error. |
| External memory/semaphore, IPC, GL/EGL interop APIs | unsupported by design | `log_not_supported_runtime_feature()` | Clear unsupported error. |
| `cuLink*`, tensor map, GPUDirect RDMA, pointer attributes, async pool allocation/pool management, memory advise/prefetch | unsupported by design | `DEFINE_CUDA_NOT_SUPPORTED_STUB` | Clear unsupported error. |
| `cuDeviceGetP2PAttribute` | unsupported by design | explicit executor case after `M02-E3` | Protocol ID now has a clear `CUDA_ERROR_NOT_SUPPORTED` disposition. |
| `cuMemsetD16` | unsupported by design | explicit executor case after `M02-E3` | Protocol ID now has a clear `CUDA_ERROR_NOT_SUPPORTED` disposition. |
| `cuMemAllocHost`, `cuMemFreeHost` protocol calls | unsupported by design | explicit executor cases after `M02-E3` | Guest-side host alloc exists, but protocol-only path now fails clearly if called. |
| `cuLaunchCooperativeKernel` | unsupported by design | explicit executor case after `M02-E3` | Cooperative launch is not implemented for the current vGPU contract. |
| `cuTexObjectCreate`, `cuTexObjectDestroy` protocol calls | unsupported by design | explicit executor cases after `M02-E3` | Guest returns unsupported for texture paths; protocol path is now explicit too. |

## CUDA Runtime API

| Entry | Status | Evidence | Risk / Notes |
| --- | --- | --- | --- |
| `cudaRuntimeGetVersion`, `cudaRuntimeGetVersion_v2`, `cudaDriverGetVersion` | implemented but Ollama-shaped | returns compatible/default version | Version is synthetic, not direct real runtime state. |
| `cudaGetDeviceCount`, `cudaGetDevice`, `cudaDeviceGetAttribute`, `cudaGetDeviceProperties`, `cudaGetDeviceProperties_v2` | implemented but Ollama-shaped | cached/default H100 properties and GGML offset patching | Critical for Ollama; not general CUDA property fidelity. |
| `cudaGetErrorString` | implemented | local string mapping | Limited error coverage. |
| `cudaGetLastError`, `cudaPeekAtLastError` | unsafe fallback | always `cudaSuccess` | Hides prior errors from frameworks. |
| `cudaMalloc`, `cudaFree` | implemented / partial | forwards to Driver API/transport | Proven in Milestone 01 Runtime probe. |
| `cudaMallocHost`, `cudaFreeHost` | partial | local aligned allocation/free | No real CUDA pinned-host semantics. |
| `cudaMallocManaged` | unsupported by design | returns `cudaErrorNotSupported` after `M02-E1` | Unified memory remains unimplemented, but now fails closed. |
| `cudaMemGetInfo` | implemented but synthetic | returns default total/free | Not live per-process allocator state. |
| `cudaHostRegister`, `cudaHostUnregister` | unsupported by design | returns `cudaErrorNotSupported` after `M02-E1` | Host registration remains unimplemented, but no longer reports fake success. |
| `cudaMemcpy`, `cudaMemcpyAsync` | implemented / partial | forwards to Driver API; async may sync-fallback | Proven in Milestone 01 Runtime probe. |
| `cudaMemcpy2DAsync`, `cudaMemcpy3DPeerAsync`, `cudaMemcpyPeerAsync` | unsupported by design | returns `cudaErrorNotSupported` after `M02-E1` | No longer silently skips unsupported copy paths. |
| `cudaMemset`, `cudaMemsetAsync` | implemented / partial | forwards to Driver memset; returns an error if forwarding fails after `M02-E1` | 1D memset only; async stream semantics remain simplified. |
| `cudaDeviceSynchronize` | implemented / partial | forwards to Driver `cuCtxSynchronize` after `M02-E1` | Preserves Driver failure instead of hiding it. |
| `cudaDeviceReset` | unsafe fallback | returns success | Does not guarantee real device reset. |
| `cudaDeviceCanAccessPeer` | partial | returns no peer access | Acceptable for single GPU if callers handle it. |
| `cudaDeviceEnablePeerAccess`, `cudaDeviceDisablePeerAccess` | unsupported by design | returns `cudaErrorNotSupported` after `M02-E1` | Peer access remains unsupported but now fails closed. |
| `cudaSetDevice`, `cudaSetDeviceFlags` | partial | device 0 only; flags no-op | Enough for single GPU, incomplete semantics. |
| `cudaStreamCreate`, `cudaStreamCreateWithFlags`, `cudaStreamDestroy` | implemented / partial | forwards to Driver API | Basic stream path proven. |
| `cudaStreamSynchronize` | implemented / partial | forwards to Driver sync and preserves failure after `M02-E1` | Basic stream sync no longer hides Driver errors. |
| `cudaStreamBeginCapture`, `cudaStreamEndCapture` | unsupported by design | returns `cudaErrorNotSupported` after `M02-E1` | Graph capture remains unimplemented but fails closed. |
| `cudaStreamIsCapturing` | stubbed | returns not-capturing | Candidate for later cleanup. |
| `cudaStreamWaitEvent` | partial | forwards if symbol exists; missing symbol is initialization error after `M02-E1` | No longer reports success when Driver symbol is absent. |
| `cudaEventCreate`, `cudaEventCreateWithFlags`, `cudaEventDestroy`, `cudaEventRecord`, `cudaEventSynchronize` | implemented / partial | forwards to Driver API where available | Missing destroy/record/sync symbols now fail with initialization error after `M02-E1`. |
| `cudaLaunchKernel` | unsupported by design | returns `cudaErrorNotSupported` after `M02-E1` | Runtime-compiled kernels no longer report fake success. |
| `cudaFuncGetAttributes`, `cudaFuncSetAttribute` | implemented but synthetic / unsafe fallback | synthetic attributes and no-op success | Useful for GGML sizing; not real function metadata. |
| Runtime occupancy functions | implemented but synthetic | local heuristic values | Must be validated with frameworks. |
| `cudaGraphDestroy`, `cudaGraphInstantiate`, `cudaGraphLaunch`, `cudaGraphExecDestroy`, `cudaGraphExecUpdate` | unsupported by design | returns `cudaErrorNotSupported` after `M02-E1` | Graph paths no longer report fake success. |
| `__cudaRegisterFatBinary`, `__cudaRegisterFatBinaryEnd`, `__cudaRegisterFunction`, `__cudaRegisterVar` | stubbed | registration placeholders | Runtime kernel path is not real. |

## cuBLAS

| Entry | Status | Evidence | Risk / Notes |
| --- | --- | --- | --- |
| `cublasCreate`, `cublasCreate_v2` | implemented / partial | RPC create, real fallback; no-transport path fails closed after `M02-E2` | No longer creates a fake success stub handle when transport/context is unavailable. |
| `cublasDestroy`, `cublasDestroy_v2` | implemented / partial | RPC destroy or local stub free | Depends on handle mode. |
| `cublasSetStream`, `cublasSetStream_v2` | implemented / partial | RPC set-stream for remote handles; historical stub handles fail closed after `M02-E2` | Remote path remains supported. |
| `cublasGetStream`, `cublasGetStream_v2` | partial | returns stored remote stream; historical stub handles fail closed after `M02-E2` | Remote path remains supported. |
| `cublasSetMathMode`, `cublasGetMathMode` | partial | remote handles return success/default mode; historical stub handles fail closed after `M02-E2` | Math mode still not fully enforced. |
| `cublasGetStatusString` | implemented | real symbol if available, local fallback strings | Low risk. |
| `cublasSgemm_v2` | implemented / partial | `CUDA_CALL_CUBLAS_SGEMM` RPC; historical stub handles fail closed after `M02-E2` | Needs independent numerical correctness gate. |
| `cublasGemmEx` | implemented / partial | `CUDA_CALL_CUBLAS_GEMM_EX` RPC; historical stub handles fail closed after `M02-E2` | Type coverage must be validated beyond Ollama shapes. |
| `cublasGemmStridedBatchedEx` | implemented / partial | `CUDA_CALL_CUBLAS_GEMM_STRIDED_BATCHED_EX` RPC; historical stub handles fail closed after `M02-E2` | Needs independent matrix correctness gate. |
| `cublasGemmBatchedEx` | implemented / partial | `CUDA_CALL_CUBLAS_GEMM_BATCHED_EX` RPC; historical stub handles fail closed after `M02-E2` | Pointer-table copy path is risky and needs focused testing. |
| `cublasStrsmBatched` | partial | real library fallback only | No mediated RPC path found. |

## cuBLASLt

| Entry | Status | Evidence | Risk / Notes |
| --- | --- | --- | --- |
| `cublasLtCreate` | stubbed | returns handle `0x1` and success | No real Lt handle. |
| `cublasLtDestroy` | stubbed | returns success | No real cleanup. |
| `cublasLtMatmulDescCreate`, `cublasLtMatmulDescDestroy` | unsupported by design | returns `CUBLAS_STATUS_NOT_SUPPORTED` after `M02-E1` | Descriptor state remains unimplemented, but no longer reports fake success. |
| `cublasLtMatrixLayoutCreate`, `cublasLtMatrixLayoutDestroy` | unsupported by design | returns `CUBLAS_STATUS_NOT_SUPPORTED` after `M02-E1` | Layout state remains unimplemented, but no longer reports fake success. |
| `cublasLtMatmulPreferenceCreate`, `cublasLtMatmulPreferenceDestroy` | unsupported by design | returns `CUBLAS_STATUS_NOT_SUPPORTED` after `M02-E1` | Preference state remains unimplemented, but no longer reports fake success. |
| `cublasLtMatmulAlgoGetHeuristic` | unsupported by design | returns `CUBLAS_STATUS_NOT_SUPPORTED` and zero algorithms after `M02-E1` | Does not return fake algorithm choices. |
| `cublasLtMatmul` | unsupported by design | returns `CUBLAS_STATUS_NOT_SUPPORTED` after `M02-E1` | cuBLASLt compute no longer reports fake success. |

## NVML

| Entry | Status | Evidence | Risk / Notes |
| --- | --- | --- | --- |
| `nvmlInit`, `nvmlInit_v2`, `nvmlShutdown` | implemented but Ollama-shaped | deferred transport and default GPU info | Designed for discovery speed and stability. |
| `nvmlSystemGetDriverVersion`, `nvmlSystemGetCudaDriverVersion`, `nvmlSystemGetCudaDriverVersion_v2`, `nvmlSystemGetNVMLVersion` | implemented but synthetic | default/derived version strings | Not live host NVML state. |
| `nvmlDeviceGetCount`, `nvmlDeviceGetCount_v2` | implemented but Ollama-shaped | returns one GPU, immediate LD_PRELOAD path | Correct for current VM, not multi-GPU aware. |
| `nvmlDeviceGetHandleByIndex`, `nvmlDeviceGetHandleByIndex_v2`, `nvmlDeviceGetHandleByUUID`, `nvmlDeviceGetHandleByPciBusId`, `nvmlDeviceGetHandleByPciBusId_v2` | partial | returns singleton `g_device` | UUID and BDF matching are synthetic/default-backed. |
| `nvmlDeviceGetIndex`, `nvmlDeviceGetName`, `nvmlDeviceGetUUID`, `nvmlDeviceGetMemoryInfo`, `nvmlDeviceGetMemoryInfo_v2`, `nvmlDeviceGetPciInfo`, `nvmlDeviceGetPciInfo_v3`, `nvmlDeviceGetCudaComputeCapability`, `nvmlDeviceGetComputeMode` | implemented but synthetic | default H100 values and discovered BDF fallback | Good for discovery; not telemetry-accurate. |
| Temperature, utilization, power, clocks, fan, ECC, process list, persistence/performance state, board/serial/architecture/display/core queries | stubbed / synthetic | fixed values returned with success | Tools may report plausible but fake telemetry. |
| `nvmlErrorString` | implemented | local mapping | Limited mapping. |
| `nvmlDeviceGetEncoderUtilization`, `nvmlDeviceGetDecoderUtilization` | unsafe fallback | returns zero/success | Comment says not-supported stubs, implementation returns success. |
| `nvmlDeviceGetBAR1MemoryInfo` | unsupported by design | returns `NVML_ERROR_NOT_SUPPORTED` | Clear unsupported error. |
| PCIe link, throughput, inforom, virtualization mode queries | stubbed / synthetic | fixed values returned with success | Not live telemetry. |

## Protocol / Executor Coverage Summary

| Area | Status | Evidence | Risk / Notes |
| --- | --- | --- | --- |
| Protocol IDs in `include/cuda_protocol.h` | implemented / partial | call IDs cover init, device, context, memory, module, launch, stream, event, cuBLAS, cuBLASLt, function metadata | After `M02-E3`, every protocol ID has an executor case or explicit unsupported disposition. |
| Executor switch in `src/cuda_executor.c` | implemented / partial | final source comparison after `M02-E3`: `protocol_ids=86`, `executor_case_ids=86`, `missing_cases=[]` | Core Milestone 01 path is covered; protocol-only unsupported IDs now fail clearly. |
| cuBLASLt protocol IDs | unsupported by design | explicit executor cases after `M02-E3`; guest cuBLASLt does not use these IDs today | cuBLASLt compute remains unsupported until a later correctness gate implements it. |
