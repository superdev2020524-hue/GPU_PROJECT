# Gate - Milestone 02 API Coverage Audit

## Gate Name

`phase3_api_coverage_audit`

## Purpose

Create an explicit API coverage matrix for the current vGPU layer before moving
to framework gates.

## Audit Scope

- CUDA Driver API: `guest-shim/libvgpu_cuda.c`, `include/cuda_protocol.h`,
  `src/cuda_executor.c`
- CUDA Runtime API: `guest-shim/libvgpu_cudart.c`
- cuBLAS: `guest-shim/libvgpu_cublas.c`
- cuBLASLt: `guest-shim/libvgpu_cublasLt.c`
- NVML: `guest-shim/libvgpu_nvml.c`

## Status Values

- `implemented`
- `implemented but Ollama-shaped`
- `partial`
- `stubbed`
- `missing`
- `unsafe fallback`
- `unsupported by design`
- `not required for current milestone`

## Pass Criteria

- API matrix exists.
- Every audited API entry has a status.
- Stubbed and unsupported calls are separated from implemented calls.
- Silent-success behavior is identified as a risk.
- Prioritized gap list exists for later milestones.
- Plan A baseline remains preserved because this is audit-only work.

## Fail Criteria

- Any core API behavior remains unclassified.
- A general-workload dependency is silently treated as supported without proof.
- Audit work changes runtime behavior without Plan A recheck.

## Required Output

- `API_COVERAGE_MATRIX.md`
- `GAP_LIST.md`
- Updated `ACTIVE_ERROR.md`, `EVIDENCE.md`, and `DECISIONS.md`
