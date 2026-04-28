# 02 - API Coverage Audit

## Purpose

Create an explicit API coverage matrix for the general vGPU layer.

## Scope

Audit the implemented and exposed behavior for:

- CUDA Driver API;
- CUDA Runtime API;
- cuBLAS;
- cuBLASLt;
- NVML.

## Status Values

- implemented;
- implemented but Ollama-shaped;
- partial;
- stubbed;
- missing;
- unsafe fallback;
- unsupported by design;
- not required for current milestone.

## Closure Criteria

- Each relevant API entry has a status.
- Unsupported calls return clear errors.
- No general-workload dependency is silently faked as success.
- Prioritized gap list exists for later milestones.
