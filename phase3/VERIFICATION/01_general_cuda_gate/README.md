# 01 - General CUDA Compatibility Gate

## Purpose

Create the first non-Ollama gate for the mediated vGPU layer.

This milestone proves basic CUDA behavior below PyTorch, TensorFlow, or other
large frameworks. It prevents the next phase from becoming another broad
workload-specific investigation.

## Required Gate Coverage

- device count and properties;
- `cudaMalloc` / `cuMemAlloc`;
- `cudaFree` / `cuMemFree`;
- HtoD copy;
- DtoH copy;
- simple kernel launch;
- stream create and synchronize;
- event record and synchronize;
- module load and function lookup;
- repeated process start/stop cleanup.

## Required Records

Before implementation starts, create:

- `BASELINE.md`
- `GATE.md`
- `ACTIVE_ERROR.md`
- `EVIDENCE.md`
- `DECISIONS.md`

Use `../templates/MILESTONE_RECORD_TEMPLATE.md`.

## Start Conditions

- Read `../VERIFICATION_RULES.md`.
- Prove `00_preserve_ollama_baseline` at least through Plan A.
- Confirm mediator, VM service, vGPU PCI device, and live artifact paths.
- Define one bounded CUDA gate runner and one JSON report path.

## Closure Criteria

- All gate cases pass in one clean VM session.
- Host evidence proves mediated physical GPU execution.
- VM evidence proves the expected API behavior.
- Memory allocations are freed or cleaned after process exit.
- No stale status or stale payload signature remains active.
- Plan A is rechecked after any shared runtime change.
