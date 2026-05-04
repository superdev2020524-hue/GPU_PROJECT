# 05 - Second Framework Gate

## Purpose

Prove that the mediated vGPU layer is not accidentally Ollama-only or
PyTorch-only.

## Candidate Runtimes

- CuPy;
- TensorFlow;
- ONNX Runtime;
- Numba;
- another agreed GPU software stack.

## Gate Coverage

- GPU detection;
- small allocation;
- transfer;
- simple compute;
- repeated process cleanup.

## Closure Criteria

- at least one second non-Ollama framework passes an agreed gate;
- unsupported behavior is documented;
- Ollama baseline does not regress.

## Current Status

Milestone 05 is **not complete for broadened framework coverage**.

The scoped CuPy lane is complete, but the TensorFlow lane is now active after a
direct TensorFlow MNIST probe trained on CPU instead of registering a GPU.

Entry baseline:

- Milestone 04 is complete.
- Final serial preservation passed Plan A, Plan B, Plan C, Milestone 01,
  Milestone 02, Milestone 03, and the scoped Milestone 04 PyTorch gate.

CuPy lane closure evidence:

- CuPy `14.0.1` is installed in `/mnt/m04-pytorch/venv`.
- The scoped CuPy gate passed 3/3 fresh-process runs:
  `/tmp/m05_cupy_preservation_after_m05/run_1.json` through `run_3.json`.
- Final serial tail after M05 passed:
  - Milestone 03 4 MiB async stream/event preservation: 3/3.
  - Milestone 04 PyTorch preservation: 3/3.
  - Milestone 05 CuPy preservation: 3/3.

Candidate follow-up:

- Additional CuPy factory/reduction kernels remain outside this scoped gate and
  should be handled by a later broader framework-coverage milestone if needed.
- TensorFlow GPU mode is no longer a vague future candidate. It is an active
  M05 framework-expansion blocker until TensorFlow can register a mediated GPU
  and run a bounded training step on GPU, or until the client explicitly removes
  TensorFlow from the required expansion surface.
