# 05 - Second Framework Gate

## Purpose

Prove that the mediated vGPU layer is not accidentally Ollama-only or
PyTorch-only.

## Candidate Runtimes

- TensorFlow;
- CuPy;
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
