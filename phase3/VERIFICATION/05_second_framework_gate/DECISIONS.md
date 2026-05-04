# Decisions - Milestone 05 Second Framework Gate

## 2026-04-29 - Start With CuPy Probe

- Decision: use CuPy as the first Milestone 05 candidate runtime.
- Reason: CuPy is an independent CUDA framework with a smaller install and gate
  surface than TensorFlow while still testing CUDA allocation, transfer,
  elementwise kernels, and GEMM.
- Rejected alternatives: start with TensorFlow, ONNX Runtime, or Numba before
  proving the lighter independent CUDA stack; modify vGPU runtime code before an
  environment probe identifies the first blocker.
- Reversal/removal condition: switch candidates if CuPy cannot be installed
  safely in the isolated VM package area or proves unsuitable for a bounded gate.

## 2026-04-29 - Install CuPy In Isolated Framework Disk

- Decision: install CuPy into `/mnt/m04-pytorch/venv` rather than system Python
  or the root filesystem.
- Reason: the first M05 probe found `ModuleNotFoundError: cupy`; the dedicated
  framework disk has about `23G` free and already contains the isolated PyTorch
  environment used for M04.
- Rejected alternatives: install into system Python, use the root filesystem, or
  switch to a heavier framework before testing whether CuPy can run.
- Reversal/removal condition: stop and reassess if the install cannot complete
  safely or if CuPy imports but cannot use the mediated CUDA path.

## 2026-04-29 - Scope CuPy Matmul To CPU-Prepared Input

- Decision: keep the M05 CuPy matmul gate focused on `a @ b`, with the
  right-hand identity matrix prepared on CPU and copied to the device.
- Reason: CuPy factory kernels such as `eye()` and scalar-multiply setup are a
  separate kernel-layout surface from the second-framework matmul proof. Mixing
  them into the active gate would obscure whether cuBLAS-backed matmul itself
  works.
- Rejected alternatives: treat every CuPy factory/reduction kernel as required
  for M05 closure, or switch away from CuPy after the scoped gate surface was
  already passing import, transfer, elementwise, and matmul.
- Reversal/removal condition: broaden CuPy coverage in a later milestone if the
  client requires full CuPy API compatibility rather than a second-framework
  milestone gate.

## 2026-04-29 - Promote TensorFlow To Active Framework Blocker

- Decision: reopen/broaden Milestone 05 for TensorFlow and promote the observed
  TensorFlow CPU fallback as `M05-E5`.
- Reason: TensorFlow was listed as a candidate framework and later tested by user
  request. It does not inherit validation from PyTorch or CuPy because its GPU
  registration path depends on TensorFlow-specific CUDA/cuDNN/NVML dynamic
  library checks.
- Corrected interpretation: M05's CuPy lane remains valid, but M05 is not closed
  for broadened framework coverage while TensorFlow is failing to register GPU.
- Rejected alternative: move TensorFlow entirely to a later milestone while still
  describing M05 as broadly complete.
- Reversal/removal condition: TensorFlow either passes a bounded GPU training
  gate with mediator evidence and preservation reruns, or the user explicitly
  removes TensorFlow from the required expansion surface.
