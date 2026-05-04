# Active Error - Milestone 05 Second Framework Gate

## Current Lane

Milestone 05: Second Framework Gate

## Current Plan A State

**`pass`** after **`M05-E5` TensorFlow closure** on VM-10:
**`/tmp/phase1_after_tf_m05.json`** → **`overall_pass=true`** (checked-in
**`phase1_milestone_gate.py`** + **`phase1_milestone_test_suite.json`**).

## Active Error

None for **`M05-E5`**.

TensorFlow **`2.16.2`** on VM-10 (`/mnt/m04-pytorch/tf123-venv`) completes the
checked-in **`tensorflow_mnist_probe.py`** on **`/GPU:0`** with
**`used_gpu_for_training=true`** and **`overall_pass=true`** (bounded MNIST
slice + manual gradient step).

Evidence:

- `M05-E1` is closed: `cupy-cuda12x==14.0.1` installed in
  `/mnt/m04-pytorch/venv`.
- `M05-E2` is closed: after Runtime and Driver pointer-attribute fixes,
  `tensor_htod_dtoh` passes in the CuPy probe.
- `M05-E3` is closed: CuPy elementwise kernels now pass verified values after
  CuPy-specific launch-parameter layout handling.
- `M05-E4` is closed: CuPy cuBLAS import/handle/version and scoped matmul now
  pass after shim export coverage and the narrowed CPU-prepared input matrix
  gate.
- Final CuPy preservation:
  `/tmp/m05_cupy_preservation_after_m05/run_1.json` through `run_3.json` ->
  `overall_pass=True` for all three runs.
- **`M05-E5` closed (2026-04-29, corrected 2026-04-30):** guest
  **`libvgpu_cuda.c`** launch heuristics for TensorFlow **`EigenMetaKernel`** —
  two-parameter compact ABI for any two-slot kernel; **`TensorAssignOp` +
  `scalar_const_op`** three-parameter kernels use
  **`CUDA_LAUNCH_PARAM_MODE_RAW_BUFFER`** with compact **`8+4+4`** layout
  (`device pointer`, scalar float, element count). The earlier legacy
  pointer-array path and later over-broad **`8+8+8`** raw layout caused
  **`CUDA_ERROR_ILLEGAL_ADDRESS`**, primary-context recovery
  (**`Recovering primary context`**), and downstream
  **`CUDA_ERROR_CONTEXT_IS_DESTROYED`** on **`cuModuleLoadData`**.

Impact:

Milestone 05 now proves **CuPy** and **TensorFlow** lanes for GPU discovery,
HtoD/DtoH, and bounded GPU execution through the mediated stack for the
checked-in probes.

## Candidate Queue

- Additional CuPy factory/reduction kernels may require more kernel parameter
  layout handling beyond the scoped gate.
- Additional TensorFlow/Eigen kernel templates beyond the exercised MNIST-style
  graph may still need explicit launch layouts.
- ONNX Runtime and Numba remain alternate candidates for later expansion.

## Closed Errors

- `M05-E1`: CuPy missing from isolated VM environment.
- `M05-E2`: CuPy host-pointer classification failed because
  `cudaPointerGetAttributes` / `cuPointerGetAttributes` returned unsupported.
- `M05-E3`: CuPy elementwise add returned wrong output due to missing
  CuPy-specific kernel parameter layout handling.
- `M05-E4`: CuPy matmul/cuBLAS path failed due to dynamic-linking/export/version
  issues and was then narrowed to the scoped CPU-prepared matrix input gate.
- **`M05-E5`**: TensorFlow GPU registration / bounded GPU training — closed per
  VM-10 probe and host mediator evidence above.

## Last Proven Checkpoint

**`M05-E5`**: VM-10 TensorFlow probe JSON **`overall_pass=true`** with mediated
GPU; dom0 **`/tmp/mediator.log`** shows successful **`cuLaunchKernel`** on
TensorFlow Eigen kernels and **no** primary-context recovery line for that run.

Full serial preservation **00–07** after this shim release was **not** executed
end-to-end in this session—required before interpreting broader milestone
closure.

## Closure Condition

Met for **`M05-E5`**: TensorFlow registers **`GPU:0`**, runs the bounded probe on
GPU, and mediator logs show successful GPU kernel execution after shim fixes.

Serial **Plan A** / milestones **00–07** preservation remains a separate explicit
rerun after guest-ship deploy per **`VERIFICATION_RULES.md`**.
