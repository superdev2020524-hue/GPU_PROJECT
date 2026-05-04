# Gate - Milestone 05 Second Framework Gate

## Candidate Runtime

Initial completed candidate: CuPy.

CuPy is selected first because it is an independent CUDA Python stack and can
exercise CUDA discovery, allocation, HtoD/DtoH transfer, elementwise kernels, and
GEMM without the larger install/runtime footprint of TensorFlow.

Broadened required candidate after 2026-04-29 correction: TensorFlow.

TensorFlow must not be inferred from PyTorch or CuPy. It has its own CUDA,
cuDNN, NVML, and dynamic-library registration path and requires direct evidence.

## Required Coverage

- import selected runtime;
- report one CUDA device through the mediated path;
- allocate a small GPU array;
- transfer CPU data to GPU and back;
- run one elementwise operation with verified values;
- run one matrix multiply with verified values;
- repeat the gate in fresh processes;
- preserve prior milestones after any runtime or shim change.

## TensorFlow-Specific Required Coverage

- import TensorFlow from an isolated VM environment;
- prove `tf.sysconfig.get_build_info()` is a CUDA build;
- prove `tf.config.list_physical_devices("GPU")` and/or logical GPU listing
  contains the mediated GPU;
- run a bounded MNIST or MNIST-shaped training step with TensorFlow assigning
  work to GPU, not silently falling back to CPU;
- capture TensorFlow stderr for CUDA/cuDNN/NVML/dlopen registration errors;
- capture host mediator evidence showing TensorFlow-generated GPU work, or
  explicitly classify the earliest missing registration dependency.

## Pass Criteria

- all required cases pass with value checks;
- failures are explicit and not fake-success output mismatches;
- no mediator crash, `CUDA_ERROR_ILLEGAL_ADDRESS`, `sync FAILED`, unsupported
  protocol regression, or poisoned follow-on context;
- final serial preservation passes before closure.

## Fail Criteria

- runtime unavailable and cannot be installed safely in an isolated location;
- CUDA device unavailable to the runtime;
- TensorFlow trains but reports no physical/logical GPU;
- any required value check fails;
- any prior milestone regresses.
