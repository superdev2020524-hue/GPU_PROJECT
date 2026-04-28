# Active Error - Milestone 04 PyTorch Gate

## Current Lane

Milestone 04: PyTorch Gate

## Current Plan A State

Pass at entry:

- `/tmp/phase1_milestone_gate_serial_00_after_planc_fix.json` ->
  `overall_pass=True`.

## Active Error

`M04-E5`: PyTorch matmul fails at cuBLAS handle creation.

Evidence:

- `M04-E1` is closed: the VM now has a dedicated 30G host-backed disk mounted at
  `/mnt/m04-pytorch`, and PyTorch `2.5.1+cu121` is installed in
  `/mnt/m04-pytorch/venv`.
- PyTorch CUDA discovery passes:
  `torch.cuda.is_available() == True`, `device_count == 1`, device name
  `HEXACORE vH100 CAP`.
- Repeated adjusted gate runs pass:
  CPU-to-CUDA copy, CUDA-to-CPU copy, and PyTorch elementwise add with verified
  values.
- Repeated adjusted gate runs fail at:
  `RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling
  cublasCreate(handle)` during `matmul = (a @ b).cpu()`.
- PyTorch resolves bundled cuBLAS from the venv:
  `/mnt/m04-pytorch/venv/.../nvidia/cublas/lib/libcublas.so.12`, not the
  `/opt/vgpu/lib/libvgpu-cublas.so.12` shim, under normal gate execution.

Impact:

Milestone 04 can now exercise PyTorch CUDA discovery, mediated allocation/copy,
and one PyTorch CUDA elementwise kernel, but it cannot yet pass the required
matrix multiply or small `torch.nn` inference coverage.

## Candidate Queue

- PyTorch CUDA factory/fill kernels such as `torch.eye(device=...)` and direct
  CUDA range/fill creation still expose unsupported kernel-parameter layouts and
  remain candidate-only because the bounded M04 gate now creates test data on CPU
  before transfer.
- A cuBLAS preload path using `/opt/vgpu/lib/libvgpu-cublas.so.12` hung during
  initialization and is not a valid fix as tested.
- PyTorch may later require cuDNN, NCCL, or broader allocator behavior not yet
  covered by the current bounded gate.

## Last Proven Checkpoint

Milestones `00`, `01`, `02`, and `03` are preserved at Milestone 04 entry,
including Plan A, Plan B, Plan C, raw CUDA, API audit consistency, and 4 MiB
async/mixed memory/sync gate.

## Closed Errors

- `M04-E1`: PyTorch not installed / insufficient VM capacity.
- `M04-E2`: `cuCtxGetStreamPriorityRange` returned
  `CUDA_ERROR_NOT_SUPPORTED` before first tensor copy.
- `M04-E3`: `cuStreamIsCapturing` returned `CUDA_ERROR_NOT_SUPPORTED` before
  first tensor copy.
- `M04-E4`: host mediator crashed with double-free after default-stream
  `cuMemcpyHtoDAsync` staging ownership.

## Closure Condition

Close `M04-E5` only after the bounded PyTorch gate passes matrix multiply and
small `torch.nn` inference with verified values, repeated fresh-process runs,
and mediator evidence showing no crash, no `CUDA_ERROR_ILLEGAL_ADDRESS`, no
poisoned follow-on context, and no hidden fake-success output mismatch.
