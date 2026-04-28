# Active Error - Milestone 04 PyTorch Gate

## Current Lane

Milestone 04: PyTorch Gate

## Current Plan A State

Pass after final M04 serial preservation:

- `/tmp/phase1_milestone_gate_serial_00_after_m04_64k_final.json` ->
  `overall_pass=True`.

## Active Error

None. Milestone 04 active errors are closed.

Evidence:

- `M04-E1` is closed: the VM now has a dedicated 30G host-backed disk mounted at
  `/mnt/m04-pytorch`, and PyTorch `2.5.1+cu121` is installed in
  `/mnt/m04-pytorch/venv`.
- PyTorch CUDA discovery passes:
  `torch.cuda.is_available() == True`, `device_count == 1`, device name
  `HEXACORE vH100 CAP`.
- `M04-E5` is closed: the final PyTorch gate passed matrix multiply, small
  `torch.nn` inference, repeated warm execution, and fresh-process repetition:
  `/tmp/m04_pytorch_probe_64k_final_repeat.json` ->
  `overall_pass=True`, `run_passes=[true,true,true]`.

Impact:

Milestone 04 now exercises PyTorch CUDA discovery, mediated allocation/copy,
elementwise operation, matrix multiply, small `torch.nn` inference, repeated
warm execution, and process restart cleanup on the scoped mediated path.

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

Milestones `00`, `01`, `02`, and `03` are preserved after the final Milestone 04
transport and PyTorch changes, including Plan A, Plan B, Plan C, raw CUDA, API
audit consistency, and the 4 MiB async/mixed memory/sync gate.

## Closed Errors

- `M04-E1`: PyTorch not installed / insufficient VM capacity.
- `M04-E2`: `cuCtxGetStreamPriorityRange` returned
  `CUDA_ERROR_NOT_SUPPORTED` before first tensor copy.
- `M04-E3`: `cuStreamIsCapturing` returned `CUDA_ERROR_NOT_SUPPORTED` before
  first tensor copy.
- `M04-E4`: host mediator crashed with double-free after default-stream
  `cuMemcpyHtoDAsync` staging ownership.
- `M04-E5`: PyTorch matmul/cuBLAS and small inference path failed until the
  scoped cuBLAS/cuBLASLt and PyTorch kernel-layout fixes were deployed.
- M03 preservation regression during M04 closure: repeated 4 MiB BAR1
  `cuMemcpyHtoDAsync` timed out with 256 KiB chunks; closed by reducing mediated
  BAR1 copy chunks to 64 KiB and rerunning the final serial chain.

## Closure Condition

Satisfied by `/tmp/m04_pytorch_probe_64k_final_repeat.json` and the final serial
preservation chain listed in `EVIDENCE.md`.
