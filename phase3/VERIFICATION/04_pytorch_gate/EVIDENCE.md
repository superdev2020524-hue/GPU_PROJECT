# Evidence - Milestone 04 PyTorch Gate

## Baseline Evidence

Milestone 04 begins after the serial preservation correction:

- Plan A passed after Plan C fix:
  `/tmp/phase1_milestone_gate_serial_00_after_planc_fix.json` ->
  `overall_pass=True`.
- Plan B passed after Plan C fix:
  `/tmp/phase1_plan_b_serial_00_after_planc_fix.json` ->
  `overall_pass=True`.
- Plan C passed after stdin CLI gate fix:
  `/tmp/phase1_plan_c_serial_00_after_m03_fixed_clean.json` ->
  `overall_pass=True`.
- Raw CUDA passed after Milestone 03:
  `/tmp/phase3_general_cuda_gate_serial_01_after_m03.json` ->
  `overall_pass=True`.
- API audit consistency passed after Milestone 03:
  `/tmp/phase3_api_audit_serial_02_after_m03.json` ->
  `overall_pass=True`.
- Milestone 03 4 MiB async/mixed gate passed:
  `/tmp/async_stream_event_probe_repeat.json` -> `overall_pass=True`.

## Evidence To Collect

- VM Python and PyTorch version/install state.
- PyTorch CUDA build metadata if installed.
- `torch.cuda.is_available()` result and error details.
- Host mediator evidence during any PyTorch CUDA attempt.
- VM journal or stderr evidence for import/runtime failures.
- Live artifact proof before interpreting any runtime result.

## Current Session Findings

Milestone 04 has started.

Initial environment probe:

- VM Python:
  `/usr/bin/python3`, Python `3.10.12`.
- `python3 -m pip show torch`:
  failed because `/usr/bin/python3` has no `pip` module.
- `python3 -m pip list`:
  failed for the same reason.
- Direct import probe with `LD_LIBRARY_PATH=/opt/vgpu/lib`:
  `ModuleNotFoundError: No module named 'torch'`.
- Guest vGPU libraries are present under `/opt/vgpu/lib`, including
  `libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12`, `libcublasLt.so.12`,
  and `libnvidia-ml.so.1`.
- Safe-install check:
  - OS: Ubuntu `22.04.5 LTS`.
  - `python3 -m venv` exists.
  - `python3 -m ensurepip` is unavailable.
  - `pip`, `pip3`, `virtualenv`, `conda`, and `micromamba` are not present.
  - `apt` and `apt-get` are present.
  - Root filesystem: `39G` total, `36G` used, `1.2G` free, `97%` used.
  - `/dev/shm`: `2.0G` tmpfs free.
  - `/usr/local/lib/ollama`: `5.4G`; this contains model/runtime baseline
    assets and must not be cleaned as part of PyTorch setup without explicit
    preservation planning.
  - `/var/cache/apt`: `216M`; `/var/lib/apt/lists`: `366M`.
  - No local `torch*.whl` or PyTorch cache was found.
- Host mediator entry health:
  `sync FAILED: count=0`,
  `CUDA_ERROR_ILLEGAL_ADDRESS: count=0`,
  `Unsupported CUDA protocol call: count=0`,
  `invalid handle: count=0`.

Active error promoted:

- `M04-E1`: PyTorch is not installed in the VM Python environment, and the VM
  does not currently have enough safe free disk for a CUDA-enabled PyTorch
  installation.

## Next Single Step

Provide or create safe package capacity for PyTorch, preferably by expanding the
VM disk or mounting an external package location, then install PyTorch in an
isolated environment and rerun the probe.

## Capacity Fix And PyTorch Install

- Safe cleanup freed disposable data only:
  systemd journal vacuumed from `2.7G` to about `224M`, apt cache/lists cleaned,
  and user Go/build caches removed. Root improved from `1.2G` free to about
  `5.3G` free.
- A direct isolated install on root still failed during package installation with
  `OSError: [Errno 28] No space left on device`.
- Host local SR had sufficient free capacity. A new 30G VDI
  `m04-pytorch-env-disk` was created on Local storage, attached live to Test-10
  as `xvdb`, formatted ext4, mounted at `/mnt/m04-pytorch`, and persisted in
  `/etc/fstab`.
- PyTorch environment:
  `/mnt/m04-pytorch/venv`, `torch==2.5.1+cu121`.
- Environment probe result:
  `torch.cuda.is_available() == True`, `device_count == 1`, device name
  `HEXACORE vH100 CAP`, CUDA build `12.1`.

## Runtime Fixes During M04

- `M04-E2` fixed in `guest-shim/libvgpu_cuda.c`:
  `cuCtxGetStreamPriorityRange()` now reports a degenerate supported priority
  range (`least=0`, `greatest=0`) instead of unsupported.
- `M04-E3` fixed in `guest-shim/libvgpu_cuda.c`:
  `cuStreamIsCapturing()` now reports `CU_STREAM_CAPTURE_STATUS_NONE` instead
  of unsupported.
- `M04-E4` fixed in `src/cuda_executor.c`:
  default-stream `cuMemcpyHtoDAsync` no longer enqueues a staging buffer before
  immediately synchronizing and freeing it. This closed the mediator
  `double free or corruption (fasttop)` crash.
- PyTorch elementwise kernel layout fixed in `guest-shim/libvgpu_cuda.c`:
  function-name keyed fallback layout for PyTorch
  `vectorized_elementwise_kernel(... Array<char*,2>)` now uses
  `int32 N @ offset 0`, functor @ offset `4`, and `Array<char*,2>` @ offset
  `16`. This changed elementwise add from wrong-output to verified pass.

## Current PyTorch Gate Evidence

- Bounded probe script:
  `phase3/tests/pytorch_gate/pytorch_probe.py`, deployed to
  `/mnt/m04-pytorch/pytorch_probe.py`.
- Adjusted repeated gate reports:
  `/tmp/m04_pytorch_probe_adjusted_1.json`,
  `/tmp/m04_pytorch_probe_adjusted_2.json`,
  `/tmp/m04_pytorch_probe_adjusted_3.json`,
  summary `/tmp/m04_pytorch_probe_adjusted_repeat.json`.
- Repeated adjusted gate status:
  `run_passes=[false,false,false]`.
- Passing cases in each run:
  CUDA available, device count/name, CPU-to-CUDA copy, CUDA-to-CPU copy, and
  PyTorch elementwise add with verified values.
- Active failure:
  `RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR when calling
  cublasCreate(handle)` at `matmul = (a @ b).cpu()`.
- Library proof:
  `ldd libtorch_cuda.so` resolves PyTorch bundled
  `nvidia/cublas/lib/libcublas.so.12` and `libcublasLt.so.12`.
- Candidate-only observation:
  direct CUDA factory/fill kernels (`torch.eye(device=...)`, CUDA range/fill)
  can still trigger unsupported PyTorch kernel layouts; these are not mixed into
  the current active cuBLAS blocker.
