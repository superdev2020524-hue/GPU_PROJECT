# Evidence - Milestone 05 Second Framework Gate

## Baseline Evidence

Milestone 05 starts from the final Milestone 04 closure chain:

- Plan A:
  `/tmp/phase1_milestone_gate_serial_00_after_m04_64k_final.json` ->
  `overall_pass=True`.
- Plan B:
  `/tmp/phase1_plan_b_serial_00_after_m04_64k_final.json` ->
  `overall_pass=True`.
- Plan C:
  `/tmp/phase1_plan_c_serial_00_after_m04_64k_final.json` ->
  `overall_pass=True`.
- Milestone 01 raw CUDA:
  `/tmp/phase3_general_cuda_gate_serial_01_after_m04_64k_final.json` ->
  `overall_pass=True`.
- Milestone 02 API audit:
  `/tmp/phase3_api_audit_serial_02_after_m04_64k_final.json` ->
  `overall_pass=True`.
- Milestone 03 memory/sync:
  `/tmp/async_stream_event_probe_after_m04_64k_final.json` ->
  `overall_pass=True`, `pass_count=3`, `runs=3`.
- Milestone 04 PyTorch:
  `/tmp/m04_pytorch_probe_64k_final_repeat.json` ->
  `overall_pass=True`, `run_passes=[true,true,true]`.

## Evidence To Collect

- VM Python environments and package locations.
- Whether CuPy is already installed.
- If CuPy imports, whether it sees the mediated CUDA device.
- If CuPy runs, value-checked transfer, elementwise, and matmul behavior.
- Host mediator health during the probe.

## Current Session Findings

Initial CuPy environment probe:

- Python path:
  `/mnt/m04-pytorch/venv/bin/python`.
- Python version:
  `3.10.12`.
- Probe report:
  `/tmp/m05_cupy_env_probe.json`.
- Result:
  `cupy_spec=null`, `cupy_import_ok=false`,
  `error="ModuleNotFoundError: cupy"`.
- Capacity check:
  `/mnt/m04-pytorch` has `23G` free and the isolated venv has working `pip`.

Active error promoted:

- `M05-E1`: CuPy is not installed in the isolated VM Python environment.

## CuPy Install And Pointer Attribute Fixes

- Installed `cupy-cuda12x==14.0.1` into `/mnt/m04-pytorch/venv`.
- First repeated CuPy gate:
  `/tmp/m05_cupy_probe_repeat.json` ->
  `overall_pass=False`, `run_passes=[false,false,false]`.
- `M05-E2` promoted:
  `tensor_htod_dtoh` failed in CuPy host-memory pin detection because
  `pointerGetAttributes` returned `cudaErrorNotSupported`.
- Runtime shim fix:
  `guest-shim/libvgpu_cudart.c` now exports `cudaPointerGetAttributes` and
  `cudaPointerGetAttributes_v2` for ordinary host pointer classification.
- Driver shim fix:
  `guest-shim/libvgpu_cuda.c` now exports narrow `cuPointerGetAttribute` and
  `cuPointerGetAttributes` support for the pointer attributes CuPy probes.
- Live artifact proof:
  - `/opt/vgpu/lib/libvgpu-cudart.so` sha256
    `364832a3aad55ffe297a0b96858d45d6fa3e51fd1badbeef04382f24e04611a4`.
  - `/home/test-10/phase3/guest-shim/libvgpu_cudart.c` sha256
    `2ee0033ee61989c8a6f0d0f3d9d209f59c19ce8cc560dc57df3849df06ce0805`.
  - `/opt/vgpu/lib/libvgpu-cuda.so.1` sha256
    `86b68cb7408701cb7e7ec6e834f41a56659fa062a435d0bc61c29c1880b14a5b`.
  - `/home/test-10/phase3/guest-shim/libvgpu_cuda.c` sha256
    `17d1466c2f211794567c1061dd7173aea06a641cf5ec65a954fd9608973b9f7d`.
  - Ollama CUDA v12 Runtime path still resolves to
    `/usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90`, not the vGPU
    Runtime shim.
- Repeated CuPy gate after pointer-attribute fixes:
  `/tmp/m05_cupy_probe_after_cuptrattrs_repeat.json` ->
  `overall_pass=False`, `run_passes=[false,false,false]`.
- Closed behavior:
  CuPy import, device count, device name, and `tensor_htod_dtoh` now pass.
- Active error promoted:
  `M05-E3`: `elementwise_add` returns mostly zeros instead of verified values.
- Candidate-only:
  CuPy matmul fails with `CUBLAS_STATUS_INTERNAL_ERROR`, but this is later than
  the elementwise wrong-output blocker.

## Elementwise And cuBLAS Fixes

- `M05-E3` closed by adding CuPy-specific launch-parameter layout handling in
  `guest-shim/libvgpu_cuda.c` for the observed `cupy_arange` and `cupy_add`
  kernels.
- `M05-E4` closed by forcing CuPy's cuBLAS extension away from the backed-up
  vendor library path, exporting the CuPy-required cuBLAS symbols from
  `guest-shim/libvgpu_cublas.c`, and implementing `cublasGetVersion_v2`.
- The scoped CuPy matmul gate prepares the right-hand matrix on CPU and transfers
  it to the device, keeping unrelated CuPy factory kernels out of this milestone
  closure.

## Closure Evidence After Serial Preservation

- Live framework packages:
  - `torch==2.5.1+cu121`, CUDA build `12.1`.
  - `cupy==14.0.1`.
- Live artifact proof:
  - `/opt/vgpu/lib/libvgpu-cuda.so.1` sha256
    `51c47c7104a45d62773ff11481af0e33fb92fc2c7a1795bc29c6d71c6eb3a1ba`.
  - `/opt/vgpu/lib/libvgpu-cudart.so` sha256
    `364832a3aad55ffe297a0b96858d45d6fa3e51fd1badbeef04382f24e04611a4`.
  - `/opt/vgpu/lib/libvgpu-cublas.so.12` sha256
    `5202e665bac786aa648b44ce1e3ee68ea48ecbaa5fd7e6425a68f428ef160f73`.
  - `/mnt/m04-pytorch/cupy_probe.py` sha256
    `cbbcba3ee14396de3d390faba1465510111b26be76cc8554f68940bfcfced403`.
  - `/mnt/m04-pytorch/pytorch_probe.py` sha256
    `f56eb37aab9896d41499bf5d064d95a3f11be4c35f1a117f647b747fd79d7e61`.
- Milestone 03 preservation after the temporary post-M05 timeout candidate:
  `/tmp/m03_preservation_after_m05_300/run_1.txt` through `run_3.txt` ->
  3/3 pass, elapsed `134.9s`, `178.2s`, and `125.9s`.
- Interpretation of the temporary M03 timeout:
  the previous 120s bound timed out while a stale owner was present; after
  mediator restart, stale-owner cleanup was observed and the same 64 KiB BAR1
  chunked 4 MiB async stream/event probe passed under the observed 300s envelope.
- Milestone 04 preservation:
  `/tmp/m04_pytorch_preservation_after_m05/run_1.json` through `run_3.json` ->
  3/3 pass, elapsed `86.8s`, `81.5s`, and `82.4s`.
- Milestone 05 CuPy preservation:
  `/tmp/m05_cupy_preservation_after_m05/run_1.json` through `run_3.json` ->
  3/3 pass, elapsed `6.0s`, `5.6s`, and `5.5s`.

The CuPy lane of Milestone 05 is complete for the scoped CuPy second-framework
gate.

## 2026-04-29 TensorFlow Correction

TensorFlow was tested after the user requested a different framework problem for
the next verification step.

Observed VM-10 TensorFlow behavior:

- TensorFlow imported and executed a bounded MNIST/MNIST-shaped training probe.
- The probe did not report TensorFlow GPU registration.
- Training completed in CPU mode, not GPU mode.
- TensorFlow logs indicated GPU registration dependency failures, including:
  `Failed to determine cuDNN version`, `NVML library doesn't have required
  functions`, `Cannot dlopen some GPU libraries`, and
  `Skipping registering GPU devices`.

Correction to prior interpretation:

- M04 proved PyTorch GPU behavior only.
- M05 proved the scoped CuPy lane only.
- M06 and M07 preservation relied on the already-scoped PyTorch/CuPy/Ollama/raw
  CUDA evidence; they did not add TensorFlow coverage.
- Any statement implying "general framework compatibility" beyond those scoped
  gates was too broad.

Active error promoted:

- `M05-E5`: TensorFlow trains on CPU because TensorFlow does not register the
  mediated GPU.

Current earliest classified cause:

- TensorFlow has a stricter GPU registration stack than PyTorch and CuPy. The
  observed failure is currently at TensorFlow's CUDA/cuDNN/NVML dynamic-library
  registration layer before a TensorFlow GPU training workload reaches the same
  kind of mediated compute path already proven by PyTorch/CuPy.
