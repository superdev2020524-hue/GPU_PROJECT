# Baseline - Milestone 06 Multi-Process And Multi-VM

Milestone 06 starts after the scoped Milestone 05 CuPy second-framework closure.

## Preserved Entry State

- Plan A passed at M05 entry:
  `/tmp/phase1_milestone_gate_serial_00_after_m04_64k_final.json` ->
  `overall_pass=True`.
- Plan B passed at M05 entry:
  `/tmp/phase1_plan_b_serial_00_after_m04_64k_final.json` ->
  `overall_pass=True`.
- Plan C passed at M05 entry:
  `/tmp/phase1_plan_c_serial_00_after_m04_64k_final.json` ->
  `overall_pass=True`.
- Milestone 01 raw CUDA passed at M05 entry:
  `/tmp/phase3_general_cuda_gate_serial_01_after_m04_64k_final.json` ->
  `overall_pass=True`.
- Milestone 02 API audit passed at M05 entry:
  `/tmp/phase3_api_audit_serial_02_after_m04_64k_final.json` ->
  `overall_pass=True`.
- Milestone 03 preservation after M05 passed:
  `/tmp/m03_preservation_after_m05_300/run_1.txt` through `run_3.txt` ->
  3/3 pass.
- Milestone 04 PyTorch preservation after M05 passed:
  `/tmp/m04_pytorch_preservation_after_m05/run_1.json` through `run_3.json` ->
  3/3 pass.
- Milestone 05 CuPy preservation passed:
  `/tmp/m05_cupy_preservation_after_m05/run_1.json` through `run_3.json` ->
  3/3 pass.

## Live Artifact Proof

- `/opt/vgpu/lib/libvgpu-cuda.so.1` sha256:
  `51c47c7104a45d62773ff11481af0e33fb92fc2c7a1795bc29c6d71c6eb3a1ba`.
- `/opt/vgpu/lib/libvgpu-cudart.so` sha256:
  `364832a3aad55ffe297a0b96858d45d6fa3e51fd1badbeef04382f24e04611a4`.
- `/opt/vgpu/lib/libvgpu-cublas.so.12` sha256:
  `5202e665bac786aa648b44ce1e3ee68ea48ecbaa5fd7e6425a68f428ef160f73`.
- `/mnt/m04-pytorch/venv` contains:
  `torch==2.5.1+cu121`, `cupy==14.0.1`.

## Baseline Interpretation

Single-process Ollama, raw CUDA, PyTorch, and CuPy gates are green. Milestone 06
must not reinterpret later concurrency failures as single-workload failure unless
one of these baselines regresses under a bounded repro.
