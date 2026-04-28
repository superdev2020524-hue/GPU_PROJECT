# Decisions - Milestone 01 General CUDA Gate

## Decision Log

### 2026-04-28 - Start With Raw CUDA, Not Frameworks

- Decision: begin with standalone CUDA tests before PyTorch or TensorFlow.
- Reason: raw CUDA isolates core vGPU behavior and avoids broad framework
  failure surfaces.
- Rejected alternatives: start directly with PyTorch; treat Ollama gates as
  sufficient proof of general CUDA behavior.
- Reversal condition: none for Milestone 01. Frameworks are later milestones.

### 2026-04-28 - Use JSON Gate Output

- Decision: the Milestone 01 runner must produce JSON.
- Reason: JSON gives repeatable machine-readable evidence for later comparison.
- Rejected alternatives: plain terminal-only output.
- Reversal condition: none; plain output can be included as a preview inside the
  JSON report.

### 2026-04-28 - Keep Plan A As First Regression Detector

- Decision: prove Plan A before raw CUDA implementation and after any shared
  runtime change.
- Reason: the Ollama canary is the strongest current end-to-end proof of the
  mediated path.
- Rejected alternatives: rely only on raw CUDA tests after Milestone 01 begins.
- Reversal condition: only if the user explicitly replaces the preserved
  baseline definition.

### 2026-04-28 - `M01-E1` Uses Legacy Parameter Fallback

- Decision: when guest `cuLaunchKernel` cannot get parameter metadata for a
  generic PTX function through `cuFuncGetParamInfo`, preserve the already
  scanned `kernelParams` count and send legacy 8-byte parameter slots instead
  of sending zero launch parameters.
- Reason: the first raw CUDA gate proved module load and function lookup, but
  launch failed because the shim serialized `params=0` after
  `cuFuncGetParamInfo -> 801`.
- Rejected alternatives: disable the simple-kernel gate, treat module load as
  enough for Milestone 01, or special-case the `add_one` test.
- Reversal/removal condition: remove or replace this fallback if it regresses
  Plan A or if a more accurate generic parameter-layout mechanism is added.

### 2026-04-28 - Runtime Shim Must Not Replace Ollama's Real `libcudart`

- Decision: deploy Milestone 01 Runtime shim updates only to `/opt/vgpu/lib`
  unless a separate Ollama compatibility change explicitly requires otherwise.
- Reason: copying the vGPU Runtime shim through Ollama's
  `/usr/local/lib/ollama/cuda_v12/libcudart.so.12` symlink overwrote
  `libcudart.so.12.8.90` and caused a Plan A regression.
- Rejected alternatives: leave the vGPU Runtime shim installed as Ollama's real
  CUDA runtime, or continue Milestone 01 while Plan A was failing.
- Reversal/removal condition: only change Ollama's real Runtime library path
  after a dedicated bounded Ollama compatibility test and immediate Plan A
  proof.

### 2026-04-28 - Runtime API Expansion Is A Separate Probe

- Decision: add `runtime_api_probe.c` alongside the existing Driver API probe
  instead of blending Runtime and Driver calls into one binary.
- Reason: separate probes keep failure classification clear. Driver API stayed
  green while Runtime API exposed the missing `cudaEventCreate` symbol.
- Rejected alternatives: start PyTorch after the Driver API probe; hide Runtime
  API coverage inside the Driver probe.
- Reversal/removal condition: none for Milestone 01. Later framework gates can
  reuse this runner or split it further.
