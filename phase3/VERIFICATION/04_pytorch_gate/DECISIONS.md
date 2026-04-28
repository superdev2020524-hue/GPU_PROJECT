# Decisions - Milestone 04 PyTorch Gate

## 2026-04-29 - Start With Environment Probe

- Decision: begin Milestone 04 with a bounded PyTorch environment probe before
  changing vGPU runtime code.
- Reason: PyTorch may be absent or may fail at import/device-discovery before any
  framework operation reaches the mediated CUDA path.
- Rejected alternatives: install packages immediately; start by modifying CUDA
  shims for guessed PyTorch requirements; run a broad training or benchmark.
- Reversal/removal condition: if PyTorch is installed and reaches CUDA but fails
  on a specific API call, promote that exact call or behavior as the active
  Milestone 04 error.

## 2026-04-29 - Preserve Prior Milestones Separately

- Decision: Milestone 04 evidence must report `00`, `01`, `02`, and `03`
  preservation separately from PyTorch evidence.
- Reason: the project goal is staged verification and expansion, not replacing
  earlier passing behavior with a new passing behavior.
- Rejected alternatives: cite only the PyTorch gate; rely only on historical
  closure records.
- Reversal/removal condition: none for Phase 3 milestone closure.

## 2026-04-29 - Do Not Install PyTorch Into A 97% Full Root Filesystem

- Decision: do not attempt a CUDA-enabled PyTorch installation on the current VM
  root filesystem.
- Reason: `/` has only `1.2G` free and no local PyTorch wheel/cache exists.
  Installing PyTorch plus dependencies could fill the disk and risk damaging the
  preserved Ollama/CUDA baseline.
- Rejected alternatives: install CPU-only PyTorch and treat it as the PyTorch GPU
  gate; delete Ollama/model/runtime assets to make room; install directly into
  system Python without an isolated environment.
- Reversal/removal condition: proceed when there is enough dedicated capacity
  for PyTorch, such as an expanded VM disk or mounted external package location,
  and the install can be isolated and documented.

## 2026-04-29 - Add Dedicated Host-Backed PyTorch Disk

- Decision: add a separate 30G VDI to the running Test-10 VM and mount it at
  `/mnt/m04-pytorch` for the PyTorch environment.
- Reason: safe cleanup improved root capacity but still could not complete the
  CUDA-enabled PyTorch install; deleting Ollama/model assets would risk the
  preserved baseline.
- Rejected alternatives: remove Ollama models, install CPU-only PyTorch, keep
  retrying on the root filesystem, or create a new VM.
- Reversal/removal condition: none for this M04 environment; the VDI is isolated
  from root and persisted via `/etc/fstab`.

## 2026-04-29 - Keep PyTorch Factory Kernels Candidate-Only

- Decision: the bounded M04 gate creates test tensors on CPU, transfers them to
  CUDA, and then tests required mediated operations.
- Reason: the documented gate requires transfer, elementwise, matmul, and small
  inference; CUDA-side factory/fill kernels exposed a separate kernel-layout
  class that should not displace the current active cuBLAS matmul blocker.
- Rejected alternatives: count `torch.eye(device=...)` as the active gate
  blocker; ignore the factory/fill failure entirely.
- Reversal/removal condition: promote the factory/fill candidate after the
  current matmul/cuBLAS blocker is closed or disproved.

## 2026-04-29 - Do Not Use cuBLAS Preload Hang As A Fix

- Decision: do not adopt `LD_PRELOAD=/opt/vgpu/lib/libvgpu-cublas.so.12` as the
  PyTorch matmul fix.
- Reason: the preload experiment hung during cuBLAS initialization and did not
  produce a bounded successful matmul result.
- Rejected alternatives: force the preload into the gate despite the hang; treat
  a hung initialization as progress.
- Reversal/removal condition: only reconsider after a separate bounded cuBLAS
  shim compatibility fix proves import, handle creation, GEMM, and process
  cleanup without hanging.

## 2026-04-29 - Close M04 With Scoped PyTorch Gate

- Decision: close Milestone 04 on the bounded PyTorch gate after matrix multiply,
  small `torch.nn` inference, repeated warm execution, and 3/3 fresh-process runs
  pass.
- Reason: the gate covers the documented M04 scope without mixing in separate
  PyTorch CUDA factory/fill and reduction-kernel families.
- Rejected alternatives: keep M04 open for every PyTorch CUDA kernel family; call
  M04 closed before rerunning the serial preservation chain.
- Reversal/removal condition: reopen only if a later final preservation run on
  the same deployed artifacts fails one of the required M04 cases or regresses a
  prior milestone.

## 2026-04-29 - Reduce BAR1 Copy Chunks For Preservation

- Decision: cap mediated BAR1 copy chunks at 64 KiB for HtoD/DtoH copy paths.
- Reason: final serial preservation found a repeated M03 4 MiB async-copy timeout
  with 256 KiB chunks; 64 KiB chunks passed 3/3 and preserved the final M04 gate.
- Rejected alternatives: accept 2/3 M03 pass as sufficient; restart the mediator
  and claim closure without a code fix; widen the timeout instead of reducing
  the burst size.
- Reversal/removal condition: only raise the cap again after a dedicated BAR1
  throughput/stability gate proves repeated large async copies without stalls.
