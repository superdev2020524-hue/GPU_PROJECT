# 03 - Memory, Synchronization, And Cleanup

## Purpose

Harden the vGPU layer for general process behavior, not just Ollama's process
shape.

## Scope

- allocation ownership;
- host-device pointer lifetime;
- HtoD/DtoH integrity;
- async copy behavior;
- streams and events;
- process exit cleanup;
- stale BAR1 or SHMEM handling;
- recovery after host CUDA errors.

## Closure Criteria

- repeated CUDA gates do not leak memory;
- forced process kill does not poison the next run;
- stream and event tests pass;
- large copy tests pass;
- mixed sync/async tests pass;
- mediator can identify and clean per-process state.

## Current Status

Milestone 03 is complete.

Initial entry condition:

- Milestone 02 is complete.
- Preserved Plan A passed before Milestone 03 work:
  `/tmp/phase1_milestone_gate_before_m03.json` -> `overall_pass=True`.

Closure evidence:

- Normal CUDA process cleanup is process-scoped and host-proven.
- Forced `SIGKILL` cleanup is handled by guest stale-owner sweep and host-proven.
- Async HtoD, DtoH, DtoD, memset, stream sync, event sync/query, and
  stream-wait behavior passed a repeated bounded gate.
- The async/mixed gate now verifies 4 MiB per fresh process and passes 5/5 runs.
- Final Plan A passed:
  `/tmp/phase1_milestone_gate_m03_final_after_chunking.json`.
- Final raw CUDA gate passed:
  `/tmp/phase3_general_cuda_gate_m03_final_after_chunking.json`.
