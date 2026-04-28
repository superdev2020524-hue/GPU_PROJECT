# Baseline - Milestone 01 General CUDA Gate

## Scope

- Milestone: `01_general_cuda_gate`
- Folder: `phase3/VERIFICATION/01_general_cuda_gate/`
- Roadmap reference: `PHASE3_GENERAL_GPU_VIRTUALIZATION_ROADMAP.md`, Milestone 1
- Work type: runtime gate
- Owner role retained: Phase 3 assistant role, non-destruction rule, one-active-error discipline

## Required Preserved Baseline

This milestone touches shared vGPU runtime behavior. Before implementation and
after risky runtime changes, preserve at least:

- `Plan A`: `qwen2.5:0.5b` checked-in canary.

Recheck `Plan B` or `Plan C` only if a change can affect Tiny behavior,
client-facing Ollama behavior, service configuration, model/runtime libraries,
or cross-model residency.

## Last Known Full Re-Baseline

- Date: 2026-04-27
- Source: `../ERROR_TRACKING_STATUS.md`, session "roadmap re-baseline before general vGPU work"
- Plan A: pass
- Plan B: pass
- Plan C: pass
- Active Phase 1 preservation error: none
- Candidate: residual non-terminating `0x00bc` / `cuFuncGetParamInfo` status noise

## Current Session Baseline

- Date: 2026-04-27 host/VM local time, started from user request on 2026-04-28 KST.
- Plan A state: pass.
- Report path: `/tmp/phase1_milestone_gate_before_m01.json`.
- `/api/ps` state: clean after Plan A force-unload (`{"models":[]}`).
- Host mediator state: running as `89028 ./mediator_phase3`.
- VM `ollama` state: active.
- vGPU PCI proof: VM `lspci -nn` shows `00:05.0 3D controller [0302]: NVIDIA Corporation Device [10de:2331] (rev a1)`.
- Host artifact proof: live `qemu-dm-1` argv includes `-device vgpu-cuda,pool_id=A,priority=medium,vm_id=10`.
- VM artifact proof: `/opt/vgpu/lib/libvgpu-cuda.so.1`, `libvgpu-cudart.so`, `libvgpu-cublas.so.12`, `libvgpu-cublasLt.so.12`, and `libvgpu-nvml.so` are present.
- Host evidence summary: `/tmp/mediator.log` contains `86928` physical GPU kernel-success lines after Plan A, with `sync FAILED count=0` and `CUDA_ERROR_ILLEGAL_ADDRESS count=0`.
- Candidate noise: `0x00bc` / `cuFuncGetParamInfo` remains candidate-only because Plan A completed successfully.

## Latest Baseline Recheck

- Date: 2026-04-27 host/VM local time, after Runtime API expansion and recovery.
- Plan A state: pass.
- Report path: `/tmp/phase1_milestone_gate_after_runtime_restore_attempt.json`.
- Reason: required recheck after `libvgpu-cudart.so` was changed and deployed.
- Recovery note: Plan A briefly regressed because the Runtime shim deployment
  overwrote Ollama's real CUDA Runtime target. The overwritten file was
  preserved as
  `/usr/local/lib/ollama/cuda_v12/libcudart.so.12.8.90.bad-vgpu-cudart-20260427-1751`,
  the Ollama runtime path was restored, and Plan A returned to pass.

## Live Artifact Requirements

Record these before interpreting any Milestone 01 gate result:

- Host mediator path and process.
- Host mediator log path and current timestamp.
- Host CUDA device identity.
- Live QEMU vGPU socket path.
- VM vGPU PCI device.
- VM shim library path.
- VM service state.

## Baseline Failure Rule

If `Plan A` fails, stop Milestone 01. The regression becomes the active error
until repaired. Do not interpret raw CUDA gate behavior on a failed preserved
baseline.
