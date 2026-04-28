# Baseline - Milestone 03 Memory, Synchronization, And Cleanup

## Scope

- Milestone: `03_memory_sync_cleanup`
- Folder: `phase3/VERIFICATION/03_memory_sync_cleanup/`
- Roadmap reference: `PHASE3_GENERAL_GPU_VIRTUALIZATION_ROADMAP.md`,
  Milestone 3
- Work type: runtime hardening and bounded gate expansion
- Required preserved lane: Plan A

## Preserved Baseline

Milestone 03 starts from the closed Milestone 02 baseline.

- Plan A state: pass
- Latest Plan A report:
  `/tmp/phase1_milestone_gate_before_m03.json`
- Latest Milestone 01 raw CUDA gate:
  `/tmp/phase3_general_cuda_gate_after_m02_e3_live_binary_restart.json`
- Active Milestone 02 error: none
- Active Milestone 03 error: none until the initial audit promotes one

## Baseline Meaning

Plan A must remain green before and after any Milestone 03 change touching:

- host mediator or executor;
- guest CUDA, Runtime, cuBLAS, cuBLASLt, or NVML shims;
- transport/protocol structures;
- service runtime configuration;
- allocation, stream, event, process cleanup, or error recovery behavior.

If Plan A regresses, Milestone 03 work stops and the Plan A regression becomes
the active error.

## Known Carry-Forward Candidates

- Residual `cuFuncGetParamInfo(0x00bc)` unsupported/invalid-value noise.
- BAR1 fallback after shmem GPA resolution fails with `pfn_hidden`.
- Runtime shim deployment scope: `/usr/lib64/libcudart.so.12` points to the
  vGPU Runtime shim.
- Remaining lower-priority silent/synthetic behavior in Runtime status queries,
  Driver optional exports, and NVML.
- Ollama-shaped device/property fallbacks that may not generalize to frameworks.
- cuBLAS GEMM numerical correctness still needs a later focused gate.
