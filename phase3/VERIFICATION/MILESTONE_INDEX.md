# Milestone Index

This index maps the Phase 3 roadmap to verification folders.

## Latest 00-07 Re-Verification

Date: 2026-05-04

Status: **Milestones 00 through 07 pass on the live system for the documented
gate scope.**

Readable report:
`REVERIFY_00_07_REPORT_2026-05-04.md`

Self-verification guide for recording a client report video:
`SELF_VERIFICATION_GUIDE_00_07.md`

Important boundary: this proves the checked-in 00-07 gates, including Ollama,
raw CUDA, API consistency, memory/sync cleanup, PyTorch, CuPy, TensorFlow,
same-VM concurrency, cross-VM concurrency, and malformed-request rejection. It is
not a blanket claim that every possible CUDA application or every model shape has
been tested.

## 00 - Preserve Ollama Baseline

Folder: `00_preserve_ollama_baseline/`

Purpose: protect the proven Ollama-mediated path and keep Plan A, Plan B, and
Plan C available as regression canaries.

Required before later milestones: Plan A pass; downstream Plan B or Plan C
recheck when a change can affect that lane.

## 01 - General CUDA Gate

Folder: `01_general_cuda_gate/`

Purpose: create the first non-Ollama compatibility gate using small CUDA tests:
device query, alloc/free, HtoD, DtoH, kernel launch, streams, events, module
load, repeated process cleanup.

Milestone 01 is complete as the preserved non-Ollama raw CUDA gate.

## 02 - API Coverage Audit

Folder: `02_api_coverage_audit/`

Purpose: document supported, partial, stubbed, missing, unsafe, and
Ollama-shaped API behavior across CUDA Driver API, CUDA Runtime API, cuBLAS,
cuBLASLt, and NVML.

Milestone 02 is complete as the API coverage and fail-closed hardening gate.
Remaining P1/P2 risks are carried forward into later milestones.

## 03 - Memory, Sync, Cleanup

Folder: `03_memory_sync_cleanup/`

Purpose: harden allocation ownership, pointer lifetime, async copies, streams,
events, stale payload handling, process exit cleanup, and recovery after host
CUDA errors.

Milestone 03 is complete. Process cleanup, forced-kill stale-owner recovery,
4 MiB async/mixed stream-event behavior, final Plan A, and final raw CUDA
regressions passed.

## 04 - PyTorch Gate

Folder: `04_pytorch_gate/`

Purpose: validate PyTorch GPU availability, tensor allocation, transfer,
elementwise operation, matmul, small model inference, repeated execution, and
process restart.

Milestone 04 is complete. The bounded PyTorch gate passed 3/3 fresh-process
runs, and final serial preservation passed Plan A, Plan B, Plan C, Milestone 01,
Milestone 02, and Milestone 03 after the 64 KiB BAR1 copy-chunk correction.

## 05 - Second Framework Gate

Folder: `05_second_framework_gate/`

Purpose: validate a second independent non-Ollama stack such as TensorFlow,
CuPy, ONNX Runtime, or another agreed runtime.

**2026-04-29 update:** the **CuPy** lane remains closed; **TensorFlow** bounded
GPU training (`TensorFlow` **2.16.2** / CUDA **12.3** wheels, checked-in
`tests/second_framework_gate/tensorflow_mnist_probe.py`) passes on VM-10 with
mediated **`GPU:0`** after **`EigenMetaKernel`** launch-layout fixes in
`guest-shim/libvgpu_cuda.c`. Serial preservation **00–07** after that shim deploy
is recorded in **`ERROR_TRACKING_STATUS.md`** (session **serial preservation**).

## 06 - Multi-Process And Multi-VM

Folder: `06_multiprocess_multivm/`

Purpose: prove virtualization behavior under concurrency, priority, fairness,
memory pressure, cancellation, cleanup, and mediator health.

Scope note: M06 documented concurrency evidence centers on Ollama, raw CUDA,
PyTorch, and CuPy; TensorFlow concurrency was not part of the original M06 gate
set. After **`M05-E5`** closure, an additional **two-process CuPy** check was run
during serial preservation without regression.

## 07 - Security And Isolation

Folder: `07_security_isolation/`

Purpose: define and test tenant assumptions, malformed requests, bounds,
MMIO/BAR policy, IOMMU expectations, quarantine behavior, and safe recovery.

Historical scope note: the original M07 write-up predated TensorFlow GPU being
in scope. **`M05-E5`** is now closed; serial re-preservation re-ran
**`malformed_socket_probe.py`** after TensorFlow/shim work. Full TensorFlow-targeted
security fuzzing remains optional if the product requires it.

## 08 - Server 2 Migration

Folder: `08_server2_migration/`

Purpose: move from Server 1 engineering proof to a controlled Server 2 mediated
deployment path, while preserving rollback and client demonstration stability.

## 09 - TWA Research

Folder: `09_twa_research/`

Purpose: begin the longer-term TWA/API research track only after the vGPU layer
has enough stability to support parallel research without disrupting the
engineering baseline.
