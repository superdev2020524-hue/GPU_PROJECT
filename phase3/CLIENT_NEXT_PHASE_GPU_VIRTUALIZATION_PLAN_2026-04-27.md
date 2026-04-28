# Phase 3 Next-Phase Plan

*Producer: Bren*

## Purpose

Phase 3 is intended to provide a virtual GPU layer for VM workloads. The target
state is that software inside a VM can see a GPU, issue GPU work through normal
interfaces, and have that work executed on the physical GPU through the mediation
layer.

The recent Ollama work is the first end-to-end proof of this path. It is not the
end of Phase 3. It gives us a working baseline from which to validate broader GPU
software, scheduling, isolation, and deployment behavior.

## Current Assessment

Server 1 is the engineering path for the mediated architecture:

`VM workload -> guest shim -> vGPU stub -> host mediator -> physical GPU -> VM result`

Using Ollama, this path has now been validated under controlled gates. The work
proved that GPU activity can originate inside the VM, cross the mediation layer,
execute on the physical H100, and return usable results.

Server 2 remains the stable client-facing environment. Its passthrough-based path
is suitable for demonstrations and external testing while the mediated path is
expanded and hardened on Server 1.

This is a verification and extension phase, not a reconstruction. The core path
exists. The remaining work is to prove how far that path generalizes, close the
gaps found by non-Ollama workloads, and prepare the system for controlled
multi-tenant use.

## What Is Proven

- The mediated path can support a real VM GPU workload.
- The guest shim and host mediator can carry CUDA-related work.
- The physical GPU can execute work initiated from inside the VM.
- The result path back into the VM is functional.
- Ollama provides a repeatable regression baseline for future changes.

## What Remains Unproven

- General CUDA programs outside Ollama.
- Full CUDA Driver and Runtime API coverage.
- Framework compatibility for PyTorch and TensorFlow.
- Multi-process and multi-VM behavior.
- Cleanup after failed or interrupted workloads.
- Scheduling policy under shared GPU use.
- Tenant isolation and malformed-request handling.
- A repeatable Server 2 migration path for mediated vGPU mode.

## Engineering Risk

The main risk is that some fixes made during the Ollama milestone may be specific
to Ollama or GGML rather than general GPU virtualization. That is expected at
this stage, but it must now be separated clearly.

The implementation should be reviewed in three groups:

- **Core vGPU behavior:** discovery, allocation, memory copy, module load, kernel
  launch, streams, events, error handling, cleanup, mediator logging.
- **Workload profile behavior:** Ollama service configuration, GGML library
  handling, model residency, and client load-time behavior.
- **Development workarounds:** tracing flags, one-off load ordering, broad
  fallbacks, and model-specific special cases.

Only the first group belongs in the general virtualization layer by default. The
second group should remain documented as workload compatibility. The third group
should be reviewed before it is carried forward.

## Next-Phase Plan

### 1. Preserve the Ollama Baseline

Keep the current Ollama gates as regression checks. Any change to the guest shim,
transport, mediator, service configuration, or CUDA executor should be checked
against this baseline.

**Reason:** this is the strongest end-to-end proof currently available. It should
not be lost while extending the system.

**Output:** recorded baseline versions for the host mediator, guest shim, service
configuration, and Ollama gate results.

### 2. Build a General CUDA Compatibility Gate

Create a small CUDA test suite independent of Ollama. It should cover:

- device query,
- memory allocation and free,
- host-to-device and device-to-host copy,
- simple kernel launch,
- stream synchronization,
- event synchronization,
- repeated process start/stop,
- cleanup after process exit.

**Reason:** raw CUDA tests isolate core GPU behavior before introducing large
frameworks. If PyTorch fails first, the failure surface is too broad.

**Output:** repeatable CUDA gate with VM-side results and host mediator evidence.

### 3. Produce an API Coverage Matrix

Document the implemented surface for:

- CUDA Driver API,
- CUDA Runtime API,
- cuBLAS,
- cuBLASLt,
- NVML.

Each entry should be marked as implemented, partially implemented, missing,
stubbed, unsupported by design, or only proven through Ollama.

**Reason:** the system cannot be treated as a general vGPU layer until the
supported API surface is explicit.

**Output:** API coverage matrix and prioritized implementation gap list.

### 4. Harden Memory, Synchronization, and Cleanup

Validate allocation ownership, pointer mapping, async copy behavior, streams,
events, process termination, stale status handling, and stale payload handling.

**Reason:** general GPU software will exercise more process and synchronization
patterns than Ollama. These areas are common sources of cross-workload failures.

**Output:** memory lifetime tests, stream/event tests, process-kill cleanup tests,
and mediator cleanup rules.

### 5. Validate PyTorch

After the raw CUDA gate is stable, validate PyTorch with:

- GPU availability,
- device name query,
- tensor allocation,
- tensor transfer,
- elementwise operation,
- matrix multiplication,
- small model inference,
- repeated execution,
- process restart.

**Reason:** PyTorch is a high-priority real workload and a strong indicator that
the mediated path is moving beyond Ollama.

**Output:** PyTorch gate and gap report.

### 6. Validate a Second Framework

Use TensorFlow, CuPy, or ONNX Runtime as a second independent workload.

**Reason:** one additional framework reduces the risk of building another
single-workload adaptation.

**Output:** second-framework gate and compatibility notes.

### 7. Validate Multi-Process and Multi-VM Behavior

Run concurrent workloads in one VM and, where available, across multiple VMs.
Include mixed long-running and short-running workloads.

**Reason:** the client requirement is shared GPU access that appears dedicated to
each VM or user. That cannot be proven with a single process.

**Output:** concurrency results, failure isolation results, and mediator health
observations.

### 8. Add Scheduling and Resource Policy

Define priority, fairness, per-VM limits, queue behavior, starvation prevention,
metrics, and administrative controls.

**Reason:** scheduling is central to the product goal, but it should be added
after the basic compatibility path is stable enough to avoid masking lower-level
failures.

**Output:** scheduling policy, metrics, and priority/fairness tests.

### 9. Add Tenant Hardening

Define and test behavior for malformed requests, resource abuse, MMIO/BAR access,
IOMMU assumptions, request bounds, quarantine, and recovery.

**Reason:** internal testing and tenant/cloud use have different risk profiles.
External use requires stricter failure handling and isolation.

**Output:** security assumptions document, negative tests, and recovery policy.

### 10. Plan Server 2 Migration

Keep Server 2 on the stable client-facing path until Server 1 passes the
compatibility and hardening gates. Then prepare a controlled mediated deployment.

**Reason:** Server 2 should remain reliable for demonstrations and client testing
while Server 1 carries engineering risk.

**Output:** deployment checklist, rollback checklist, mediated-mode acceptance
gate, and demo script.

### 11. Start TWA / Future API Research

Begin TWA research once the GPU virtualization work is stable enough to move into
maintenance and compatibility expansion.

The first research output should compare existing APIs and libraries, identify
whether CUDA-like semantics are enough, and define what a minimal software
prototype would need to prove before hardware work.

**Reason:** the you has a longer-term hardware direction, but that work should
not displace completion of the GPU virtualization layer.

**Output:** research memo, API comparison, prototype proposal, and hardware-facing
requirements draft.

## Near-Term Deliverables

1. Baseline record for the current Ollama-mediated path.
2. General CUDA compatibility gate.
3. API coverage matrix.
4. Memory and synchronization hardening tests.
5. PyTorch validation gate.
6. Second-framework validation gate.

## Decision

The next engineering task should be the general CUDA compatibility gate. It is
the shortest path from the current Ollama proof to a reliable assessment of
general GPU virtualization readiness.
