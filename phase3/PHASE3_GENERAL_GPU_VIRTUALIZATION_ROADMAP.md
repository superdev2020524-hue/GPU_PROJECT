# Phase 3 General GPU Virtualization Roadmap

*Created: 2026-04-27*

> Internal planning draft. The client-facing version is
> `phase3/CLIENT_NEXT_PHASE_GPU_VIRTUALIZATION_PLAN_2026-04-27.md`.

This document is the planning draft for the next phase after the Ollama milestone.
It translates the client discussions and the current Phase 3 implementation state
into a concrete path toward the real goal: a general-purpose virtual GPU layer for
GPU-using software running inside VMs.

---

## 1. Plain-English Summary

Phase 3 is not an Ollama project.

Ollama was the first proof vehicle because it is a real GPU workload with model
load, CUDA discovery, library loading, memory movement, kernel launch, residency,
and repeated inference behavior. Passing the Ollama gates proves that the basic
guest-to-host GPU remoting path can work.

The real Phase 3 goal is broader:

- a VM user sees what appears to be a GPU,
- GPU software inside the VM calls normal GPU APIs,
- those calls are captured by the guest layer,
- the calls and data are transported through the vGPU stub and mediator,
- the physical GPU executes the work,
- results return to the correct VM/process,
- the user does not need to know this is mediated.

The next phase is therefore not "make more Ollama tests pass." The next phase is
to turn the Ollama-proven path into a general GPU virtualization layer.

---

## 2. What The Client Appears To Want

Based on the February and April client discussions, the client's plan is:

1. **Functionality first**
   - Prove that software inside a VM can use a virtual GPU and have real work
     execute on the physical GPU.
   - Ollama was selected as the first practical target because it covers a large
     class of model-serving behavior.

2. **Invisible layer**
   - The VM user should not need to understand the mediation layer.
   - Each VM should behave as though it has its own GPU, while the host layer
     routes and schedules requests behind the scenes.

3. **Scheduling after the basic path works**
   - Priority, fairness, pooling, and multi-tenant scheduling matter, but they
     should be built after the function path is stable enough to trust.

4. **Hardening after functionality**
   - Security, tenant isolation, IOMMU-related hardening, abuse controls, and
     cloud policy enforcement are required for production, but not before the
     core path proves itself.

5. **Server roles**
   - Server 1 is the engineering/development path for the mediated vGPU layer.
   - Server 2 can remain a stable client/demo environment, currently via
     passthrough or a simpler proven configuration, until the mediated layer is
     mature enough to move there.

6. **Longer-term platform direction**
   - After the GPU virtualization layer stabilizes, the client wants research
     into TWA and possibly a CUDA-like or API-like layer for future hardware.
   - That research should begin only as a controlled parallel track once the GPU
     virtualization workload becomes maintenance/improvement work rather than
     daily core debugging.

---

## 3. Current Baseline

### 3.1 What Is Proven

The current Phase 1/Ollama baseline is documented as closed under three lanes:

- **Plan A:** preserved canary, `qwen2.5:0.5b`, default milestone gate.
- **Plan B:** revised Tiny gate, `tinyllama:latest`.
- **Plan C:** client-facing normal usage lane, `qwen2.5:3b`.

These gates prove that the repaired GPU-backed path can support Ollama inference
under controlled conditions.

### 3.2 What This Does Not Prove

The Ollama baseline does **not** prove that arbitrary GPU software will work.

It does not fully prove:

- broad CUDA Driver API coverage,
- broad CUDA Runtime API coverage,
- PyTorch compatibility,
- TensorFlow compatibility,
- CuPy / Numba / ONNX Runtime compatibility,
- multi-process behavior,
- multi-VM scheduling behavior,
- tenant isolation,
- general memory lifetime safety,
- production lifecycle management,
- cloud integration.

The next phase exists to close those gaps.

---

## 4. Important Concern: Ollama-Specific Assumptions

During the Ollama milestone, some implementation details were necessarily shaped
by Ollama and GGML behavior. This was acceptable for Phase 1 because the goal was
to prove one real end-to-end workload.

For general GPU virtualization, we must now classify every such behavior into one
of three categories.

### Category A: General vGPU Behavior

Keep and harden these. Examples:

- guest-to-host CUDA call transport,
- device discovery,
- device memory allocation and free,
- host/guest pointer mapping,
- HtoD and DtoH copies,
- stream and event handling,
- module/library load,
- kernel launch,
- error propagation,
- per-process cleanup,
- mediator logging and metrics.

### Category B: Workload Adapter Behavior

Keep but isolate behind explicit compatibility rules. Examples:

- Ollama service environment setup,
- GGML CUDA library deployment,
- model residency gates,
- model-specific prompt checks,
- client timeout handling during model load.

These are useful, but they should not be mixed into the core vGPU layer as hidden
assumptions.

### Category C: Historical Debug Workarounds

Do not allow these to become permanent architecture unless they are re-proven as
general mechanisms. Examples may include:

- one-off shim load ordering hacks,
- model-specific library path fixes,
- special cases for individual GGML kernels,
- temporary tracing environment variables,
- broad fallback rules introduced only to get past one Ollama blocker.

The next phase must review these and either:

1. promote them into documented general mechanisms,
2. confine them to the Ollama compatibility profile,
3. or remove them.

---

## 5. Guiding Rule For The Next Phase

Do not repeat the Ollama process by guessing a large set of hidden assumptions.

Instead, build a compatibility ladder:

1. simple CUDA C programs,
2. CUDA Runtime API programs,
3. cuBLAS programs,
4. PyTorch,
5. TensorFlow,
6. higher-level applications.

Each step should have a small gate. A gate should prove one defined class of GPU
behavior before moving upward.

This avoids turning every future workload into another months-long Ollama-style
investigation.

---

## 6. Proposed Milestone Structure

### Milestone 0: Preserve The Ollama Baseline

Purpose:

- protect the work already achieved,
- prevent regressions while generalizing the layer.

Required work:

- keep Plan A, Plan B, and Plan C as non-regression gates,
- run Plan A first before risky runtime changes,
- re-run Plan A and the relevant downstream gate after any risky change,
- keep `/api/ps` residency clean between cross-model checks.

Acceptance criteria:

- Plan A passes,
- Plan B passes when required,
- Plan C remains available as a client-facing confidence check,
- no broad Phase 3 change is accepted if it breaks this baseline without an
  explicit repair cycle.

### Milestone 1: Define The General CUDA Compatibility Gate

Purpose:

- create the next proof target below PyTorch/TensorFlow,
- avoid jumping directly from Ollama to complex frameworks.

Gate cases:

- device count and device properties,
- `cudaMalloc` / `cuMemAlloc`,
- `cudaFree` / `cuMemFree`,
- HtoD copy,
- DtoH copy,
- simple kernel launch,
- stream create/synchronize,
- event record/synchronize,
- module load and function lookup,
- repeated process start/stop cleanup.

Deliverables:

- `phase3/tests/general_cuda_gate/`,
- small C/CUDA test programs,
- a Python or shell runner that produces JSON,
- host mediator evidence capture,
- VM-side result capture.

Acceptance criteria:

- all tests pass in one clean VM session,
- no stale status or stale payload signatures,
- all memory allocations are freed or cleaned after process exit,
- host logs show the expected physical GPU execution path.

### Milestone 2: Complete API Coverage Audit

Purpose:

- determine which CUDA calls are supported, stubbed, partially supported, or
  missing.

Required work:

- audit `guest-shim/libvgpu_cuda.c`,
- audit `guest-shim/libvgpu_cudart.c`,
- audit `guest-shim/libvgpu_cublas.c`,
- audit `guest-shim/libvgpu_cublasLt.c`,
- audit `include/cuda_protocol.h`,
- audit `src/cuda_executor.c`.

Deliverables:

- an API coverage matrix,
- one row per CUDA/CUDART/cuBLAS/cuBLASLt function,
- status values:
  - implemented,
  - implemented but Ollama-shaped,
  - stubbed,
  - missing,
  - unsafe fallback,
  - not required for current milestone.

Acceptance criteria:

- no unknown behavior in the core path,
- every unsupported call returns a clear and correct error,
- no silent fake success for calls that general workloads depend on.

### Milestone 3: Memory And Synchronization Hardening

Purpose:

- make the vGPU layer reliable across software, not just one process shape.

Required work:

- track allocation ownership per VM and process,
- track host-device pointer lifetimes,
- validate HtoD/DtoH data integrity with checksums in debug mode,
- handle async copies correctly,
- support streams/events well enough for real frameworks,
- clean resources when a guest process exits,
- prevent stale BAR1 or stale SHMEM reuse,
- define a recovery rule for host CUDA errors.

Acceptance criteria:

- repeated CUDA gate runs do not leak memory,
- forced process kill does not poison the next run,
- stream/event tests pass,
- large copy tests pass,
- mixed sync/async tests pass,
- mediator can identify and clean per-process state.

### Milestone 4: Framework Gate - PyTorch

Purpose:

- validate the most important general ML framework path.

Gate cases:

- `torch.cuda.is_available()`,
- device name and memory query,
- tensor allocation,
- tensor HtoD/DtoH,
- elementwise operation,
- matrix multiply,
- small neural network inference,
- repeated warm execution,
- process restart cleanup.

Acceptance criteria:

- all cases pass on the mediated path,
- GPU execution evidence appears in the mediator,
- failure mode is clean if a feature is unsupported,
- Ollama Plan A still passes after PyTorch testing.

### Milestone 5: Framework Gate - TensorFlow / Other Runtime

Purpose:

- prove the layer is not accidentally PyTorch-only or Ollama-only.

Gate cases:

- TensorFlow GPU detection,
- small matmul,
- simple model inference,
- memory growth behavior,
- repeated process cleanup.

Optional later gates:

- CuPy,
- Numba,
- ONNX Runtime,
- llama.cpp direct,
- raw CUDA sample suite.

Acceptance criteria:

- at least two independent non-Ollama GPU software stacks pass,
- unsupported behavior is clearly documented,
- no regression to Ollama canaries.

### Milestone 6: Multi-Process And Multi-VM Behavior

Purpose:

- move from "one workload works" to "virtualization layer works."

Required work:

- two processes in one VM,
- multiple VMs if available,
- priority scheduling,
- fairness policy,
- memory pressure handling,
- long-running workload plus short interactive workload,
- cancellation and cleanup,
- mediator health under load.

Acceptance criteria:

- no cross-process data leakage,
- no stale status from one process affects another,
- priority policy is observable,
- one failed workload does not poison the mediator,
- metrics identify which VM/process owns work.

### Milestone 7: Security And Isolation Hardening

Purpose:

- prepare for tenant/cloud use.

Required work:

- define trusted vs untrusted tenant assumptions,
- validate MMIO/BAR access policy,
- define IOMMU expectations,
- restrict guest-controlled inputs,
- validate mediator request bounds,
- prevent malformed guest requests from corrupting host state,
- introduce quarantine/kill behavior for abusive VMs.

Acceptance criteria:

- malformed request tests fail safely,
- unauthorized memory access is blocked,
- mediator remains alive under bad input,
- logs clearly identify the offender,
- recovery does not require host reboot in normal cases.

### Milestone 8: Server 2 Migration / Product Demonstration Path

Purpose:

- transition from engineering success to deployable platform behavior.

Required work:

- keep Server 2 passthrough path available for stable demos until mediation is
  ready,
- define criteria for switching Server 2 to mediated vGPU mode,
- create a deployment checklist,
- create rollback instructions,
- create client demonstration scripts.

Acceptance criteria:

- Server 1 passes general gates,
- Server 2 can deploy the same mediated stack,
- rollback to known stable mode is documented,
- client-facing demo does not depend on hidden manual state.

### Milestone 9: TWA / Future Compute API Research Track

Purpose:

- begin the client's longer-term hardware/API planning without destabilizing the
  vGPU layer.

Start condition:

- Ollama baseline preserved,
- general CUDA gate passing,
- at least one non-Ollama framework gate passing or actively isolated.

Research questions:

- what existing APIs or libraries can express TWA-style workloads,
- whether CUDA-like semantics are enough,
- whether a new operation set is required,
- what a minimal software prototype should demonstrate before silicon work,
- how future hardware would expose itself to applications.

Deliverables:

- research memo,
- existing API comparison,
- minimal prototype proposal,
- risks and unknowns.

---

## 7. Workstreams

### Workstream A: Compatibility Gates

Build small gates from raw CUDA up to frameworks. This is the main method for
turning Phase 3 from Ollama-proven to general-purpose.

### Workstream B: API Coverage

Finish the protocol/shim/executor coverage table and implement missing calls in
priority order based on the gates.

### Workstream C: Runtime Robustness

Focus on memory, streams, events, cleanup, process lifetime, and error recovery.

### Workstream D: Scheduling And Multi-Tenancy

Build priority and fairness only after the single-workload path is stable enough
to avoid confusing scheduling failures with basic API failures.

### Workstream E: Productization

Create repeatable deployment, verification, rollback, and demo procedures for
Server 1 and Server 2.

### Workstream F: Future API / TWA Research

Run as a parallel research thread once the core vGPU work enters a stable
maintenance rhythm.

---

## 8. Immediate Next Actions

### Step 1: Freeze Current Baseline

Record:

- current Plan A status,
- current Plan B status,
- current Plan C status,
- host mediator build identity,
- guest shim identity,
- service config identity.

### Step 2: Write The General CUDA Gate

Create a minimal test suite that proves the core CUDA operations independently of
Ollama.

Recommended path:

- `phase3/tests/general_cuda_gate/`

Initial cases:

1. device discovery,
2. memory alloc/free,
3. HtoD/DtoH copy,
4. vector-add kernel,
5. stream sync,
6. event sync,
7. repeat loop,
8. process-exit cleanup.

### Step 3: Build The API Coverage Matrix

Create:

- `phase3/API_COVERAGE_MATRIX.md`

Track every implemented and missing function across:

- CUDA Driver API,
- CUDA Runtime API,
- cuBLAS,
- cuBLASLt,
- NVML.

### Step 4: Decide Compatibility Profile Boundaries

Create explicit profiles:

- `core-vgpu`: must be app-independent,
- `ollama-profile`: allowed to contain Ollama/GGML-specific config,
- `framework-profile`: PyTorch/TensorFlow-specific compatibility notes,
- `debug-profile`: temporary tracing only.

This prevents Ollama-specific fixes from silently becoming general architecture.

### Step 5: Run One Non-Ollama Gate

Before adding more Ollama refinements, run and stabilize at least one independent
CUDA program through the mediated path.

---

## 9. Success Definition For Full Phase 3

Phase 3 should not be called complete until all of the following are true:

1. The Ollama Plan A/B/C baseline still passes.
2. The general CUDA gate passes.
3. At least two non-Ollama GPU software stacks pass meaningful gates.
4. Multi-process behavior is validated.
5. Multi-VM or equivalent scheduling behavior is validated.
6. Cleanup after process failure is reliable.
7. Unsupported calls fail safely and clearly.
8. Tenant/security assumptions are documented and tested.
9. Server 2 has a documented path from stable passthrough demo mode to mediated
   vGPU mode.
10. Client-facing demos are repeatable from clean startup instructions.

---

## 10. Main Risks

### Risk 1: Hidden Ollama Assumptions

Some current behavior may only work because Ollama/GGML uses a specific call
pattern. Mitigation: build raw CUDA and framework gates immediately.

### Risk 2: Silent Fake Success

Returning success for unsupported calls can make simple workloads appear healthy
while corrupting complex frameworks. Mitigation: strict API coverage matrix and
explicit unsupported errors.

### Risk 3: State Leakage

One process or VM may poison another through stale memory, stale status, or stale
handles. Mitigation: per-process ownership and cleanup gates.

### Risk 4: Scheduling Too Early

Adding scheduling before basic compatibility is stable can hide root causes.
Mitigation: scheduling starts after the general CUDA gate is green.

### Risk 5: Client Misinterpretation

The client may hear "Ollama works" as "full GPU virtualization is done."
Mitigation: reports must say Ollama is the first proof vehicle, not final Phase 3.

---

## 11. Communication Guidance

When reporting to the client:

- say that the core mediated GPU path has been proven through Ollama,
- say that Server 2 remains the stable demo path,
- say that Server 1 is the engineering path toward the full mediated vGPU layer,
- avoid saying "full GPU virtualization is complete" until non-Ollama and
  multi-tenant gates pass,
- explain that the next phase is compatibility expansion and hardening, not a
  restart from zero.

Recommended phrasing:

> We have proven the mediated GPU path with Ollama as the first major workload.
> The next phase is to generalize and harden that path so it supports CUDA
> programs and major GPU frameworks inside VMs, then add scheduling, isolation,
> and cloud integration.

---

## 12. Planning Decision

The next phase should begin with **general CUDA compatibility**, not with another
large application.

Reason:

- raw CUDA gates expose missing API and memory/stream bugs directly,
- PyTorch/TensorFlow failures are too large to debug without a lower ladder,
- Ollama already proved the architecture but may have introduced workload-shaped
  assumptions,
- the client wants a general invisible GPU layer, not another single-app success.

Therefore the immediate engineering priority is:

1. preserve Ollama gates,
2. create and pass the general CUDA gate,
3. create API coverage matrix,
4. expand to PyTorch,
5. expand to TensorFlow,
6. then proceed to scheduling, hardening, Server 2 migration, and cloud.
