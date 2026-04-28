# Decisions - Milestone 02 API Coverage Audit

## 2026-04-28 - Audit Before Framework Work

- Decision: run a source-level API coverage audit before starting PyTorch or
  TensorFlow gates.
- Reason: Milestone 01 proves a bounded raw CUDA path, but frameworks depend on
  many API calls that may currently be stubbed, Ollama-shaped, or silently
  successful.
- Rejected alternatives: jump directly to PyTorch after Milestone 01; assume
  Ollama and raw probes imply broad API compatibility.
- Reversal/removal condition: none for Milestone 02.

## 2026-04-28 - Do Not Fix During First Audit Pass

- Decision: classify behavior first, then fix in later bounded steps.
- Reason: changing runtime behavior during a broad audit would mix discovery and
  correction, making active-error tracking unreliable.
- Rejected alternatives: patch every suspicious stub as it is found.
- Reversal/removal condition: user explicitly requests a specific bounded fix
  after the matrix is created.

## 2026-04-28 - Promote Silent Success As Active Error

- Decision: promote `M02-E1` for silent-success API stubs that can affect
  general workloads.
- Reason: unsupported APIs that return success are more dangerous than clear
  unsupported errors because later framework gates can pass discovery while
  silently skipping work.
- Rejected alternatives: treat all stubs as candidate-only; proceed directly to
  PyTorch without changing failure behavior.
- Reversal/removal condition: the highest-risk paths either gain real semantics
  or fail closed with clear unsupported errors, and the next milestone gate
  confirms it does not depend on deferred stubs.

## 2026-04-28 - Fail Closed Before Framework Gates

- Decision: close `M02-E1` by returning explicit unsupported or initialization
  errors for high-risk Runtime and cuBLASLt paths that previously returned
  success without real work.
- Reason: a clean unsupported error is safer than a fake success before PyTorch
  or other frameworks are allowed to judge correctness.
- Rejected alternatives: keep success stubs for compatibility; jump straight to
  implementing full Runtime kernel and cuBLASLt semantics.
- Reversal/removal condition: replace each fail-closed path with real mediated
  semantics and a correctness gate.

## 2026-04-28 - Keep cuBLASLt Create/Destroy Discovery-Compatible

- Decision: keep `cublasLtCreate` and `cublasLtDestroy` returning success for
  now, but fail closed descriptor/layout/preference/heuristic/matmul calls.
- Reason: this preserves library-loading compatibility while preventing fake
  cuBLASLt compute success.
- Rejected alternatives: make `cublasLtCreate` fail immediately; leave all
  cuBLASLt calls as success stubs.
- Reversal/removal condition: implement real cuBLASLt handle and matmul support,
  or prove no future target framework probes cuBLASLt at load time.

## 2026-04-28 - Promote cuBLAS Stub Handle As Next Active Risk

- Decision: after closing `M02-E1`, promote `M02-E2` for cuBLAS stub-handle fake
  success behavior.
- Reason: `cublasCreate_v2` can still create a stub handle if transport is
  unavailable, and later GEMM calls on that stub handle can return success
  without host compute.
- Rejected alternatives: defer cuBLAS until PyTorch; treat it as documentation
  only.
- Reversal/removal condition: cuBLAS either connects to host transport or fails
  closed without returning a stub compute-success handle.

## 2026-04-28 - Close cuBLAS Stub Handle Path

- Decision: close `M02-E2` by making cuBLAS creation fail when neither remote
  transport nor a real compute context is available, and by rejecting any
  historical stub handle in control or compute calls.
- Reason: no BLAS handle should exist if later GEMM calls cannot execute on the
  host or a real CUDA-backed library path.
- Rejected alternatives: keep the local stub handle for discovery compatibility;
  only fail GEMM calls while still allowing stub handle creation.
- Reversal/removal condition: introduce a real fallback compute implementation
  with a correctness gate.

## 2026-04-28 - Promote Protocol Coverage As Next Active Risk

- Decision: after closing `M02-E2`, promote `M02-E3` for protocol IDs without
  explicit executor handling.
- Reason: a protocol ID that exists without a clear executor case can produce
  ambiguous failure behavior later. Milestone 02 is the right place to classify
  those paths before framework gates.
- Rejected alternatives: defer protocol cleanup until a framework hits one of
  the IDs; leave them documented only.
- Reversal/removal condition: each protocol ID either maps to implemented
  executor behavior or returns a clear unsupported status.

## 2026-04-28 - Close Protocol Coverage Gap Explicitly

- Decision: close `M02-E3` by adding named executor cases for every protocol ID
  that previously had no explicit switch disposition.
- Reason: a generic default already failed unsupported calls, but Milestone 02
  requires auditable classification. Future framework failures must show a named
  unsupported call rather than an ambiguous protocol fallthrough.
- Rejected alternatives: rely on the generic executor default; implement broad
  semantics for protocol-only APIs before a workload requires them.
- Reversal/removal condition: replace any explicit unsupported protocol case
  with real executor behavior plus a focused correctness gate.

## 2026-04-28 - Require Live Binary Proof For Host Changes

- Decision: do not accept a host executor deployment as verified unless the
  running mediator process is executing the rebuilt binary path.
- Reason: the first `M02-E3` restart attempt left PID `89028` running from
  `/root/phase3/mediator_phase3 (deleted)`, so Plan A evidence from that process
  was not sufficient proof of the deployed code.
- Rejected alternatives: accept rebuild SHA alone; accept a passing canary while
  `/proc/<pid>/exe` still points at a deleted executable.
- Reversal/removal condition: none for host-side runtime changes.

## 2026-04-28 - Close Milestone 02

- Decision: mark Milestone 02 complete after `M02-E1`, `M02-E2`, and `M02-E3`
  were closed and the matrix/gap list classified remaining synthetic,
  Ollama-shaped, partial, and unsupported behavior.
- Reason: the Milestone 02 gate is an API audit and fail-closed hardening gate.
  Remaining P1/P2 items are now known inputs for Milestone 03 and framework
  gates, not unclassified Milestone 02 blockers.
- Rejected alternatives: continue implementing every P1/P2 API under Milestone
  02; start PyTorch without preserving the closure evidence.
- Reversal/removal condition: reopen Milestone 02 only if a later gate finds an
  audited API that was misclassified or still reports fake success in a
  general-workload path that was supposed to fail closed.
