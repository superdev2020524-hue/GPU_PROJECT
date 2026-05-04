# Decisions - Milestone 06 Multi-Process And Multi-VM

## 2026-04-29 - Start With Same-VM Two-Process CuPy

- Decision: make the first M06 gate two concurrent CuPy probes in the same VM.
- Reason: CuPy is the fastest freshly proven framework gate, so it is a good
  concurrency sentinel before mixing in heavier PyTorch or Ollama workloads.
- Rejected alternatives: start with multi-VM work before a second mediated VM is
  prepared, or start with Ollama/PyTorch concurrency before a small same-VM
  ownership/status probe is green.
- Reversal/removal condition: if two-process CuPy proves unsuitable as a
  concurrency sentinel, switch to a smaller raw CUDA two-process gate.

## 2026-04-29 - Advance To Mixed PyTorch Plus CuPy

- Decision: after the two-process CuPy sentinel and killed-process recovery
  passed, run a mixed PyTorch plus CuPy same-VM probe.
- Reason: PyTorch is the heavier proven framework gate and CuPy is the faster
  independent framework gate, so running them together broadens M06 without
  immediately mixing in model-server residency and prompt behavior.
- Rejected alternatives: claim M06 closure from CuPy-only concurrency, or start
  Ollama concurrency before proving two independent framework stacks can coexist.
- Reversal/removal condition: if later Ollama or multi-VM tests regress the
  baseline, return to this mixed-framework gate as the known-good concurrency
  checkpoint.

## 2026-04-29 - Reuse Test-6 For Second Mediated VM

- Decision: reuse existing Test-6 as the second mediated VM instead of creating a
  fresh VM.
- Reason: the repo's VM creation history documents ISO SR/network fragility for
  new VM creation, while Test-6 had historical SSH evidence and was recoverable
  by restoring vCPUs.
- Rejected alternatives: create a new VM immediately, use Test-8 after it failed
  expected-IP reachability, or use Test-5 after SSH port 22 refused connections.
- Implementation: attach Test-6 to the existing single mediator as
  `-device vgpu-cuda,pool_id=B,priority=low,vm_id=6`, then copy the known-good
  Test-10 shim library set into `/opt/vgpu/lib` on Test-6.
- Reversal/removal condition: if Test-6 later proves unstable under heavier
  workloads, keep it as the initial multi-VM smoke baseline and prepare a fresh
  VM through the documented `vm_create/create_vm.sh` path.

## 2026-04-29 - Expand M06 Beyond First Smoke Pass

- Decision: do not close M06 from the initial same-VM and Test-6 raw CUDA smoke
  gates; require framework-on-both-VMs, cross-VM failure isolation, Ollama plus
  second-VM load, fairness/ownership evidence, and final preservation.
- Reason: the milestone is specifically about concurrency and multi-VM
  virtualization behavior, so a single raw second-VM allocation/free pass is not
  enough to remove hidden scheduling, cleanup, or poisoning risks.
- Rejected alternatives: defer Test-6 framework setup to a later milestone, or
  treat Pool A/Pool B counters as sufficient without a bounded delta workload.
- Reversal/removal condition: if a future QoS milestone requires stronger
  scheduling guarantees than M06 observes, define a separate priority-policy
  gate instead of reopening the M06 closure.

## 2026-04-29 - Keep Test-6 Framework Runtime Minimal

- Decision: install CuPy and the missing NVRTC/runtime wheels on Test-6, but not
  a full PyTorch stack there during M06.
- Reason: Test-10 already provides the heavy PyTorch framework lane; Test-6 CuPy
  is enough to prove framework-level work from both VMs through one mediator
  while avoiding unnecessary disk and package churn on the recovered VM.
- Rejected alternatives: leave Test-6 at raw CUDA only, or install PyTorch on
  both VMs before closing M06.
- Reversal/removal condition: if Milestone 07/08 requires symmetric heavy
  framework coverage across VMs, install PyTorch on Test-6 under a separate gate
  with fresh disk-capacity evidence.
