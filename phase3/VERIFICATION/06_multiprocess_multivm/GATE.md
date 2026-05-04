# Gate - Milestone 06 Multi-Process And Multi-VM

## Scope

Milestone 06 proves that the mediated vGPU path remains correct when more than
one guest workload uses it.

Initial scope is same-VM multi-process behavior on Test-10. Multi-VM behavior is
included only when a second mediated VM is available and can be tested without
destroying the preserved baseline.

## Required Gate Cases

1. Two independent CUDA framework processes in the same VM run concurrently and
   both pass value checks.
2. Each process produces its own result artifact; no stale status or output from
   one process is accepted for the other.
3. A failed or timed-out process does not poison a following known-good probe.
4. Mediator logs identify process ownership sufficiently to diagnose which
   VM/process owns work.
5. Prior single-process gates remain green after the concurrency experiment.
6. A second existing VM is recovered or created as a mediated VM without adding a
   second mediator process.
7. The single host mediator serves both VM sockets simultaneously.
8. Both VMs execute mediated CUDA work concurrently.
9. Framework-level work is tested from both VMs where package capacity allows.
10. A killed workload in one VM does not poison the other VM.
11. Ollama baseline traffic is tested while another VM uses the mediated CUDA
    path.
12. Pool/priority/fairness observability is recorded from mediator stats and
    request ownership logs.

## First Bounded Probe

Run two CuPy M05 probes concurrently in the same VM using the isolated framework
environment:

- Python: `/mnt/m04-pytorch/venv/bin/python`.
- Probe: `/mnt/m04-pytorch/cupy_probe.py`.
- Environment: `LD_LIBRARY_PATH=/opt/vgpu/lib:$LD_LIBRARY_PATH`.
- Timeout: `300s` per child.

The first probe is intentionally narrow and quick. It should expose stale status,
owner cleanup, or mediator interleaving bugs before heavier PyTorch/Ollama
concurrency is attempted.

## Pass Criteria

- Both child processes exit `0`.
- Both child JSON reports have `overall_pass=True`.
- The orchestrator report has `overall_pass=True`.
- A follow-up single-process CuPy probe still passes.

## Fail Criteria

- Either child times out, exits non-zero, emits invalid JSON, or reports a failed
  case.
- A child report contains values that match the other child instead of its own
  expected values.
- A follow-up known-good probe fails after the concurrent run.

## Deferred Scope

Do not defer reachable M06 work just because an earlier smoke test passed. A
case may be deferred only when the current host/VM resources make it unavailable
or when running it would risk destroying the preserved Test-10 baseline. Any
deferral must identify the missing resource and the next safe way to remove it.

## Expanded M06 Gate Ladder

1. Same-VM CuPy/CuPy concurrency on Test-10.
2. Same-VM mixed PyTorch/CuPy concurrency on Test-10.
3. Same-VM killed-process recovery on Test-10.
4. Second mediated VM bring-up using an existing VM where possible.
5. True multi-VM CUDA smoke: Test-10 framework gate plus Test-6 raw mediated
   allocation/free.
6. Test-6 framework readiness: install or reuse an isolated framework runtime
   and run value-checked CuPy.
7. True multi-VM framework gate: Test-10 CuPy or PyTorch plus Test-6 CuPy.
8. Cross-VM failure isolation: kill a Test-6 CUDA process while Test-10 runs a
   known-good gate, then rerun known-good probes on both sides.
9. Ollama plus second-VM CUDA/framework concurrency.
10. Pool/priority/fairness observation: record Pool A/Pool B processed counts,
    queue depth, request ownership, and whether high/medium/low priority labels
    are visible in logs or configuration.
11. Final preservation: rerun the required prior milestone canaries after the M06
    changes.
