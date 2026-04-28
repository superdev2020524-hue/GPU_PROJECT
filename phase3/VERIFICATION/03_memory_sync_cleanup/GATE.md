# Gate - Milestone 03 Memory, Synchronization, And Cleanup

## Gate Name

`phase3_memory_sync_cleanup_gate`

## Purpose

Prove the vGPU layer can safely handle repeated non-Ollama process behavior:
allocation ownership, copies, sync/async sequencing, streams/events, process
exit, and recovery after failed or interrupted work.

## Required Coverage

The Milestone 03 gate must cover:

- repeated allocation/free cycles;
- large HtoD and DtoH copies with data verification;
- DtoD copies and memset verification;
- async HtoD/DtoH behavior with explicit stream synchronization;
- event record/synchronize/elapsed-time behavior;
- process restart after successful runs;
- forced process termination during or after GPU allocation;
- next-run health after forced termination;
- mediator evidence for per-VM/per-process cleanup or retained state.

## Initial Gate Shape

Start from the existing Milestone 01 raw CUDA gate, then add a new focused probe
under `phase3/tests/memory_sync_cleanup/`.

The first probe should be intentionally small and deterministic:

- allocate multiple buffers;
- copy a large enough payload to expose chunking/lifetime mistakes;
- run a basic kernel or DtoD/memset sequence;
- verify returned bytes;
- create and synchronize streams/events;
- repeat several times in fresh processes;
- run one controlled kill/restart scenario only after the safe path passes.

## Pass Criteria

- Plan A passes before Milestone 03 changes.
- The safe memory/sync probe passes for repeated fresh processes.
- The forced-kill scenario does not poison the next safe probe.
- Host mediator evidence does not show fresh `sync FAILED`,
  `CUDA_ERROR_ILLEGAL_ADDRESS`, or stale-handle failures during the bounded gate.
- VM-side probe output verifies data integrity, not only API return codes.
- Any runtime behavior changed during Milestone 03 is followed by Plan A and
  raw CUDA regression checks.

## Fail Criteria

- Any copy, memset, DtoD, stream, event, or cleanup path returns success while
  producing incorrect data.
- A killed process poisons a later process.
- Mediator per-VM/per-process state cannot be explained from logs or state
  tracking.
- Plan A regresses after a Milestone 03 runtime change.

## Required Output

- `BASELINE.md`
- `GATE.md`
- `ACTIVE_ERROR.md`
- `EVIDENCE.md`
- `DECISIONS.md`
- new or updated bounded probe artifacts, if implementation is required
