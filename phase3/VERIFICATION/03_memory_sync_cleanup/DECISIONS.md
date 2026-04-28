# Decisions - Milestone 03 Memory, Synchronization, And Cleanup

## 2026-04-28 - Start From Audit Plus Small Gate

- Decision: begin Milestone 03 by defining a focused memory/sync/cleanup gate
  and auditing current behavior before changing code.
- Reason: Milestone 03 touches shared runtime behavior where broad changes can
  easily regress Plan A or the raw CUDA gate.
- Rejected alternatives: jump directly to PyTorch; start by refactoring memory
  ownership without a bounded repro.
- Reversal/removal condition: none for Milestone 03.

## 2026-04-28 - Treat Kill/Restart As Required But Later In The Gate

- Decision: run forced process termination only after the safe repeated
  memory/sync probe passes.
- Reason: kill/restart can leave GPU and mediator state in unusual conditions;
  a safe-path baseline is needed before interpreting cleanup failures.
- Rejected alternatives: begin with kill testing; treat repeated normal process
  runs as enough to close cleanup.
- Reversal/removal condition: a source audit proves a cleanup bug that should be
  fixed before any kill test is meaningful.

## 2026-04-28 - Keep One Active Error

- Decision: no Milestone 03 active error is promoted at milestone start.
- Reason: the folder creation and Plan A baseline are not failures. The active
  error must come from the first concrete audit finding or bounded gate failure.
- Rejected alternatives: pre-promote all carry-forward candidates from
  Milestone 02.
- Reversal/removal condition: promote one candidate once evidence proves it is
  the earliest Milestone 03 blocker.

## 2026-04-28 - Promote Last-Connection Cleanup As `M03-E1`

- Decision: promote `M03-E1` for missing executor cleanup when the last
  persistent CUDA connection for a VM closes.
- Reason: this is the earliest direct Milestone 03 cleanup blocker found by
  source audit. The executor has cleanup machinery, but the mediator does not
  invoke it on guest process disconnect.
- Rejected alternatives: immediately add true guest-PID ownership to the wire
  protocol; clean the whole VM on every fd close; defer cleanup until a PyTorch
  failure appears.
- Reversal/removal condition: replace last-connection cleanup with true
  process-scoped cleanup once the protocol carries guest process identity and a
  multi-process gate exists.

## 2026-04-28 - Reject Mediator-Only Cleanup As Closure

- Decision: do not close `M03-E1` from mediator connection cleanup alone.
- Reason: the persistent mediator connection belongs to the QEMU vGPU stub, not
  to each guest CUDA process. Short raw CUDA probes did not close the mediator
  fd or fire cleanup, so this does not prove process-exit cleanup.
- Rejected alternatives: declare closure from Plan A/raw CUDA pass alone; clean
  all VM executor state on every process-like event.
- Reversal/removal condition: none. Process ownership must cross the guest
  transport/stub/mediator boundary.

## 2026-04-28 - Use BAR0 Scratch For Process Ownership

- Decision: draft the cross-layer fix using existing BAR0 scratch register
  `0x03c` to carry guest PID to the QEMU vGPU stub. The stub derives an internal
  owner id from `(vm_id,pid)`, while the socket header still carries the real VM
  id for scheduling and metrics.
- Reason: this avoids widening every CUDA argument payload and avoids cleaning
  unrelated processes in the same VM.
- Rejected alternatives: add a new MMIO register immediately; overload one of
  the 16 CUDA inline args; key ownership by mediator fd.
- Reversal/removal condition: replace with an explicit protocol-versioned
  process-id field if the QEMU/MMIO ABI is revised later.

## 2026-04-28 - Close `M03-E1`

- Decision: close `M03-E1` for normal CUDA process exit cleanup.
- Reason: the deployed guest transport, QEMU vGPU stub, mediator, and executor
  now carry process ownership and the host log proves cleanup ran for 12 owner
  ids. Plan A and the raw CUDA gate both passed after the live deployment.
- Rejected alternatives: treat forced `SIGKILL` as closed by destructor-based
  cleanup; keep `M03-E1` open after normal process cleanup proof.
- Reversal/removal condition: reopen only if a normal process exit no longer
  emits `CUDA_CALL_PROCESS_CLEANUP` or if Plan A/raw CUDA regresses because of
  the owner-id routing.

## 2026-04-28 - Close `M03-E2`

- Decision: close `M03-E2` by adding guest-side stale-owner sweep.
- Reason: forced `SIGKILL` bypasses destructors, so the next CUDA process must
  clean stale registered owners. The forced-kill gate proved owner `167776108`
  was cleaned by the next process and Plan A/raw CUDA both passed afterward.
- Rejected alternatives: rely only on Driver shim destructor cleanup; accept a
  leak because the next process still passed; clean all VM state on every new
  process.
- Reversal/removal condition: reopen if a killed process owner remains in the
  registry and is not cleaned by the next CUDA process, or if stale-owner sweep
  regresses Plan A/raw CUDA.

## 2026-04-28 - Close `M03-E3`

- Decision: close `M03-E3` by replacing fake Driver event and stream-wait
  success paths with host RPC-backed behavior.
- Reason: Milestone 03 requires stream/event behavior to be proven by host state
  and byte verification, not by local fake handles. The repeated async/mixed
  probe passed 5 fresh processes and final Plan A/raw CUDA regressions passed.
- Rejected alternatives: keep local fake events because the current probes pass;
  defer event correctness to the PyTorch milestone.
- Reversal/removal condition: reopen if event or stream-wait calls stop reaching
  the host, if byte verification fails after an async/mixed run, or if the final
  regression gates fail after related changes.

## 2026-04-28 - Close `M03-E4`

- Decision: cap BAR1 copy chunks to 256 KiB for HtoD and DtoH transport paths.
- Reason: the 4 MiB async/mixed gate exposed a stall when async HtoD used one
  single BAR1 transaction. Chunking the same payload passed 5 fresh process
  runs with 4 MiB byte verification and preserved final Plan A/raw CUDA.
- Rejected alternatives: close Milestone 03 using only the 4 KiB async/mixed
  probe; block on shmem GPA resolution before proving BAR1 fallback behavior.
- Reversal/removal condition: reopen if a multi-megabyte BAR1 copy stalls again,
  produces incorrect bytes, or regresses Plan A/raw CUDA.

## 2026-04-28 - Close Milestone 03

- Decision: mark Milestone 03 complete.
- Reason: process-scoped cleanup, forced-kill stale-owner cleanup, 4 MiB
  async/mixed stream-event behavior, copy/memset/DtoD integrity, final Plan A,
  and final raw CUDA gates all passed with host evidence.
- Rejected alternatives: block Milestone 03 on BAR1 fallback after shmem GPA
  resolution reports `pfn_hidden`; treat residual `cuFuncGetParamInfo(0x00bc)`
  noise as an active Milestone 03 blocker.
- Reversal/removal condition: reopen only if one of the Milestone 03 closure
  gates regresses or a later milestone proves the carried-forward observations
  are the immediate blocker.
