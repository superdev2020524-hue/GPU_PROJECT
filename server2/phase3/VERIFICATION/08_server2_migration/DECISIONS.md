# Decisions - Milestone 08 Server 2 Migration

## 2026-05-01 - Use Server 2 Registry Only

Decision:

- All M08 work will be recorded under `server2/phase3/VERIFICATION/08_server2_migration/`.
- Root `phase3/` remains the protected Server 1 track and is not edited for M08.

Reason:

- `server2/phase3/SERVER2_ISOLATION_AND_MISSION_RULES.md` explicitly requires
  Server 2 work to stay under `server2/`.

## 2026-05-01 - Prefer Passthrough Fast Path For Server 2 Demo

Decision:

- M08 starts from the documented Server 2 passthrough fast path, not from
  deploying the mediated Server 1 stack to Server 2 immediately.

Reason:

- Server 2 mission notes identify the final exposure method as real GPU PCI
  passthrough for client-facing compatibility.
- This keeps Server 2 stable for demonstrations while Server 1 remains the
  mediated-vGPU engineering proof.

## 2026-05-01 - Stop Runtime Work On Connectivity Failure

Decision:

- Do not attempt deployment, rollback, VM attach, or guest cleanup while
  `M08-E1` is active.

Reason:

- Current workstation has no route to Server 2 host `10.25.33.20` or target VM
  `10.25.33.21`.
- Without live host/VM inventory, any runtime conclusion would be historical
  rather than current evidence.
