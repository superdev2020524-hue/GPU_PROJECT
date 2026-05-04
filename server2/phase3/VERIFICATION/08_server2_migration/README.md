# 08 - Server 2 Migration

## Purpose

Prepare a controlled path from Server 1 engineering success to Server 2
client-facing deployment.

## Scope

- Keep Server 2 stable for demonstrations until a cutover is explicitly chosen.
- Use the Server 2 registry only (`server2/phase3/`), preserving the Server 1
  root `phase3/` tree.
- Prefer the verified passthrough fast path for Server 2 client compatibility:
  real GPU attached to the VM, no mediated CUDA/NVML shims active in the final
  passthrough guest.
- Define deployment, rollback, and client-demo verification steps.

## Current Status

M08 is open.

Current active error: `M08-E1` workstation cannot reach Server 2 host
`10.25.33.20` or target VM `10.25.33.21` (`No route to host`, ping loss).

## Closure Criteria

- Server 1 preserved baseline is recorded before Server 2 work.
- Server 2 host and VM are reachable or an approved alternate access path is
  documented.
- Server 2 current mode is classified: stable passthrough, mediated candidate,
  or unavailable.
- Deployment and rollback checklists are complete.
- Client-facing verification commands are repeatable and do not depend on hidden
  manual state.
