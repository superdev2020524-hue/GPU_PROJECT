# Active Error - Milestone 08 Server 2 Migration

## Current Lane

Milestone 08: Server 2 Migration / Product Demonstration Path.

## Current Plan A State

Server 1 Plan A is preserved from the refreshed M07 closure:

- `/tmp/m07_final_after_tf_3param_planA.json` -> `overall_pass=True`.

## Active Error

`M08-E1`: workstation cannot reach Server 2 host or target VM.

Evidence:

- `ssh root@10.25.33.20` -> `No route to host`.
- `ssh root@10.25.33.21` -> `No route to host`.
- `ping 10.25.33.20` -> 100% packet loss.
- `ping 10.25.33.21` -> 100% packet loss.
- Current route table has no working route to `10.25.33.0/24`.
- Server 1 dom0 (`10.25.33.10`) is on `10.25.33.0/24` but also cannot ping
  `10.25.33.20` or `10.25.33.21`; its neighbor table shows both as `FAILED`
  / ARP `<incomplete>`.

Impact:

No live Server 2 host or guest state can be trusted from this session beyond
historical documents. M08 cannot perform host inventory, VM inventory,
deployment, rollback, or client demo verification until reachability is restored
or an alternate access path is provided.

## Candidate Queue

- Server 2 may already be in the historically verified passthrough state, but
  this is historical until refreshed.
- Target VM `10.25.33.21` may be powered off or moved.
- Server 2 host may be powered off, disconnected, on a different address, or not
  reachable from the current lab network.
- Existing Server 2 guest may still contain mediated-path leftovers if it is an
  older mixed VM.

## Last Proven Checkpoint

Historical checkpoint from `server2/HOST2_PASSTHROUGH_FAST_PATH.md`: Server 2
passthrough was previously validated with guest `lspci`, `nvidia-smi`, Ollama
CUDA discovery, and a `qwen2.5:0.5b` generation. This is not current live
evidence.

## Closure Condition

`M08-E1` closes only when either:

- Server 2 host `10.25.33.20` and target VM `10.25.33.21` are reachable from the
  workstation and read-only inventory is captured; or
- the user provides an alternate approved access path and equivalent host/VM
  inventory evidence is captured.
