# Evidence - Milestone 08 Server 2 Migration

## Current Session Findings

M08 has started under the Server 2 isolation rule. The authoritative working
tree for this milestone is `server2/phase3/`.

## Source/Document Evidence

- `server2/README.md` defines Server 2 as a full Phase 3 mirror for the second
  host, with host `10.25.33.20`.
- `server2/phase3/SERVER2_ISOLATION_AND_MISSION_RULES.md` requires all Server 2
  edits to stay under `server2/` and identifies:
  - primary host: `10.25.33.20`;
  - target VM: `10.25.33.21`;
  - final exposure method: real GPU PCI passthrough;
  - final guest policy: no mediated CUDA/NVML shims active in the deployed
    passthrough VM.
- `server2/HOST2_PASSTHROUGH_FAST_PATH.md` documents the historical fast path:
  `attach_passthrough_vm.sh <vm-uuid>`, then `fix_pci_ids_vm.py` for clean VMs
  or `clean_passthrough_vm.py` for old mixed mediated/passthrough VMs.
- `server2/CUSTOMER_GPU_VERIFICATION_CHECKLIST.md` defines the client-facing
  commands for `lspci`, `nvidia-smi`, Ollama, TensorFlow, and PyTorch.

## Server 1 Preservation Baseline

Server 1 refreshed M07 closure is the starting baseline:

- Plan A: `/tmp/m07_final_after_tf_3param_planA.json` -> pass.
- Raw CUDA: `/tmp/m07_final_after_tf_3param_m01.json` -> pass.
- PyTorch: `/tmp/m07_final_after_tf_3param_pytorch.json` -> pass.
- CuPy: `/tmp/m07_final_after_tf_3param_cupy.json` -> pass.
- TensorFlow: `/tmp/m07_current_preserve_tensorflow_after_3param_fix.json` ->
  pass.
- M07 malformed: `/tmp/m07_final_after_tf_3param_malformed.json` -> pass.

No root `phase3/` files were modified for M08.

## Live Connectivity Attempt

Commands attempted from the workstation:

```bash
ssh root@10.25.33.20
ssh root@10.25.33.21
ping -c 2 -W 2 10.25.33.20
ping -c 2 -W 2 10.25.33.21
```

Observed:

- SSH to `10.25.33.20`: `No route to host`.
- SSH to `10.25.33.21`: `No route to host`.
- Ping to both addresses: 100% packet loss.

Route table at time of failure:

```text
default via 192.168.119.2 dev ens33 proto dhcp src 192.168.119.128 metric 100
10.10.20.0/24 dev wg0 proto kernel scope link src 10.10.20.14
172.17.0.0/16 dev docker0 proto kernel scope link src 172.17.0.1
192.168.119.0/24 dev ens33 proto kernel scope link src 192.168.119.128 metric 100
```

## Active Error

`M08-E1`: Server 2 host/VM unreachable from the workstation.

## Alternate Path Check From Server 1

Because Server 1 dom0 (`10.25.33.10`) is on the same `10.25.33.0/24` network,
it was tested as a possible jump point.

Server 1 route table includes:

```text
default via 10.25.33.254 dev xenbr0
10.25.33.0/24 dev xenbr0 proto kernel scope link src 10.25.33.10
```

Observed from Server 1:

- `ping 10.25.33.20`: 100% packet loss.
- `ping 10.25.33.21`: 100% packet loss.
- Neighbor table:
  - `10.25.33.20 dev xenbr0 FAILED`
  - `10.25.33.21 dev xenbr0 FAILED`
  - ARP entries remained `<incomplete>`.

Interpretation: `M08-E1` is not only the workstation route. Server 2 host/VM are
not visible from a same-subnet Server 1 host either, so the current blocker is
likely Server 2 power/network/host availability or changed addressing.

## Next Single Step

Restore or provide an access path to Server 2, then rerun the read-only
connectivity, host inventory, and guest inventory gates before any deployment or
rollback action.
