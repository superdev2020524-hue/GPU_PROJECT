# Gate - Milestone 07 Security And Isolation

## Scope

Milestone 07 prepares the mediated vGPU layer for tenant and cloud-facing use.
It does not claim production security. It defines the current trust boundary,
then proves that reachable malformed or abusive guest inputs fail safely without
corrupting host state or poisoning other VMs.

## Trust Assumptions

- The host and mediator are trusted.
- VM tenants are not fully trusted.
- Guest user processes can call intercepted CUDA APIs and may attempt malformed
  values through normal guest-visible paths.
- A malicious tenant with root inside a guest may be able to write to exposed
  BAR/MMIO resources if guest permissions allow it.
- M07 must separate current engineering trust assumptions from production
  requirements that need IOMMU or hypervisor policy enforcement.

## Required Gate Cases

1. Document the trusted/untrusted boundary for guest process, guest root, QEMU
   vGPU stub, mediator, executor, and physical GPU.
2. Audit request bounds for guest-controlled fields:
   `call_id`, `seq_num`, `vm_id`, `num_args`, `data_len`, inline args, payload
   length, result length, BAR1 offsets, and shmem registration.
3. Prove malformed request headers fail safely:
   invalid magic, unsupported `call_id`, oversized payload length, truncated
   payload, excessive `num_args`, and impossible `vm_id` where reachable.
4. Prove mediator remains alive and logs the offending VM/process after malformed
   request probes.
5. Prove a bad VM/probe does not poison a known-good VM gate.
6. Document existing rate-limit/quarantine behavior and test the least risky
   reachable rejection path.
7. Re-run required `00 -> 06` preservation after M07 probes or fixes.

## Pass Criteria

- Malformed probes return rejection or error, not process crash or host reboot.
- Mediator remains running after malformed probes.
- Logs identify the offending VM or socket.
- Test-10 and Test-6 known-good gates pass after malformed probes.
- No new `sync FAILED`, `CUDA_ERROR_ILLEGAL_ADDRESS`, unsupported-call crash,
  invalid-handle poisoning, or queue-depth leak appears as the active blocker.

## Fail Criteria

- Mediator exits, wedges, or requires host reboot.
- A malformed request corrupts another VM's state.
- A bad request causes a later known-good gate to fail.
- Logs cannot attribute the bad request to a VM/socket.
- Oversized or truncated payloads are accepted as valid work.

## Deferred Scope

- Formal adversarial security proof.
- Full IOMMU/hypervisor hardening.
- Host kernel attack resistance.
- Strong per-request priority enforcement beyond current pool/ownership
  observability.

These are production-hardening candidates. They should be recorded, but they do
not block M07 unless the current mediated path already exposes a reachable
crash/corruption condition.
