# Decisions - Milestone 07 Security And Isolation

## 2026-04-29 - Start With Malformed Request Boundary

- Decision: start M07 with the mediator/vGPU-stub request boundary instead of a
  broad production-security review.
- Reason: malformed guest-controlled request fields are the nearest reachable
  security/isolation risk in the current implementation, and they can be tested
  with bounded probes while preserving the M06 baseline.
- Rejected alternatives: begin with formal IOMMU policy or cloud isolation
  claims before proving the existing request parser survives bad input.
- Reversal/removal condition: if source audit proves malformed requests cannot be
  safely injected without risking the live baseline, switch first to an offline
  parser/unit harness and document live injection as deferred.

## 2026-04-29 - BAR Policy Classification

- Decision: treat guest BAR `0666` access as an explicit current engineering
  trust assumption for the experimental non-root shim path.
- Reason: the live guests and `guest-shim/install.sh` intentionally make
  `resource0` and `resource1` writable so non-root CUDA/Ollama processes can use
  the BAR0/BAR1 transport. Changing this mid-M07 would likely break already
  preserved framework gates unless a narrower access model is designed and
  deployed.
- Production hardening candidate: replace world-writable BAR access with a
  group/device-policy model or stronger hypervisor/IOMMU mediation.
- Reversal/removal condition: if any current M07 probe proves that BAR `0666`
  enables host crash, mediator corruption, or cross-VM poisoning, promote that
  as a new active M07 blocker instead of deferring it.

## 2026-04-29 - Fix Quarantine Reload

- Decision: fix DB-to-watchdog quarantine sync in the mediator rather than only
  documenting the gap.
- Reason: `vgpu-admin quarantine-vm` already claimed that new GPU submissions
  would be rejected, but the live mediator config reload did not apply the DB
  quarantine field to the watchdog state.
- Rejected alternatives: testing rate limit only, or requiring a full mediator
  restart for every quarantine state change.
- Reversal/removal condition: if serial preservation shows this change regresses
  earlier gates, repair the regression before interpreting any further M07
  behavior.

## 2026-04-29 - Add Opt-In Group BAR Policy

- Decision: add an opt-in group-based BAR permission mode to
  `guest-shim/install.sh`, while preserving the historical `0666` compatibility
  default.
- Reason: the M07 gate proved the mediator-side security boundary, but the live
  guest baseline still exposes BAR0/BAR1 to all guest users. A group mode gives
  us a bounded path to test narrower in-guest access without breaking the
  already-green VM-10/VM-6 preservation baseline.
- Implementation: `VGPU_BAR_ACCESS_MODE=group` makes the installer use
  `root:${VGPU_BAR_GROUP}` and `0660` for `resource0`/`resource1`; optional
  `VGPU_BAR_USERS` adds selected service users to that group. Default remains
  `VGPU_BAR_ACCESS_MODE=world`.
- Deployment rule: do not enable group mode on VM-10 or VM-6 until it has first
  passed on a non-baseline VM or a clearly reversible maintenance window.
- Reversal/removal condition: if group mode breaks non-root CUDA/Ollama transport
  even after adding the intended service users to the group, keep the mode
  documented as experimental and leave the compatibility default in place.
