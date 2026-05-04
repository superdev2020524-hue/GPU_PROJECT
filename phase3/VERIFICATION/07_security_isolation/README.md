# 07 - Security And Isolation

## Purpose

Prepare the mediated vGPU layer for tenant and cloud-facing use.

## Current Status

Milestone 07 is complete for the defined security/isolation gate after the
2026-04-30 refresh. It does not claim production security or full tenant
hardening.

The refresh re-ran the live malformed-socket gate, VM-6 quarantine/recovery,
VM-6 rate-limit/recovery, VM-10 cross-VM non-poisoning, and final preservation
after the TensorFlow guest-shim launch-layout correction. TensorFlow bounded GPU
training now passes earlier in `05_second_framework_gate`, and the refreshed M07
record no longer treats TensorFlow GPU readiness as blocked.

## Scope

- trusted vs untrusted tenant assumptions;
- malformed request handling;
- mediator request bounds;
- MMIO/BAR access policy;
- IOMMU expectations;
- unauthorized memory access prevention;
- abusive VM quarantine or kill behavior;
- recovery without host reboot where possible.

## Closure Criteria

- malformed request tests fail safely;
- unauthorized socket/CUDA-header abuse is blocked or rejected;
- mediator remains alive under bad input;
- logs identify the offending VM/process;
- recovery behavior is documented and repeatable.

## Production-Hardening Candidate

Current guests expose vGPU BAR0/BAR1 resources as `0666` so non-root CUDA/Ollama
processes can use the experimental transport. This is documented as an
engineering trust assumption, not as production tenant isolation. A narrower
group/device-policy or hypervisor/IOMMU model should replace it before
production tenant use.

An opt-in group policy is now available in `guest-shim/install.sh`:

```bash
sudo VGPU_BAR_ACCESS_MODE=group VGPU_BAR_GROUP=vgpu \
  VGPU_BAR_USERS="ollama test-10" ./install.sh
```

This mode is not the live VM-10/VM-6 default yet. It must be deployed first on a
non-baseline VM or during a reversible maintenance window, then followed by raw
CUDA/CuPy preservation.

Pilot status: group mode has been applied on non-baseline Test-4 only. It
proved BAR file narrowing (`root:vgpu 0660`), blocked an unauthorized user, and
allowed authorized `test-4`/`ollama` BAR opens. Test-10 CuPy remained green
afterward.
