# 07 - Security And Isolation

## Purpose

Prepare the mediated vGPU layer for tenant and cloud-facing use.

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
- unauthorized memory access is blocked;
- mediator remains alive under bad input;
- logs identify the offending VM/process;
- recovery behavior is documented and repeatable.
