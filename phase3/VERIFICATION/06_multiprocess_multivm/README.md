# 06 - Multi-Process And Multi-VM

## Purpose

Move from single-workload success to virtualization behavior.

## Scope

- two processes in one VM;
- multiple VMs when available;
- priority scheduling;
- fairness policy;
- memory pressure;
- long-running plus short interactive workload;
- cancellation and cleanup;
- mediator health under load.

## Closure Criteria

- no cross-process data leakage;
- no stale status from one process affects another;
- priority policy is observable;
- one failed workload does not poison the mediator;
- metrics identify which VM/process owns work.
