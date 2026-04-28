# 04 - PyTorch Gate

## Purpose

Validate the mediated vGPU path with PyTorch after the raw CUDA gate is stable.

## Gate Coverage

- `torch.cuda.is_available()`;
- device name and memory query;
- tensor allocation;
- tensor HtoD/DtoH;
- elementwise operation;
- matrix multiply;
- small neural network inference;
- repeated warm execution;
- process restart cleanup.

## Closure Criteria

- all cases pass on the mediated path;
- mediator evidence shows physical GPU execution;
- unsupported behavior fails cleanly;
- Plan A still passes after PyTorch testing.

## Current Status

Milestone 04 is active.

Entry baseline:

- Milestone 00 Plan A, Plan B, and Plan C are passing after the Plan C gate
  invocation fix.
- Milestone 01 raw CUDA gate is passing.
- Milestone 02 API audit consistency is passing.
- Milestone 03 memory/sync/cleanup gate is passing.

Current first step:

Run a bounded PyTorch environment probe and promote exactly one active error if
the probe finds a blocker.
