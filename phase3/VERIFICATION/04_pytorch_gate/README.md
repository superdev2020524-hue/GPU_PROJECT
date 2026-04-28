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

Milestone 04 is complete.

Entry baseline:

- Milestone 00 Plan A, Plan B, and Plan C are passing after the Plan C gate
  invocation fix.
- Milestone 01 raw CUDA gate is passing.
- Milestone 02 API audit consistency is passing.
- Milestone 03 memory/sync/cleanup gate is passing.

Closure result:

- Final PyTorch gate passed 3/3 fresh-process runs:
  `/tmp/m04_pytorch_probe_64k_final_repeat.json` -> `overall_pass=True`.
- Final serial preservation passed:
  Plan A, Plan B, Plan C, Milestone 01 raw CUDA, Milestone 02 API audit, and
  Milestone 03 4 MiB async/mixed memory/sync.
- M03 BAR1 async-copy preservation regression found during closure was repaired
  by reducing mediated BAR1 copy chunks to 64 KiB before the final chain.
