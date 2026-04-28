# Gate - Milestone 04 PyTorch

## Gate Name

`phase3_pytorch_gate`

## Purpose

Prove whether PyTorch can use the mediated vGPU path for basic framework
operations after the lower CUDA and memory/sync gates are stable.

## Required Coverage

The bounded PyTorch gate must cover:

- Python and PyTorch import;
- `torch.cuda.is_available()`;
- CUDA device count, name, capability, and memory query;
- CPU tensor to CUDA tensor copy;
- CUDA tensor to CPU copy with byte/value verification;
- elementwise CUDA operation;
- CUDA matrix multiply;
- a small `torch.nn` inference on CUDA;
- repeated warm execution in fresh processes;
- process cleanup and post-run `/api/ps` / mediator health.

## Pass Criteria

- All gate cases pass on the mediated path.
- The output values are verified, not only API return codes.
- Host mediator evidence shows no `sync FAILED`,
  `CUDA_ERROR_ILLEGAL_ADDRESS`, unsupported protocol surprises, or invalid
  handle failures during the bounded gate.
- Plan A is rechecked after any runtime/shared-layer change.
- Prior milestone preservation is reported separately before closure.

## Fail Criteria

- PyTorch cannot import or is not installed in the VM.
- PyTorch does not see CUDA and the cause is not cleanly classified.
- Any tensor operation returns success but produces wrong values.
- A PyTorch process poisons later CUDA or Ollama behavior.
- Any fix regresses Plan A or the raw CUDA gate.

## Initial Probe Shape

Start with an environment probe:

- Python version;
- PyTorch install/version;
- linked CUDA version reported by PyTorch;
- `torch.cuda.is_available()`;
- one minimal CUDA allocation if available.

If PyTorch is missing, classify that as an environment/setup blocker before
changing runtime code.
