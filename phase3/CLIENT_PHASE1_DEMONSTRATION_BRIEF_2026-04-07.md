# Phase 1 Demonstration Brief (2026-04-07)

This document is the short client-facing proof package for the current Phase 1 result.

Here, `demonstration` means the concrete evidence set you can show the client:

- what was run,
- what passed,
- and why the result is proven to be the real GPU-backed `shmem` path rather than a CPU fallback or `BAR1` bulk-transfer fallback.

## 1. Current statement

As of 2026-04-07, the current approved Phase 1 milestone is passing on `test-10`:

- `Plan A` passes.
- `Plan B` passes.
- both were re-run fresh today.
- both were re-proven on the GPU-backed path.
- both were re-proven with `data_path=shmem`.

## 2. Fresh proof artifacts

### Plan A

- report: `/tmp/phase1_milestone_gate_post_fix_check.json`
- model: `qwen2.5:0.5b`
- result: `overall_pass=true`

Fresh measured results:

- `A1_exact_string`: pass, `HTTP 200`, `58.827s`
- `A2_arithmetic`: pass, `HTTP 200`, `12.334s`
- `A3_json_shape`: pass, `HTTP 200`, `36.207s`
- `cold`: pass, `HTTP 200`, `23.910s`
- `warm`: pass, `HTTP 200`, `22.266s`
- `keep_loaded`: pass
- `force_unload`: pass

### Plan B

- report: `/tmp/phase1_plan_b_tiny_gate_post_fix_check.json`
- model: `tinyllama:latest`
- result: `overall_pass=true`

Fresh measured results:

- `B1_cold_residency_pin`: pass, `HTTP 200`, `39.398s`
- `B2_warm_arithmetic_strict`: pass, `HTTP 200`, strict `{"sum": 95}`, `36.088s`
- `B3_warm_json_strict`: pass, `HTTP 200`, strict `{"ok": true, "n": 7}`, `50.081s`
- `B4_force_unload`: pass

## 3. Why this is proven GPU mode

The matching live runtime logs show:

- `gpu memory ... library=CUDA`
- `model weights device=CUDA0`
- `kv cache device=CUDA0`
- `compute graph device=CUDA0`
- `llama_model_load_from_file_impl: using device CUDA0`
- `load_tensors: offloaded ... layers to GPU`

The matching host-side mediator logs also show real physical GPU replay for `vm=10`:

- `cuLaunchKernel SUCCESS: kernel executed on physical GPU`

Therefore this is not a CPU-mode result.

## 4. Why this is proven `shmem` and not `BAR1`

The matching fresh runtime logs for both the re-run `Plan A` and re-run `Plan B` show:

- `SHMEM_REG ... pagemap_st=ok`
- `Connected (vm_id=10) data_path=shmem status_from=BAR1-status-mirror`

Interpretation:

- bulk transfer path: `shmem`
- status mirror path: `BAR1-status-mirror`
- this is not `data_path=BAR1`

## 5. Important note about the earlier false alarm

The earlier alarming result came from a standalone preflight path that was not running under the same privilege and environment shape as the live `ollama` service.

That mismatch was corrected by making the preflight run under a transient service-equivalent systemd context instead of a plain interactive shell context.

After that correction:

- the preflight returned to the expected path,
- `Plan A` passed again,
- `Plan B` passed again.

## 6. Short client wording

If you need one short sentence for the client, use this:

`Phase 1 is currently passing on the live Test-10 baseline for both Plan A and Plan B, and today's fresh reruns proved that the results are coming from the GPU-backed shared-memory path rather than from CPU mode or BAR1 bulk fallback.`
