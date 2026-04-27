# Phase 1 Plan C Client Gate

This is the binding `Plan C` gate for normal client-facing usage.

Target model:

- `qwen2.5:3b`

Purpose:

- prove that standard one-shot user prompts now return correct answers on the live GPU-backed Ollama service,
- do so without mutating the proven `Plan A` / `Plan B` models,
- leave the service in a clean state so `Plan A` and `Plan B` can be re-verified immediately afterward.

## Preconditions

This gate is valid only when:

1. `Plan A` and `Plan B` remain the preserved infrastructure milestones.
2. The live artifact path is already proven.
3. GPU mode is already proven.
4. The gate is run **serially**, not concurrently with `Plan A` or `Plan B`.
5. `/api/ps` is forced clean before the gate starts.

## Binding cases

The checked-in runner uses the real `ollama run` CLI path with `OLLAMA_HOST` set to the target service. This is the strict user-facing path for `Plan C`.

### C1_small_arithmetic_cli_style

- prompt: `What is 4 + 8? Reply with digits only.`
- pass:
  - CLI return code `0`
  - stdout is exactly `12`
  - model is resident afterward

### C2_large_arithmetic_cli_style

- prompt: `What is 444 + 8? Reply with digits only.`
- pass:
  - CLI return code `0`
  - stdout is exactly `452`
  - model is resident afterward

### C3_second_large_arithmetic_cli_style

- prompt: `What is 444 + 18? Reply with digits only.`
- pass:
  - CLI return code `0`
  - stdout is exactly `462`
  - model is resident afterward

### C4_reference_arithmetic_cli_style

- prompt: `What is 37 + 58? Reply with digits only.`
- pass:
  - CLI return code `0`
  - stdout is exactly `95`
  - model is resident afterward

### C5_force_unload

- request:
  - unload `qwen2.5:3b`
- pass:
  - `HTTP 200`
  - `/api/ps` no longer shows `qwen2.5:3b`

## Serial isolation rule

Do not run `Plan A`, `Plan B`, and `Plan C` gates in parallel.

After any `Plan C` proof:

1. unload `qwen2.5:3b`,
2. confirm `/api/ps` is empty,
3. then re-run `Plan A`,
4. then re-run `Plan B`.

This rule exists because concurrent cross-model gate runs can create false regression signals that do not reflect the true preserved baseline.

## Run command

```bash
cd /home/david/Downloads/gpu
python3 phase3/phase1_plan_c_client_gate.py \
  --base-url http://127.0.0.1:11434 \
  --output /tmp/phase1_plan_c_client_gate_report.json
```

Run it on `VM10` or any host that has the `ollama` CLI installed.

## Direct operator checks

Run these directly on `VM10` with standard Ollama commands:

```bash
ollama run qwen2.5:3b "What is 4 + 8? Reply with digits only."
ollama run qwen2.5:3b "What is 444 + 8? Reply with digits only."
ollama run qwen2.5:3b "What is 444 + 18? Reply with digits only."
```

Expected outputs:

- `12`
- `452`
- `462`

## Closure rule

`Plan C` closes only when all five binding cases pass in one bounded serial session and the same session preserves `Plan A` and `Plan B` after the forced-clean handoff.
