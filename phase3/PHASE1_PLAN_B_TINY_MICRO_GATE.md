# Phase 1 Plan B Tiny Micro-Gate

Purpose: evaluate `tinyllama:latest` on a preserved `Plan A` baseline without confusing transport health with strict output-shape compliance.

This gate is valid only after:

1. `Plan A` passes.
2. GPU mode is proven.
3. The live artifact path is proven.

## What this gate answers

This gate separates three outcomes:

1. **Pass**
   - Tiny satisfies both semantic correctness and strict output-shape compliance.
2. **Strict-shape failure**
   - Tiny produces prompt-related or prefix-correct output, but does not satisfy the exact required format.
   - This weakens transport-corruption theories and shifts focus to prompt/stop/budget/compliance behavior.
3. **Semantic failure**
   - Tiny does not satisfy even the looser semantic checks.
   - This keeps broader correctness issues active.

## Requests used

1. **Arithmetic**
   - prompt: `What is 37 + 58? Reply with digits only.`
   - options: `temperature=0`, `seed=1234`, `num_predict=16`
   - semantic pass: response contains `95`
   - strict pass: response is exactly `95`

2. **JSON**
   - prompt: `Respond with JSON only: {"ok":true,"n":7}`
   - options: `temperature=0`, `seed=1234`, `num_predict=32`
   - semantic pass: the response starts with a parseable JSON object containing `{"ok": true, "n": 7}`
   - strict pass: that JSON object is the entire response except for trailing whitespace

## Run command

```bash
cd /home/david/Downloads/gpu/phase3
python3 phase1_plan_b_tiny_micro_gate.py \
  --base-url http://10.25.33.110:11434 \
  --output /tmp/phase1_plan_b_tiny_micro_gate_report.json
```

## Interpretation rule

- If arithmetic and JSON both fail semantically, keep the active error in broader Tiny correctness.
- If at least one case passes semantically but strict checks fail, classify the active error as Tiny strict output-shape / compliance failure.
- If both strict checks pass, the Tiny micro-gate passes and the next `Plan B` step can move to a larger written gate.
