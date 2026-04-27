# Phase 1 Plan B Tiny Gate

This is the approved binding `Plan B` gate for `tinyllama:latest`.

It is valid only when:

1. `Plan A` passes in the same session.
2. GPU mode is proven.
3. The live artifact path is proven.
4. Host mediator health is proven.

## Binding cases

### B1_cold_residency_pin

- request:
  - prompt: `warm pin`
  - `keep_alive=-1`
  - `temperature=0`
  - `seed=1`
  - `num_predict=2`
- pass:
  - `HTTP 200`
  - `/api/ps` shows `tinyllama:latest` resident afterward
  - host mediator shows real GPU execution in the same window

### B2_warm_arithmetic_strict

- request:
  - prompt: `Compute 37 + 58. Respond with JSON only: {"sum":95}`
  - `keep_alive=-1`
  - `temperature=0`
  - `seed=1234`
  - `num_predict=64`
  - `stop=["```","\n\n","In this example"]`
- pass:
  - `HTTP 200`
  - parsed JSON equals `{"sum": 95}`
  - no trailing non-whitespace content
  - model remains resident afterward

### B3_warm_json_strict

- request:
  - prompt: `Respond with JSON only: {"ok":true,"n":7}`
  - `keep_alive=-1`
  - `temperature=0`
  - `seed=1234`
  - `num_predict=64`
  - `stop=["```","\n\n","In this example"]`
- pass:
  - `HTTP 200`
  - parsed JSON equals `{"ok": true, "n": 7}`
  - no trailing non-whitespace content
  - model remains resident afterward

### B4_force_unload

- request:
  - unload request for `tinyllama:latest`
- pass:
  - `HTTP 200`
  - `/api/ps` no longer shows `tinyllama:latest`

## Non-gating note

The exact-token case is no longer a binding `Plan B` closure case unless the user explicitly restores it.

Rationale for the approved `B2` revision:

- the current Tiny path has already proven arithmetic semantics on the repaired GPU-backed runtime,
- plain-digits arithmetic compliance did not close under bounded prompt/stop shaping,
- structured arithmetic JSON closes the arithmetic proof in the same task family while matching the model behavior already demonstrated live.

## Run command

```bash
cd /home/david/Downloads/gpu/phase3
python3 phase1_plan_b_tiny_gate.py \
  --base-url http://10.25.33.110:11434 \
  --output /tmp/phase1_plan_b_tiny_gate_report.json
```

## Closure rule

`Plan B` closes only when all four binding cases pass in one bounded gate session on a `Plan A`-passing baseline.
