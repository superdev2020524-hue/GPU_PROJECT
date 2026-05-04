# Baseline - Milestone 05 Second Framework Gate

## Entry Condition

Milestone 05 starts only after Milestone 04 PyTorch gate closure and final serial
preservation.

## Preserved Prior Milestones

- `00_preserve_ollama_baseline`
  - Plan A:
    `/tmp/phase1_milestone_gate_serial_00_after_m04_64k_final.json` ->
    `overall_pass=True`.
  - Plan B:
    `/tmp/phase1_plan_b_serial_00_after_m04_64k_final.json` ->
    `overall_pass=True`.
  - Plan C:
    `/tmp/phase1_plan_c_serial_00_after_m04_64k_final.json` ->
    `overall_pass=True`.
- `01_general_cuda_gate`
  - `/tmp/phase3_general_cuda_gate_serial_01_after_m04_64k_final.json` ->
    `overall_pass=True`, Driver API 5/5 and Runtime API 5/5.
- `02_api_coverage_audit`
  - `/tmp/phase3_api_audit_serial_02_after_m04_64k_final.json` ->
    `overall_pass=True`.
- `03_memory_sync_cleanup`
  - `/tmp/async_stream_event_probe_after_m04_64k_final.json` ->
    `overall_pass=True`, `pass_count=3`, `runs=3`, `bytes_per_run=4194304`,
    `chunk_cap_bytes=65536`.
- `04_pytorch_gate`
  - `/tmp/m04_pytorch_probe_64k_final_repeat.json` ->
    `overall_pass=True`, `run_passes=[true,true,true]`.

## Live Artifact Baseline

- Guest `cuda_transport.c` deployed SHA:
  `81e3017a6ffdb4ba182c79ba48199782fa86f5c32e0eef983b5fdea316251be4`.
- Guest `/opt/vgpu/lib/libvgpu-cuda.so.1` deployed SHA:
  `c97cfb9e619bbb110a72f4b94505bbab4a5ae88350bc207a52051e5268fc835b`.
- PyTorch environment remains isolated at `/mnt/m04-pytorch/venv`.
