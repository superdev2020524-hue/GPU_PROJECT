# Verified State on 2026-03-28

This file records what was personally verified live in the 2026-03-28 session, not just what older docs claimed.

## 1. VM live checks

Verified from the workstation using the existing VM connection path:

- VM host responded as `test4-HVM-domU`
- user was `test-4`
- `systemctl is-active ollama` returned `active`
- `curl http://127.0.0.1:11434/api/tags` returned models, including:
  - `tinyllama:latest`
  - `llama3.2:1b`

Latest GPU-mode discovery line observed in the current boot:

- `library=CUDA`
- `compute=9.0`
- description `HEXACORE vH100 CAP`

That means:

- Checkpoint A passes
- Checkpoint B passes
- old E2 (`compute=8.9`) is not the active story in this session

## 2. VM library state

Verified:

- `/usr/lib64/libvgpu-cuda.so` exists
- `/usr/local/lib/ollama/cuda_v12/libcublas.so.12 -> /opt/vgpu/lib/libvgpu-cublas.so.12`
- `/usr/local/lib/ollama/cuda_v12/libcublasLt.so.12 -> libcublasLt.so.12.3.2.9`
- `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so` exists and is recent

Important implication:

- mediated `libvgpu-cublas` routing is in place
- the old `cublasCreate_v2` issue is no longer the main blocker

## 3. Host live checks

Verified directly on dom0:

- host responded as `xcp-ng-syovfxoz`
- user was `root`
- `mediator_phase3` was running
- `/tmp/mediator.log` existed and had current traffic

Current host log result:

- no live `401312`
- no live `INVALID_IMAGE`
- no live `rc=700`
- current mediator lines showed success for:
  - `call_id=0xb5`
  - `call_id=0x26`

Important implication:

- the older E1 and E4 signatures are not the strongest current live blocker

## 4. Short preflight result

Verified by running:

- `phase3/run_preflight_gemm_ex_vm.sh`

Observed result:

- mediated alloc/HtoD occurred
- `cublasGemmEx()` returned success
- `cuCtxSynchronize` after each GemmEx returned `0`
- final result was `PREFLIGHT_OK all GemmEx + sync passed`

Important implication:

- the mediated CUDA path is not fundamentally broken
- the mediated cuBLAS GemmEx path is healthy enough for short tests
- Phase 3 is beyond the "can we even talk to the GPU?" stage

## 5. Latest full-load failure signature

The strongest currently relevant full-load failure in the VM journal was:

- model load reached `1.00`
- `mmq_x_best=0`
- `mmq.cuh:3884: fatal error`
- `error loading llama server`
- runner terminated with abort / core-dumped wording

Interpretation:

- this matches E5, not E1
- the crash is in the GGML CUDA MMQ / graph-reserve path
- it is not best explained as a transport failure or mediator Gemm failure

## 6. Coredump status

Checked:

- `coredumpctl list --no-pager`

Result:

- `No coredumps found.`

Implication:

- if the next bounded full-load run crashes again, improving actual core capture is still valuable

## 7. Correct high-level assessment

The current state should be summarized like this:

- GPU mode works
- compute capability reporting works
- short mediated GemmEx works
- old E1 invalid-fatbin and E4 illegal-address stories are not the strongest live blockers
- the best current blocker is E5: MMQ / graph reserve during full model init
