# Phase 1 — executive summary (Mar 22, for review / handoff)

## Goal (Phase 1 milestone)

Ollama in the **VM** completes **GPU-mode inference** end-to-end: guest → vGPU path → **host GPU** → response.

## Current blockers (evidence-based)

| ID | Symptom | Layer | Fix owner |
|----|---------|-------|-----------|
| **E1** | Host **`module-load`** **`data_len=401312`** → **`CUDA_ERROR_INVALID_IMAGE`** | **cuBLASLt** fatbin / driver acceptance on **H100** | **dom0** — align **`libcublasLt.so.12`** (and related) per **`FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`** |
| **E3** | **`llama runner terminated`** **`exit status 2`** after long HtoD / around **graph_reserve** | Native **GGML/CUDA** in runner | **VM** — **coredump** + **`gdb bt full`**; optional reduce **`num_ctx`** / layers for faster repro |

**Note:** Rebuilding **`libggml-cuda.so`** (sm_90) fixes **discovery CC** and GGML kernels; it does **not** replace the **401312** **cuBLASLt** blob (documented in **`ERROR_TRACKING_STATUS.md`**).

## What we ran today (monitored)

- **5 min** `curl` **`/api/generate`** (`tinyllama`) with **60 s** sampling → **`curl` exit 28**, **0 bytes** (no HTTP body within 5 min).
- **Journal** in that window: **no new** `model load progress` lines in the filtered tail (load may be slower than 5 min or logging delayed).

## Required process (no blind long runs)

**`INCREMENTAL_RUN_MONITORING.md`** — explicit approval, **3–5 min** VM+host samples, **early stop**.

## Fastest next actions (ordered)

1. **dom0:** Plan **libcublasLt** / driver-aligned libraries for **H100** (E1).
2. **VM:** Ensure **coredumps** enabled; on next **exit 2**, **`coredumpctl dump`** + **`gdb`**.
3. **VM:** Retry generate with **client timeout ≥** measured load time **or** **preload** model; use **incremental** monitoring.

---

*Technical facts only — not legal advice.*
