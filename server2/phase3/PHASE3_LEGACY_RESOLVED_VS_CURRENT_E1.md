# PHASE3 — “resolved before” vs **current E1** (401312)

*Why this exists:* Some PHASE3 files say INVALID_IMAGE / GPU path issues are **fully resolved**. Live logs + **`cuobjdump`** show **E1** is still a **real** blocker until the **blob** matches **H100**.

---

## What **was** resolved (still valid, different layer)

| Topic | Typical doc | What it fixed |
|-------|-------------|----------------|
| **Discovery / CPU fallback** | `BREAKTHROUGH_SUMMARY.md`, symlinks, `OLLAMA_LLM_LIBRARY` | Ollama sees **CUDA** / **libggml-cuda** |
| **cudart version / fatbin registration** | Version script, `libcudart` symlinks | Loader / registration issues |
| **Host mediator context** | `HOST_FIX_MODULE_LOAD_PRIMARY_CTX.md`, `OLLAMA_VGPU_REVISIONS_STATUS.md` | **Some** `cuModuleLoadFatBinary` paths: **primary context**, **raw `0xBA55ED50` then wrapper** — fixes **wrong call shape**, not wrong **SASS arch** inside the blob |
| **Transport / chunks** | `E1_ERROR_TRACING_NEXT_METHODS.md` Method 4 | **401312** reassembly **matches**; not a chunk-length bug for logged events |

---

## What **OLLAMA_VGPU_REVISIONS_STATUS.md (Mar 15)** actually claimed

- It says **module-load `rc=0`** and **“INVALID_IMAGE is resolved”** for a **session** where that held.
- **`ERROR_TRACKING_STATUS.md`** notes **doc drift:** later **Mar 19–22** logs still show **`data_len=401312` → `rc=200` `INVALID_IMAGE`**. So the **401312 Lt** load is **not** the same outcome as **“first small fatbin ok”** in a single sentence.

---

## What **current E1** is (authoritative)

- **`FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`** + host **`cuobjdump`** on **`/tmp/fail401312.bin`**: **`arch = sm_80`**, Ampere GEMM names — **not** Hopper.
- Standalone host test: **`CUDA_ERROR_NO_BINARY_FOR_GPU`** — driver has **no** image for **H100** for **that** file.
- **Mitigations in repo:** **`gpu_properties.h` 9.0**, **`fetch_gpu_info()`** CC force, **VM `libcublas` 12.3.2.9** align — **do not** change the **embedded sm_80** slice inside that **Lt** fatbin selection by themselves.

---

## **Next step** (unchanged, after reading PHASE3)

1. **Do not** rely on **Feb 2026 “complete solution”** alone for **E1** — it lists **infrastructure** fixes, not **Lt kernel package** for **401312**.
2. **Do** use **NVIDIA driver / cuBLASLt compatibility matrix** for **H100** and try a **different** **`libcublas`/`libcublasLt`** build if **401312** stays **sm_80** after re-dump.
3. **Host** `cuda_executor` **raw/wrapper** path: keep per **`OLLAMA_VGPU_REVISIONS_STATUS.md`** — it **does not** replace **wrong-arch** content.

---

*See **`ERROR_TRACKING_STATUS.md`** § “Doc drift to ignore” and **`COMPREHENSIVE_REVIEW_RESULTS.md`** (older; superseded for E1 by Mar 21+ analysis).*
