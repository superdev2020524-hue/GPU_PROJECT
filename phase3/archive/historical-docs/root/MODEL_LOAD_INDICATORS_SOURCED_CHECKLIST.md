# Model loading — checklist **sourced from Ollama + NVIDIA + public docs**

**Why this file exists:** An earlier checklist leaned on **this repo** and **live VM/host probes** without first anchoring items in **Ollama** and **NVIDIA** (and related) **published** material. That did **not** match the request to investigate those sources. This document corrects that: **each section cites where the indicator comes from.**

**How to use:**  
1. Use **§A–§C** for **vendor- and community-documented** signals.  
2. Use **§D** for **Phase3 / vGPU / mediator** (not “the whole world,” but **your** stack — explicit).  
3. Use **`MODEL_LOAD_INDICATORS_CHECKLIST_RESULTS.md`** for **last run** on your host/VM (point-in-time).

---

## A. Ollama (product / API / runtime) — documented indicators

**Sources (official / primary):**

- **List running models (`GET /api/ps`)** — fields `size`, `size_vram`, `expires_at`, etc.  
  - **https://docs.ollama.com/api/ps** (OpenAPI: models loaded into memory; `size_vram` = VRAM usage in bytes).  
- **Documentation index (discover all pages)** — **https://docs.ollama.com/llms.txt**  
- **Preload without prompt** — empty `POST` to `/api/generate` (or chat/embed) loads model; FAQ / issues discuss workflow.  
  - Example discussion: **https://github.com/ollama/ollama/issues/2431**  
- **Configurable model load timeout** — **`OLLAMA_LOAD_TIMEOUT`** introduced for long loads (was hardcoded ~10 min).  
  - **https://github.com/ollama/ollama/pull/4123**  
  - **https://github.com/ollama/ollama/issues/4350**  
  - Reported application bugs: **https://github.com/ollama/ollama/issues/6678** (`OLLAMA_LOAD_TIMEOUT` not applied in some versions — verify your build).  
- **Keep-alive / residency** — **`keep_alive`**, **`OLLAMA_KEEP_ALIVE`** affect how long a model **stays** loaded after a request, not the same as “load succeeded,” but relevant to whether you **see** a loaded model in `/api/ps`.  
  - Community writeups cite env + API; verify against your Ollama version.

| # | Indicator | What it means for “loading” |
|---|-----------|------------------------------|
| A1 | **`GET /api/ps`** returns **200** with expected schema | Server reachable; **loaded** models listed with **`size` / `size_vram`** per docs. |
| A2 | **`size` / `size_vram` stuck at 0** while you expect a load | **Not** defined in OpenAPI as “percent done” — usually means **still loading** or **not yet accounted**; combine with logs. |
| A3 | **`OLLAMA_LOAD_TIMEOUT`** | Server-side **abandon** of load if exceeded (see PR #4123 / issues). |
| A4 | **HTTP client timeout** (browser, `curl -m`) | **Independent** of A3: client can drop while server still loading → **`context canceled`** style failures (see GitHub issues on API timeout). |
| A5 | **Empty-body preload** (`/api/generate` with model, no prompt) | Documented pattern to **warm** load without doing inference (issues/FAQ). |
| A6 | **Journal / log: `model load progress`** (if enabled in your build) | Ollama logging of fractional load — **implementation detail**; not always in OpenAPI. |

---

## B. NVIDIA / CUDA driver API — documented indicators (GPU path)

**Sources (NVIDIA):**

- **Module management: `cuModuleLoadFatBinary`** — fat binary contains multiple SASS/PTX; driver selects arch.  
  - **https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__MODULE.html** (Module Management; see module load APIs).  
  - Legacy mirror often cited: developer.download.nvidia.com CUDA Driver API **group__CUDA__MODULE** pages.  
- **Fatbin / nvFatbin** — arch mismatch concepts (**`NVFATBIN_ERROR_*`**) when **building** fatbins; loading still depends on **valid image** for **GPU**.  
  - **https://docs.nvidia.com/cuda/nvfatbin/**  
- **`CUDA_ERROR_INVALID_IMAGE`** — driver rejects module image (invalid or incompatible content for load path).  
  - Discussed in NVIDIA API docs and forums; Stack Overflow **https://stackoverflow.com/questions/18844976/cuda-error-invalid-image-during-cumoduleload** (community; cross-check with official docs).  

| # | Indicator | What it means for “loading” |
|---|-----------|------------------------------|
| B1 | **`cuMemAlloc` / `cuMemcpyHtoD`** success | **Weights / buffers** actually reach **device memory** (Driver API). |
| B2 | **`cuModuleLoadFatBinary` → CUDA_SUCCESS** | **Kernel/module** image accepted for **this** GPU context. |
| B3 | **`cuModuleLoadFatBinary` → CUDA_ERROR_INVALID_IMAGE** | **Binary not accepted** — **not** “HTTP failed”; fix **fatbin/GPU arch/build** alignment (see nvFatbin + driver). |
| B4 | **Out-of-memory / `CUDA_ERROR_OUT_OF_MEMORY`** | VRAM pressure during alloc — load can fail **after** partial progress. |
| B5 | **Driver / GPU visibility (`nvidia-smi`)** | GPU **present** to host driver; unrelated to guest Ollama but **required** for bare-metal host execution. |

---

## C. Cross-cutting (memory, disk, OS) — standard ML inference indicators

**Sources:** General CUDA + Linux + Ollama issue trackers (not Ollama-specific only).

| # | Indicator | Notes |
|---|-----------|--------|
| C1 | **Disk:** GGUF readable, space, permissions | Ollama reads model from disk first. |
| C2 | **RAM:** enough for mmap / CPU tensors** | Before or besides VRAM. |
| C3 | **Process OOM killer** | dmesg / journal — load dies without clean API error. |
| C4 | **Concurrent requests** | Competing loads / VRAM (Ollama issues discuss contention). |

---

## D. Phase3-specific (vGPU guest → host mediator) — **not** in Ollama/NVIDIA generic docs

**Sources:** Your repo (`ERROR_TRACKING_STATUS.md`, `PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md`, `TRANSPORT_FIX_*.md`, etc.).

| # | Indicator | Meaning |
|---|-----------|--------|
| D1 | **Mediator process up**; **`/tmp/mediator.log`** activity | Guest CUDA RPC **requires** mediator. |
| D2 | **`HtoD progress`** in mediator log | Mediated **weight** upload progressing (your stack’s log string). |
| D3 | **`vm_id` / session cleanup** | Restarting mediator mid-load **invalidates** session — **operational** indicator. |
| D4 | **BAR0 / vGPU PCI / `resource0` permissions** | Guest **cannot** open device → no transport. |
| D5 | **401312 / `INVALID_IMAGE` on host** | **Same B3**, but **proven** on your path with **libcublas Lt**-sized fatbin — tracked as **E1** in-repo. |

---

## E. What was **not** done (honest scope limit)

- **“Whole world”** is not a finite set of URLs. This file uses **Ollama official docs + NVIDIA CUDA docs + primary GitHub** + explicit Phase3 overlay.  
- **Every** forum post or vendor ticket was **not** exhaustively crawled.  
- **Your** last **host/VM** results remain in **`MODEL_LOAD_INDICATORS_CHECKLIST_RESULTS.md`** — they validate **D*** + **B*** **for your deployment**, not **every** item in §A–§C without a new run.

---

## F. Relationship to the older `MODEL_LOAD_INDICATORS_CHECKLIST.md`

- **`MODEL_LOAD_INDICATORS_CHECKLIST.md`** — operational matrix aligned with **phase3** debugging (still useful).  
- **`MODEL_LOAD_INDICATORS_SOURCED_CHECKLIST.md` (this file)** — **traceability** to **Ollama + NVIDIA + public** sources + **§D** Phase3.  

Use **both**: **sourced** for **why** an indicator exists; **phase3** matrix for **order of checks** on your system.

---

*If you want a single merged file only, say so and we can combine §A–§D into one printable doc without dropping citations.*
