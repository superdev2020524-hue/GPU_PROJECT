# Model load — indicators checklist (Ollama + NVIDIA + Phase3 vGPU)

**Sourced checklist (Ollama docs + NVIDIA/CUDA + GitHub + §D Phase3):**  
→ **`MODEL_LOAD_INDICATORS_SOURCED_CHECKLIST.md`** — use that file when you need **citations** to vendor/public material. This file remains the **operational** matrix for phase3.

**Purpose:** Every row is a **signal** that something in the load pipeline is healthy or broken. Model loading is **not** one metric; use this list to **localize** failures (discovery → scheduler → HtoD → module load → runner).

**How to use:** Run **VM** checks on the guest; **HOST** checks on the GPU mediator (dom0). **During an active load**, repeat **§5–§7** or tail logs. **Session** column: fill from `MODEL_LOAD_INDICATORS_CHECKLIST_RESULTS.md` (or paste below).

---

## Legend

| Status | Meaning |
|--------|--------|
| **PASS** | Observed value matches expected |
| **FAIL** | Observed value indicates misconfiguration or blocker |
| **WARN** | Ambiguous or needs correlation (e.g. stale log line) |
| **N/A** | Not applicable without a running load |

---

## 1. Ollama service & API (guest)

| ID | Indicator | Expected | Notes |
|----|-----------|----------|--------|
| **O1** | `systemctl is-active ollama` | `active` | If inactive, no load |
| **O2** | Listener `127.0.0.1:11434` (or configured bind) | `ss`/`lsof` shows LISTEN | No API → no load |
| **O3** | `GET /api/tags` | HTTP 200, JSON with models | Proves HTTP + server up |
| **O4** | `OLLAMA_LOAD_TIMEOUT` in runner env (journal) | Matches policy (e.g. `60m`) | Internal cancel if load exceeds |
| **O5** | `OLLAMA_DEBUG` / `CUDA_TRANSPORT_TIMEOUT_SEC` | Set per `vgpu.conf` | Affects logging and transport waits |
| **O6** | Last `inference compute` line | `library=CUDA`, plausible `compute=` (e.g. 9.0 on H100) | If `library=cpu` → no GPU load path |

---

## 2. NVIDIA / discovery / GGML (guest)

| ID | Indicator | Expected | Notes |
|----|-----------|----------|--------|
| **N1** | `libggml-cuda.so` resolvable under `OLLAMA_LIBRARY_PATH` | `ldd` / symlinks under `/usr/local/lib/ollama/cuda_v12` | No backend → CPU or fail |
| **N2** | `libcublas.so.12` / `libcublasLt.so.12` in `cuda_v12` | Symlinks to intended **`.so.12.x.y`** | Wrong Lt → E1 / wrong arch |
| **N3** | No **fake** `libcublas` in `/opt/vgpu/lib` masking discovery | Only real shim files you intend | Past issue: dummy CUBLAS → `device_count=0` |
| **N4** | `libvgpu-cuda.so` present (`/usr/lib64` or `LD_LIBRARY_PATH`) | Loader picks shim before system `libcuda` | Wrong order → no mediated CUDA |

---

## 3. vGPU device & BAR0 (guest)

| ID | Indicator | Expected | Notes |
|----|-----------|----------|--------|
| **V1** | vGPU / stub PCI visible | `lspci` shows NVIDIA / vgpu-cuda BDF | No device → transport never starts |
| **V2** | `resource0` openable by service user | Permissions (e.g. not root-only) | Past fix: chmod/udev |
| **V3** | `[cuda-transport]` / `ensure_connected` in logs during compute | Appears when first real CUDA RPC runs | Never appears → path not reached |

---

## 4. Scheduler / Go layer (guest, patched Ollama)

| ID | Indicator | Expected | Notes |
|----|-----------|----------|--------|
| **S1** | `NumGPU` / GPU list for load | Non-empty GPU list when GPUs exist | `NumGPU==0` bug → CPU load, no HtoD |
| **S2** | Refresh warning | Ideally absent; if present, correlate | “unable to refresh free memory” → may use stale VRAM |

---

## 5. Mediator & transport (host + correlated guest)

| ID | Indicator | Expected | Notes |
|----|-----------|----------|--------|
| **M1** | Mediator process running | `pgrep mediator` | Down → guest RPC blocks |
| **M2** | `mediator.log` growing on load | New lines during load | Stale log only → no activity |
| **M3** | `vm_id` consistent in CUDA results | Same VM id for session | Cleanup/restart mid-load → broken session |
| **M4** | `cuMemcpyHtoD` / `0x0032` / “HtoD progress” | Progress during weight upload | Absent → scheduler/CPU path or blocked before HtoD |
| **M5** | BAR1 / shmem messages | e.g. `mmap shmem` / poll `0x0032` | Shows which path is used |

---

## 6. Host CUDA execution (host)

| ID | Indicator | Expected | Notes |
|----|-----------|----------|--------|
| **H1** | `cuModuleLoadFatBinary` smaller payloads | Often `rc=0` in log | Baseline “host accepts fatbins” |
| **H2** | `data_len=401312` (or known E1 size) | **Either** absent **or** `rc=0` after fix | `INVALID_IMAGE` → E1 (fatbin/arch/Lt), **not** chunk reassembly bug per prior analysis |
| **H3** | GPU visible to host driver | `nvidia-smi` lists H100 (or expected GPU) | Driver/domain issue |

---

## 7. Runner lifecycle (guest journal)

| ID | Indicator | Expected | Notes |
|----|-----------|----------|--------|
| **R1** | `model load progress` | Increments during load | Flat ≥15 min → stuck |
| **R2** | `load_tensors` / backend messages | Appears for GPU load | Absent → wrong path |
| **R3** | `timed out waiting for llama runner` / `context canceled` | Absent if client decoupled | HTTP/client dropped before load done |
| **R4** | `llama runner` **exit status 2** | Absent for success | E3: crash after load / graph |
| **R5** | `error loading llama server` | Absent on success | Aggregate failure |

---

## 8. HTTP / operator (client)

| ID | Indicator | Expected | Notes |
|----|-----------|----------|--------|
| **C1** | Client timeout vs load duration | Client ≥ load + margin **or** background preload | `curl -m` too short → false failure |
| **C2** | Mediator **not** restarted during load | No `Cleaned up VM` mid-request | Invalidates session |

---

## Interpretation matrix (quick)

| If … | Suspect first … |
|------|------------------|
| **O6** CPU / no CUDA in journal | N1–N4, S1, V1 |
| **M1** down / **M4** no HtoD | Mediator, transport, S1 |
| **M4** HtoD slow but **R3** errors | C1, O4 timeout |
| **H2** `401312` + `INVALID_IMAGE` | E1 / cuBLAS Lt / arch (host + VM Lt) |
| **R4** after HtoD OK | E3 / GGML graph / guest crash |

---

## References (phase3)

- `ERROR_TRACKING_STATUS.md`, `SYSTEMATIC_ERROR_TRACKING_PLAN.md`
- `PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md`, `INCREMENTAL_RUN_MONITORING.md`
- `STAGE1_SCHED_NUMGPU_FIX.md`, `REFRESH_AND_GPU_DETECTION_INVESTIGATION.md`
- `PHASE3_INFERENCE_ISSUES_AND_NEXT_STEPS.md`, `WORK_NOTE_MAR19_LOADMODEL_BLOCKER.md`
- `FAIL401312_DUMP_WHY_AND_HOW.md`, `TEMPORARY_STAGE_E1.md`

---

## Live investigation session

Results are written by the assistant to **`MODEL_LOAD_INDICATORS_CHECKLIST_RESULTS.md`** (timestamped). Re-run after **config changes** or **before/after** a load attempt.
