# Model load indicators — checklist results (host + VM)

**Run timestamp (UTC):** `2026-03-23T06:59:14Z` (host start) · VM batch same session (~2026-03-23 06:58–07:00Z)  
**Follow-up O1–O3 (UTC):** `2026-03-23T07:00Z` approx.  
**Method:** `connect_vm.py` → `test-4@10.25.33.12` · `connect_host.py` → `root@10.25.33.10`  
**Idle probe (first pass):** **No** concurrent load. **Live probe:** **§5** — background **`/api/generate`** on VM + read-only **`mediator.log`** samples on host (**2026-03-23**).

---

## 1. Status summary (checklist IDs)

| ID | Status | Notes |
|----|--------|------|
| **O1** | **PASS** | `systemctl is-active ollama` → **`active`** |
| **O2** | **PASS** | `127.0.0.1:11434` **LISTEN** |
| **O3** | **PASS** | `GET /api/tags` → **200**; `GET /api/ps` → **200** |
| **O4** | **PASS** | **`OLLAMA_LOAD_TIMEOUT=60m`** in `vgpu.conf` (via journal grep) |
| **O5** | **PASS** | **`OLLAMA_DEBUG=1`**, **`CUDA_TRANSPORT_TIMEOUT_SEC=3600`** |
| **O6** | **PASS** | Last **`inference compute`**: **`library=CUDA`**, **`compute=9.0`**, H100 |
| **N1** | **WARN** | **`ldd`** on `libggml-cuda.so` **without** runner `LD_LIBRARY_PATH` shows **`libggml-base.so.0 => not found`**, **`libcublas.so.12 => not found`** — **expected** when not using Ollama’s library path; **runtime** uses **`OLLAMA_LIBRARY_PATH`** (see journal). |
| **N2** | **PASS** | **`libcublas.so.12` → `…12.8.5.7`**, **`libcublasLt.so.12` → `…12.8.5.7`** |
| **N3** | **PASS** | **No** `libcublas` under **`/opt/vgpu/lib`** |
| **N4** | **PASS** | **`/usr/lib64/libvgpu-cuda.so`** present |
| **V1** | **PASS** | **`10de:2331`** HEXACORE vH100 |
| **V2** | **PASS** | **`resource0` mode `666`** |
| **V3** | **PASS** (historical) | **Boot journal** contains **`[cuda-transport]`** polls **`0x0030`/`0x0032`/`0x0042`** from **`BAR1`** — transport **was** used; no separate **`ensure_connected`** string in grep (may log under different tag). |
| **S1** | **PASS** | **`sched.go`** contains **Phase3/vGPU** `NumGPU = -1` when GPUs exist |
| **S2** | **WARN** | **`unable to refresh free memory`**: **58** occurrences **this boot** |
| **M1** | **PASS** | **`mediator_phase3` PID 2487438** |
| **M2** | **PASS** (live 2026-03-23) | **`wc -l /tmp/mediator.log`:** **4200 → 4223 → 4230 → 4245** during background **`/api/generate`** — log **grew** (see §5). |
| **M3** | **PASS** (live 2026-03-23) | **`vm_id=9`** on **`[MEDIATOR]`** and **`[cuda-executor]`** lines during same session (see §5). |
| **M4** | **PASS** (historical) | **`HtoD progress`** lines **vm=9** to **~1548 MB** |
| **M5** | **PASS** (historical) | **Guest journal:** **`from=BAR1`**, **`call_id=0x0032`**; host **`HtoD progress`** |
| **H1** | **PASS** | Multiple **`module-load … rc=0`** for smaller payloads |
| **H2** | **FAIL** | **`401312` → `CUDA_ERROR_INVALID_IMAGE`** still present; **live run** reproduced **module-load** path (see §5). |
| **H3** | **PASS** | **`nvidia-smi`**: **H100 PCIe**, driver **545.23.06** |
| **R1** | **PASS** (live start) | **Mar 23** load: **`model load progress 0.00`** after **`load_tensors`** (see §5); full 0→1 not waited in this probe. |
| **R2** | **PASS** (historical) | **`load_tensors`** / **CUDA0** / layers **Mar 22** |
| **R3** | **FAIL** (historical) | **`timed out waiting for llama runner … context canceled`** **Mar 22** |
| **R4** | **FAIL** (historical) | **`llama runner terminated` `exit status 2`** **Mar 22 20:44** |
| **R5** | **FAIL** (historical) | **`error loading llama server`** tied to **R3** |
| **C1** | **WARN** | **`/`** **89%** used (**33G/39G**, **~4.2G free**) — risk for large models / caches |
| **C2** | **PASS** | **~3.1 GB available** RAM (incl. cache) |
| **C3** | **PASS** | **No** OOM lines in **`dmesg`** tail (empty or none) |
| **C4** | **N/A** | Not assessed |

**§8 HTTP / operator:** **Partial** — background **`curl`** **`/api/generate`** (**`num_predict=2`**, **600s** max) started from VM; **no** laptop client involved.

**Host permissions:** Per **`ASSISTANT_PERMISSIONS.md`**, assistant used **read-only** host access (**`wc`/`grep`/`tail`** on **`/tmp/mediator.log`** only). **No** dom0 edit/build/restart.

---

## 2. Gaps — still open

| Item | Why |
|------|-----|
| **N1** | Re-run **`ldd`** with **same `LD_LIBRARY_PATH`** as runner (optional static check). |
| **V3** | **`ensure_connected`** log tag still not isolated; **cuda-transport** + **HtoD** confirm path. |
| **R1** | **Fraction** only seen from **0.00** onward in short window; **full** load to completion not awaited. |
| **§8** | **Decoupled** / **preload-only** / **very long** client tests not run. |
| **C4** | Not assessed. |
| **H2 / E1** | **Fix** requires **VM** fatbin/**Hopper `libggml-cuda.so`** + policy per **`CURRENT_STATE_AND_DIRECTION.md`** — **not** host log reading alone. |

---

## 3. Raw excerpts (abbreviated)

### VM — O1–O3 (follow-up)

```
active
LISTEN ... 127.0.0.1:11434
tags=200
ps=200
```

### VM — disk / memory

```
/dev/xvda2  39G  33G  4.2G  89%  /
Mem: 3.8Gi total, ~3.1Gi available
```

### VM — `pgrep`

`/usr/local/bin/ollama.bin.new serve` **PID 453459**

### Host — mediator / HtoD / E1 counts

- **`mediator.log`:** **4200** lines  
- **`grep -c 401312`:** **4**  
- **`grep -c INVALID_IMAGE`:** **2** (substring count in file)  
- **HtoD tail:** **1435–1548 MB** **vm=9**  
- **`nvidia-smi`:** **H100**, **915 MiB** used on GPU

---

## 4. Interpretation (this run)

- **Guest** is **healthy for API + GPU discovery + transport activity** (historical).  
- **Host** mediator **up**; **HtoD** and **small module loads** succeed in log.  
- **Outstanding blockers** in **documented history:** **E1** (**401312** / **INVALID_IMAGE**), **client timeout** (**R3**), **runner exit 2** (**R4**).  
- **Disk** **89%** — **review** model cache / logs before long runs.

---

## 5. Live load session (2026-03-23) — assistant-run, read-only host

**VM:** `nohup curl -m 600` **`POST /api/generate`** `tinyllama:latest`, **`num_predict=2`** → PID **454007**, output **`/tmp/_cl_gen.out`**.  
**Host:** samples **`wc -l /tmp/mediator.log`** and **`grep`** on **`[cuda-executor]`** lines.

| Time | `mediator.log` lines | Notes |
|------|----------------------|--------|
| **BASELINE** | **4200** | Before load |
| **~50s** | **4223** | **`[MEDIATOR]`** **`vm_id=9`**, **`call_id=0x30`**, **`0x32`** |
| **~2m** | **4230** | **`cuMemAlloc`**, **`module-load`** **4432 B** **`rc=0`**; then **`Cleaned up VM 9`** / **`Initialized`** / **`CUDA_CALL_INIT`** / **`HtoD progress: 10 MB`** (mediator reset mid-sequence — correlate with policy) |
| **~4m** | **4245** | **`HtoD progress`** **10→20 MB** (new segment) and historical **1496–1548 MB** lines in tail |

**E1 (H2):** `grep` still shows **`data_len=401312`** → **`CUDA_ERROR_INVALID_IMAGE`** (**same failure mode** as before).

**R1/R2:** VM journal at load start: **`load_tensors`** **23/23** layers, **`model load progress 0.00`**.

---

*File updated with live session. Re-run after any config change.*
