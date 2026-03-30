# Temporary Stage — E1 (fatbin sm_80 vs H100)

## Temporary process (operator sequencing)

Use this **order** before revisiting dump / `cuobjdump` questions:

| Step | Goal |
|------|------|
| 1 | VM cuBLAS / paths / CC path checks (Stages 1–2 in the table below). |
| 2 | `ldd` / env — runner uses intended **`cuda_v12`** (legacy **Stage 3** row below). |
| **3** | **Model load reliability** — Ollama must **load** models without hanging (`/api/ps` stuck at 0 forever, no `model load progress`, stuck `systemctl stop` in **`deactivating`**). Fix **before** treating E1 dump / **`fail401312.bin`** as authoritative. |
| 4 | Resume **Stage 1** dump + **`cuobjdump`**, then Stage 4 matrix if still **`sm_80`**. |

**2026-03-23 — forced service recovery (VM `test4`):** Ollama was **not** stopped correctly (**`ActiveState=deactivating`**, **`SubState=stop-sigterm`**). **`sudo systemctl kill -s KILL ollama`** applied; unit has **`Restart=always`** → new **`active (running)`** process (**`ollama.bin.new serve`**, **`127.0.0.1:11434`**). **Guest VM OS was not powered off** — only the stuck service was killed and auto-restarted.

---

**Why no new `fail401312.bin`?** See **`FAIL401312_DUMP_WHY_AND_HOW.md`** — dom0 **`cuda_executor.c` currently lacks** the **dump patch**; only **2** `dumped` lines **ever** in **`mediator.log`**.

**Rule:** Complete each stage; if the required signal is missing or unchanged, advance to the next.

| Stage | One-line goal |
|-------|----------------|
| **1** | After VM cuBLAS swap: prove **new** failure blob → **`cuobjdump`** shows **sm_XX** (still 80 vs now 90). |
| **2** | If still sm_80: verify **CC=9.0** path (**journal** + shim **`gpu_properties` / `fetch_gpu_info`**). |
| **3** | If still sm_80: **ldd** / env — runner uses **only** intended **`cuda_v12`** Lt/cublas. |
| **4** | If still sm_80: **NVIDIA matrix** / alternate **libcublasLt** build for driver+H100; optional **Method 5** on dom0. |

**Current:** **Temporary Stage 2 — DONE (2026-03-23)** — **E2 / CC path OK** in checks below; **Stage 1** still **no fresh** **`fail401312.bin`**. Next lever: **Stage 4** (NVIDIA matrix / alternate Lt) or **Stage 1** when a long load can complete.

**Temporary Stage 2 (results):**
- **(a) VM journal** (this boot, last lines): **`inference compute`** → **`compute=9.0`**, **`library=CUDA`**, H100 — **no `8.9` in tail.**
- **(b) Repo** **`phase3/GOAL/SOURCE/gpu_properties.h`**: **`GPU_DEFAULT_CC_MAJOR 9`**, **`MINOR 0`.**
- **(c) Repo** **`fetch_gpu_info()`** (**`libvgpu_cuda.c` ~3841–3845**): after live overlay, **`g_gpu_info.compute_cap_{major,minor} = GPU_DEFAULT_*`** (forced **9.0**).
- **(d) VM** **`/usr/lib64/libvgpu-cuda.so`** (Mar 20): present; **`strings`** includes **`GPU info (live): … CC=%d.%d`** / H100 defaults — consistent with built shim.

**Method 3 (2026-03-23, VM):** **`readlink -f`** **`libcublasLt.so.12`** → **`…/libcublasLt.so.12.8.5.7`**; **`libcublas.so.12`** → **`…12.8.5.7`**. Runner **`LD_LIBRARY_PATH`**: **`ldd libggml-cuda.so`** → **`libcublas` / `libcublasLt`** both **`=> /usr/local/lib/ollama/cuda_v12/…`**.

**2026-03-23 detached run (assistant):** Killed stuck **`curl`**, **`ollama`=`active`**, **`/api/tags`** OK; **new** **`nohup curl`** → **`NEW_PID=452067`**, log **`/tmp/ollama_gen_detach.out`**. Poll host **`stat /tmp/fail401312.bin`** then **`cuobjdump`**.

**2026-03-22 detached run (assistant):** **`nohup curl … -m 14400`** → **`DETACH_PID=451428`** (superseded).

**2026-03-22 runs (assistant):** (1) **14 min** **`curl`** → **~15 %** load, **context canceled**. (2) **60 min** **`curl`** → still **`curl (28)`** timeout, **0 bytes**; journal shows **HtoD** polls **after** first cancel (**seq** up through **~534 @ 20:19**) but **no** new **`fail401312.bin`** (**mtime** still **Mar 20**); **`cuobjdump`** on that file still **`sm_80`**. Conclusion: **Stage 1** blocked until a load **finishes past** weight upload (needs **>60 min** client **or** non-blocking strategy per **`PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md`**).

**Request (you):** One VM load that reproduces **401312**, then on dom0: `stat /tmp/fail401312.bin` (confirm new date) and `cuobjdump -elf /tmp/fail401312.bin | grep 'arch ='`.
