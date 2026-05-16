# Strategy: stop relying on “long HTTP / transmission time”

**Problem:** A slow **vGPU BAR1 → host GPU** path makes the **first** model load wall‑clock long. Extending **`curl -m`** does **not** move bytes faster; it only avoids **client‑side** cancel. It is **not** an effective *method* — it’s waiting with a longer leash.

**Goal:** Use approaches that either **remove the HTTP client from the hot path**, **pay the cost once** and reuse, **reduce bytes moved**, or **fix transport** — not “longer timeout” as the primary tool.

---

## 1) Decouple load from any single HTTP client (primary operational fix)

**Idea:** The load runs **on the VM** against **127.0.0.1**. Nothing requires your laptop or a single **`curl`** process to stay open for the whole BAR1 transfer.

| Method | What to do |
|--------|------------|
| **A. Ollama‑documented preload (empty request)** | `curl -sS -X POST http://127.0.0.1:11434/api/generate -d '{"model":"YOURMODEL","keep_alive":-1}'` — **omit `prompt`** (see **`ollama-src/docs/faq.mdx`** “preload … empty request”). Same server work, but you typically pair with **B** so no one **`curl`** must block. |
| **B. Background + poll (no `-m` on the critical path)** | Start preload in **`nohup`** (or **systemd oneshot**), write stdout/stderr to **`/tmp/`**, then **poll** completion: **`journalctl`** (`model load progress`, errors), **`GET /api/ps`**, or **`/tmp/ollama_preload.*`** — see **`vm_async_preload.sh`**. |
| **C. VM CLI (no HTTP at all)** | From an SSH session on the VM: `ollama run YOURMODEL ""` — blocks in the **terminal**, not in **`curl`**. Use **`tmux`/`screen`** if the session must stay attached. |

**Why this is different:** You are **not** “extending transmission time” as the *strategy*; you are **not tying success to one client’s timeout**. Progress is observed **incrementally** (already aligned with **`INCREMENTAL_RUN_MONITORING.md`**).

---

## 2) Pay once: `keep_alive` / `OLLAMA_KEEP_ALIVE`

After the **first** successful load, keep the model resident so **later** tests don’t repeat full HtoD.

- Per request: `"keep_alive": -1` on preload/generate (**faq.mdx**).
- Global: **`OLLAMA_KEEP_ALIVE=-1`** in **`ollama.service`** / drop‑in (same semantics as upstream docs).

**Caveat:** This does **not** shorten the **first** load; it removes repeat loads from the test loop.

---

## 3) Reduce work over BAR1 (real “shorter path”, not longer wait)

Tune so **less** data / fewer GPU layers move during bring‑up (faster **time‑to‑ready** on a slow link):

- **Smaller model** (e.g. already using **tiny** variants for triage).
- **More layers on CPU** (if your build exposes **`num_gpu` / GPU layer controls** in the API or Modelfile) — **smaller** VRAM upload for first validation.
- **Single user / no parallel models** — avoid queue contention.

Use this for **Phase 1 “does the stack live?”** before maxing GPU layers.

---

## 4) Fix throughput or correctness (host / mediator — not longer `curl`)

If BAR1 is intrinsically slow or buggy, the effective fix is **implementation**, not timeout:

- Larger chunks, pinned paths, **mediator** tuning, host driver/module issues (**`401312` / INVALID_IMAGE** — see **`FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`**, **`ERROR_TRACKING_STATUS.md`**).

**Assistant role:** document and grep logs; **dom0** changes stay **human** per **`ASSISTANT_PERMISSIONS.md`**.

---

## 5) What *not* to treat as “the solution”

| Approach | Why it’s weak as the main fix |
|----------|-------------------------------|
| Only **`curl -m 7200`** | Does not speed HtoD; only avoids client cancel. |
| Only **preload FAQ** **without** B/C | Same blocking duration if one client still waits with a short **`-m`**. |

---

## 6) Script

**`vm_async_preload.sh`** — background preload + optional **`/api/ps`** poll (run **on the VM**).

---

## 7) VM‑6 / mediated BAR1 — **CPU layer priming** before **GPU** `generate`

**Symptom (2026‑05):** After a **decoupled FAQ preload** (**§1**) or **`systemctl restart ollama`**, **`POST /api/generate`** with **default** GPU layer assignment (**`num_gpu: -1`**) sometimes returns **`HTTP 200`** with **`done: true`** but a **`response`** full of **non‑printable** bytes — **`context`** token ids may still look sane (**`ERROR_TRACKING_STATUS.md`**, **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`** §6).

**Additional symptom (2026‑05‑16, resident path):** Archival **Test‑4** (**omit `num_gpu`**, **`curl -m` ~185**) run **by itself** (no **§7** step in that shell / after idle unload) may return **`HTTP 000`**, **0 bytes**, **`curl` exit 28** at the client ceiling — even when **`GET /api/tags`** still lists **`tinyllama:latest`**. **Fix:** always run **§7** ( **`num_gpu: 0`** prime) immediately before strict **Test‑4** in the same session, or use the full **§8** chain after **`systemctl restart`** (**`ERROR_TRACKING_STATUS.md`** rolling row).

**Workaround (reproducible on VM‑6, no service restart between steps):**

1. One bounded **`POST /api/generate`** with **`"options": {"num_gpu": 0, …}`** — all layers on **CPU** (**`ollama-src/server/sched.go`** when **`NumGPU == 0`** → empty GPU list; **`TRANSMISSION_OUTCOMES_AND_PROGRESS_ASSESSMENT.md`** **`num_gpu`** note).
2. Immediately after **`200`** + coherent text, repeat **`/api/generate`** with **`num_gpu: -1`** (normal GPU path).

**Language / sampling (TinyLlama + VM‑6):** Use a **plain** English instruction (e.g. “In one short English sentence…”). **Avoid** negative lists of forbidden languages (“do not use Italian / Spanish…”) — that pattern can produce **meta** answers **in another language** instead of the requested English. For the **CPU** step, **`"top_p": 0.85`** alongside **`temperature` ~0.3** helps keep **English** stable; guest reference **`/tmp/ab_cpu_then_gpu.py`** on **VM‑6** encodes this pair (**rolling log** **2026‑05‑15** in **`ERROR_TRACKING_STATUS.md`**).

**Observability:** Piping **`curl`** through **`head -c`** on the **same** stdout stream can **drop** the **`curl -w`** trailer (**`HTTPCPU`/`HTTPT4`** metrics missing in logs). Prefer **`-o /tmp/<tag>.json`** plus **`-w`** on stdout, then **`head -c`** the file (**`run_resident_mar29_test4.sh`**, **2026‑05‑16**).

**Effect:** **GPU** pass then often returns **readable English** at **`temperature`** **0.3** in the same session (**rolling log** **2026‑05‑15** in **`ERROR_TRACKING_STATUS.md`**). Treat as **operational stabilization**, not a substitute for **§4** transport / **`BUILD_AND_DEPLOY_LIBGGML_CUDA_PHASE3.md`** fixes for **cold** **`load_duration`** parity with archived **Mar 29** **Test‑4**.

---

## 8) March 29 **Test‑4** shape on **VM‑6** after **cold** `restart` (bounded **`curl -m 185`**)

**Context:** Archival **Test‑4** (`tinyllama`, `prompt: Hello`, `num_predict: 16`, **omit** `num_gpu`, **`curl -m` ~180) returns **`HTTP 000`** if run **alone** right after load or **`systemctl restart`** on **BAR1** — see **`ERROR_TRACKING_STATUS.md`**. **§1** plus **§7** ordering **before** the default‑GPU request **closes** that gap.

**Runbook (on the VM, phase3 docs only):**

1. **`sudo systemctl restart ollama`** (or cold boot) and wait until **`GET /api/tags`** responds.
2. **§1 decoupled preload:** run **`vm_async_preload.sh`** (or equivalent `nohup curl -m 0` empty‑body FAQ preload per **§1**).
3. **Poll `GET /api/ps`** until **`tinyllama`** appears (often ~15–18×60 s wall on **VM‑6**).
4. **§7 CPU prime** using the **same `prompt` and `num_predict`** you will use for **Test‑4** (e.g. **`Hello`**, **`16`**): one **`POST /api/generate`** with **`"num_gpu": 0`** (bounded client, e.g. `curl -m 120`).
5. **Archival **Test‑4****: **`POST /api/generate`** with **default** GPU (**omit `num_gpu`**), same JSON as step 4 otherwise, **`curl -m 185`** — **expect **`HTTP 200`**, **`done: true`**, human‑readable **`response`**.

**One-shot on the VM:** `bash run_mar29_section8_chain.sh` (**`phase3/`** in the repo) runs steps **1–5** in order. **Resident only** (no restart / preload): `bash run_resident_mar29_test4.sh` runs **§7** then strict **Test‑4** — avoids **`HTTP 000`** from running **Test‑4** alone (**`ERROR_TRACKING_STATUS.md` **2026‑05‑16**). Scripts must use **Unix LF** line endings (**CRLF** from some editors breaks `systemctl` / `bash` after `base64` transfer). **`ERROR_TRACKING_STATUS.md`** (**2026‑05‑16**) records a **`connect_vm.py`** + **`CONNECT_VM_COMMAND_TIMEOUT_SEC=2700`** wall‑clock sample.

**`load_duration` note:** The **~8 s**‑class first load in this chain is on the **CPU** (`num_gpu: 0`) leg; the **Test‑4** leg is **resident** and typically reports **much shorter** `load_duration` (nanoseconds per **`ollama-src/docs/api.md`**). Parity with archival **`vm=9` single‑request ~7.46 s default‑GPU cold** remains a **§4** transport / **shmem** topic (**`TRANSPORT_SHMEM_CONTIGUITY.md`**).

---

*Added Mar 22, 2026 — explicit alternative to “extend transmission time” as the primary tactic.*

*Section **7** added 2026‑05‑15 — CPU **`num_gpu`** priming before GPU **inference** (VM‑6 **BAR1** **text‑sanity** **class**). **§7** **language/`top_p` **note** **2026‑05‑15**.*

*Section **8** added 2026‑05‑15 — cold **restart** → **§1** → **§7** → archival **Test‑4** bounded client (**`ERROR_TRACKING_STATUS.md`** cold chain row).*
