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

*Added Mar 22, 2026 — explicit alternative to “extend transmission time” as the primary tactic.*
