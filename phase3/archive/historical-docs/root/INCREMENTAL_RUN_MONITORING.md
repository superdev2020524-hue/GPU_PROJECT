# Incremental monitoring (mandatory for long loads)

**Problem:** A single **`curl -m 7200`** with no sampling wastes hours and produces no actionable signal until the end.

**Rules (assistant + human):**

1. **No multi-minute blind generate** unless the user **explicitly** approves the wall-clock window *and* the stop conditions below.
2. **Every ~3–5 minutes** (or every **+0.01** on **`model load progress`** if visible):
   - **VM:** `journalctl -u ollama -n 30 --no-pager | grep -E 'model load progress|ERROR|exit status'`
   - **Host:** `grep -E 'HtoD progress|module-load.*401312|INVALID_IMAGE|FAILED' /tmp/mediator.log | tail -20`
3. **Stop early** if:
   - **`model load progress`** unchanged for **≥15 min** (stuck), or  
   - **`exit status 2`** / **`ERROR`** / **`401312` + `INVALID_IMAGE`** with no forward progress, or  
   - **`context canceled`** / **499** (client dropped).
4. **After stop:** capture **`/tmp/vgpu_call_sequence.log`** tail, **`journalctl`** slice, host **`mediator.log`** tail — then decide (fix, shorter repro, coredump).

**Existing artifacts:** `/tmp/vgpu_current_call.txt`, `/tmp/vgpu_call_sequence.log` (guest); mediator stderr (host).

---

## Automated Markdown log (10-minute ticks, workstation)

For **4h long-runs** (`run_longrun_4h_capture.sh` / `reset_and_start_longrun_4h.sh`), run **`phase3_longrun_10min_monitor.sh`** on the **same machine** that can SSH to the VM and dom0. It appends **`LONGRUN_SESSION_<PHASE3_LONGRUN_TS>.md`** every **600s** (override with **`PHASE3_MONITOR_INTERVAL_SEC`**) with:

- filtered **`journalctl -u ollama`**
- filtered **`/tmp/mediator.log`**
- **`/api/ps`**
- whether the long-run **`curl`** is still alive

Set **`PHASE3_LONGRUN_TS`** to the session id (same as **`/tmp/phase3_longrun_<TS>/`** on the VM). You can skim **`LONGRUN_SESSION_*.md`** at ~10-minute intervals without tailing logs manually. After the run, keep the file for error archaeology alongside **`collect_host_longrun_slice.sh`** output.

---

*Added Mar 22, 2026 — in response to triage process failure (blind 2h run).*

---

## Do not use “longer `curl -m`” as the main fix

Extending HTTP timeout does **not** speed BAR1/HtoD. Prefer **decoupled load** (background preload, VM CLI, **`keep_alive`**), **smaller GPU upload**, or **transport fixes**. See **`PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md`** and **`vm_async_preload.sh`**.
