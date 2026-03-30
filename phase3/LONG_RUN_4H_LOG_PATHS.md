# Long-run (4h) verification — log files (host + VM)

**Context:** `FATBIN_TRACE_RECORD.md` documents **`model load progress 0.92`** after extended wall time (~150+ min in that trace). **`INVESTIGATION_CUBLASCREATE_V2.md`** notes runner issues near **~0.92** load. **`PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md`** — match **server** `OLLAMA_LOAD_TIMEOUT` and **client** `curl -m` to the same order of magnitude.

## Server / transport (VM)

- **`/etc/systemd/system/ollama.service.d/vgpu.conf`** — repo: **`vm_ollama_vgpu.conf`** (`OLLAMA_LOAD_TIMEOUT=4h`, `CUDA_TRANSPORT_TIMEOUT_SEC=14400`).
- **`journalctl -u ollama`** — authoritative Ollama + wrapper logs.

## Client (VM, same host as Ollama)

- **`/tmp/longrun_generate_4h.json`** — `curl` body from `POST /api/generate` (if HTTP completes).
- **`/tmp/longrun_generate_4h.err`** — `curl` stderr.

## Optional VM capture (if started)

- **`/tmp/ollama_journal_longrun.log`** — `journalctl -u ollama -f` stream (background).

## Host (dom0, mediator)

- **`/tmp/mediator.log`** — primary mediator + cuda-executor trace (rotated before clean restart).
- **`/tmp/mediator.log.bak.<timestamp>`** — backup from clean restart (`HOST_MEDIATOR_CLEAN_RESTART.md`).
- **`/tmp/fail401312.bin`** — only if a **`401312`** fatbin dump occurs (`cuda_executor`).

## Commands (reference)

```bash
# VM — tail Ollama
sudo journalctl -u ollama -f

# Host — tail mediator
tail -f /tmp/mediator.log

# VM — long client (4h), tinyllama
curl -sS -m 14400 -w '\nHTTP_CODE:%{http_code}\n' -X POST http://127.0.0.1:11434/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"model":"tinyllama:latest","prompt":"Say hello in one sentence.","stream":false,"options":{"num_predict":32}}' \
  -o /tmp/longrun_generate_4h.json 2>>/tmp/longrun_generate_4h.err
```

*Added: 2026-03-25 — 4h long-run alignment with repo `vm_ollama_vgpu.conf`.*

## Reset + full capture (2026-03-26)

From workstation `phase3/`:

1. **`./reset_and_start_longrun_4h.sh`** — backs up **`/tmp/mediator.log`** on dom0 to **`/tmp/mediator.log.bak.<TS>`**, truncates log, restarts **`mediator_phase3`**, restarts VM **`ollama`**, uploads and runs **`run_longrun_4h_capture.sh`** on the VM (4h **`curl -m 14400`**, **`journalctl -u ollama -f`** to **`journal_ollama_follow.log`**, curl stdout/stderr/headers/body under **`/tmp/phase3_longrun_<TS>/`**).
2. After the run (or on failure): **`./collect_host_longrun_slice.sh [outfile.txt]`** — saves a grep slice + **`fail401312.bin`** listing from dom0 into the repo tree.

Session example: **`TS=20260326_175917`**, VM dir **`/tmp/phase3_longrun_20260326_175917/`**, baseline host slice **`phase3_host_mediator_slice_baseline_20260326_175917.txt`** (workspace).

3. **10-minute Markdown monitor (workstation):** **`./phase3_longrun_10min_monitor.sh`** with **`export PHASE3_LONGRUN_TS=<TS>`** — writes **`LONGRUN_SESSION_<TS>.md`** (see **`INCREMENTAL_RUN_MONITORING.md`** § Automated Markdown log).
