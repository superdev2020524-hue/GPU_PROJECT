# Host: Commands and Logs for Live Production Test

Use this when the guest fails with "unable to allocate CUDA0 buffer" so we can see what the host mediator and cuda-executor do.

---

## 1. Where to run

**On the host** (the machine that runs the VM and the Phase 3 mediator). All host-side code is under `phase3/` (e.g. `phase3/src/mediator_phase3.c`, `phase3/src/cuda_executor.c`).

---

## 2. Commands to run on the host

### 2.1 Ensure the mediator is running and logging

If the mediator is already running (e.g. as a service), ensure its **stderr** is captured. The mediator writes all `[cuda-executor]` and `[SOCKET]` messages to **stderr** (not to a file by default).

**Option A – Mediator not yet running**

```bash
cd /path/to/gpu/phase3/src
# Build if needed: make mediator_phase3 (or your build command)
sudo ./mediator_phase3 2>/tmp/mediator.log
# Leave this running in a terminal (or run in background with nohup)
```

**Option B – Mediator already running (e.g. systemd)**

Find where its stderr goes (e.g. journal or a log file). If you can restart it once with stderr to a file:

```bash
sudo ./mediator_phase3 2>/tmp/mediator.log
```

Or capture the next test from journalctl (see 2.3).

---

### 2.2 Live test (trigger from guest)

1. On the **host**, start log capture (choose one):

   **Capture to file (if mediator stderr is redirected to /tmp/mediator.log):**
   ```bash
   # Clear or rotate so we only see this test
   echo "=== LIVE TEST $(date -Iseconds) ===" >> /tmp/mediator.log
   tail -f /tmp/mediator.log
   ```
   Leave this running.

   **Or capture from systemd/journal (if mediator runs as a service):**
   ```bash
   sudo journalctl -u mediator_phase3 -f --no-pager
   ```
   (Adjust unit name if different, e.g. `vgpu-mediator`.)

2. On the **guest** (VM), trigger one generate so the guest sends `cudaMalloc` (e.g. call_id 0x0030):

   ```bash
   curl -s -X POST http://127.0.0.1:11434/api/generate \
     -d '{"model":"llama3.2:1b","prompt":"Hi","stream":false}' --max-time 120
   ```

3. On the **host**, stop the `tail -f` or journalctl after the request finishes (or after ~30 seconds). You should see either:
   - `[cuda-executor] cuMemAlloc: allocating ... bytes on physical GPU (vm=...)`
   - then either `cuMemAlloc SUCCESS` or `cuMemAlloc FAILED: rc=...`
   - or no such lines (mediator not receiving the call, or not logging).

---

### 2.3 One-shot: recent mediator output (no live tail)

If you already ran a generate from the guest and want to see what the host saw:

```bash
# If mediator stderr goes to /tmp/mediator.log:
tail -300 /tmp/mediator.log

# If mediator runs under systemd:
sudo journalctl -u mediator_phase3 -n 300 --no-pager
```

---

## 3. What to send back

Please send:

1. **Snippet of host log from the moment of the test** (last 150–300 lines of `/tmp/mediator.log` or the equivalent journal slice). Include:
   - Any `[SOCKET]` / `[CONNECTION]` / `[PERSIST]` lines (connection from VGPU-STUB).
   - Any `[cuda-executor]` lines, especially:
     - `cuMemAlloc: allocating ... bytes on physical GPU (vm=...)`
     - `cuMemAlloc SUCCESS` or `cuMemAlloc FAILED: rc=... (vm=...)`
   - Any `[ERROR]` or `[WARNING]` lines in that window.

2. **How the mediator was started** (e.g. `sudo ./mediator_phase3 2>/tmp/mediator.log`, or systemd unit name and whether stderr is logged).

3. **VM id** if you see it in the logs (e.g. `vm=13` in the cuMemAlloc line). Optional but helpful.

---

## 4. Quick reference

| What | Command (host) |
|------|----------------|
| Mediator stderr to file | `sudo ./mediator_phase3 2>/tmp/mediator.log` |
| Follow mediator log live | `tail -f /tmp/mediator.log` |
| Recent mediator log | `tail -300 /tmp/mediator.log` |
| If using systemd | `sudo journalctl -u mediator_phase3 -n 300 --no-pager` |
| GPU memory (host) | `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` |

Trigger from guest: `curl -s -X POST http://127.0.0.1:11434/api/generate -d '{"model":"llama3.2:1b","prompt":"Hi","stream":false}' --max-time 120`
