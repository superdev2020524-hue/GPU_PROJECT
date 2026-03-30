# Host (dom0): deploy `CUDA_CALL_STREAM_DESTROY` fix

This matches the **SCP single-file** workflow used elsewhere in PHASE3 (see **`PHASE3_TEST3_DEPLOY.md`**: prefer **SCP**, not full-tree sync tools).

**Change:** `src/cuda_executor.c` — unknown stream handle → **`CUDA_ERROR_INVALID_HANDLE`** (no silent **SUCCESS**).  
**Binary:** `mediator_phase3` links **`cuda_executor.o`** (`make mediator_phase3`).

**Host:** `root@10.25.33.10` — **`MEDIATOR_HOST`** / **`MEDIATOR_USER`** in **`vm_config.py`**.

---

## Context (read once)

| Topic | Detail |
|-------|--------|
| **Phase 1 milestone** | End-to-end **GPU-mode** Ollama in the VM: discovery → vGPU path → host GPU → inference response (**`SYSTEMATIC_ERROR_TRACKING_PLAN.md`**, **`ERROR_TRACKING_STATUS.md`**). |
| **Assistant role** | **`ASSISTANT_PERMISSIONS.md`**: assistant may **read** host logs via **`connect_host.py`**; **you** copy sources, **build**, and **restart** the mediator on dom0. |
| **VM** | Current target in repo: **test-4@10.25.33.12** (`VM_USER` / `VM_HOST` in **`vm_config.py`**). GPU mode is verified with **`journalctl`** (`library=CUDA`, **`inference compute`**) — re-check after any change. |
| **Config files** | **Host:** `phase3/` tree at **`/root/phase3`** (adjust if yours differs). **VM:** Ollama overrides under **`/etc/systemd/system/ollama.service.d/`** (e.g. **`vgpu.conf`**) per existing deploy docs — not replaced by this host-only fix. |

---

## 0) From your local PC — copy **only** the modified file (SCP)

Run on the machine that has this repo (paths match a typical layout):

```bash
cd /home/david/Downloads/gpu/phase3

scp -o StrictHostKeyChecking=no \
  src/cuda_executor.c \
  root@10.25.33.10:/root/phase3/src/cuda_executor.c
```

Enter **`root`**’s password when prompted.

If you use **`sshpass`** non-interactively (same pattern as **`PHASE3_TEST3_DEPLOY.md`**):

```bash
# Password: use the same as MEDIATOR_PASSWORD / VM_PASSWORD in vm_config.py (do not commit secrets).
sshpass -p 'YOUR_ROOT_PASSWORD' scp -o StrictHostKeyChecking=no \
  src/cuda_executor.c \
  root@10.25.33.10:/root/phase3/src/cuda_executor.c
```

**If your host `phase3` path is not `/root/phase3`**, change the destination directory only; keep **`.../src/cuda_executor.c`**.

---

## 1) On the host (SSH as root) — build

```bash
ssh root@10.25.33.10

cd /root/phase3

grep -n 'CUDA_ERROR_INVALID_HANDLE' src/cuda_executor.c | head -5
grep -A15 'CUDA_CALL_STREAM_DESTROY' src/cuda_executor.c | head -25

make clean
make mediator_phase3

ls -la mediator_phase3
```

Resolve any **`make`** errors before restarting.

---

## 2) On the host — restart mediator

```bash
cd /root/phase3

pkill -f mediator_phase3 || true
sleep 2

nohup ./mediator_phase3 2>>/tmp/mediator.log </dev/null &
disown
sleep 2

pgrep -a mediator_phase3
tail -30 /tmp/mediator.log
```

---

## 3) Log bundle to paste back (for assistant review)

```bash
echo "=== date ==="
date -Iseconds

echo "=== mediator process ==="
pgrep -a mediator_phase3 || echo "NOT RUNNING"

echo "=== mediator binary mtime ==="
ls -la /root/phase3/mediator_phase3

echo "=== recent mediator.log ==="
tail -80 /tmp/mediator.log

echo "=== module-load / INVALID / 401312 (recent) ==="
grep -E 'module-load|INVALID_IMAGE|401312' /tmp/mediator.log | tail -40
```

---

## 4) After paste — next steps (plan)

1. Assistant reviews the bundle (mediator running, sane startup).  
2. **VM:** checkpoints **A–C** then long generate / **`RUNNER_EXIT2_NEXT_STEPS_MAR22.md`** (coredump / sequence logs).  
3. **Host:** §6 Step 3–4 (**`load_host_module`**, **libcublasLt** ) only if **E1** (**401312** / **INVALID_IMAGE**) reappears.

---

## Reference

- **`RESTART_MEDIATOR_ON_HOST.md`**
- **`PHASE3_TEST3_DEPLOY.md`** (SCP policy)
- **`ASSISTANT_PERMISSIONS.md`**, **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`**
