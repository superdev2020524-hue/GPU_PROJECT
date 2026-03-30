# Host (dom0): clean FATBIN artifact + fresh `mediator.log` + restart mediator

**When:** Before a new FATBIN / `401312` tracing session so old **`/tmp/fail401312.bin`** and a huge **`mediator.log`** don’t confuse you.

**Run as:** `root` on the **GPU host** (e.g. `10.25.33.10`).

**Adjust if needed:** Mediator binary path is often **`/root/phase3/mediator_phase3`** (see **`HOST_SETUP_BEGINNER_GUIDE.md`**). If your tree is different, change `cd` below.

---

## One block (copy–paste)

```bash
set -e
TS=$(date +%Y%m%d_%H%M%S)

# 1) Stop mediator
pkill -TERM mediator_phase3 2>/dev/null || true
sleep 2
pkill -KILL mediator_phase3 2>/dev/null || true
sleep 1

# 2) Remove old dumped fatbin (so the next dump is obviously new)
rm -f /tmp/fail401312.bin

# 3) Rotate mediator log (keep backup; start empty log)
if [ -f /tmp/mediator.log ]; then
  mv /tmp/mediator.log "/tmp/mediator.log.bak.${TS}"
fi
: > /tmp/mediator.log

# 4) Start mediator again (typical layout)
cd /root/phase3
nohup ./mediator_phase3 >> /tmp/mediator.log 2>&1 &

sleep 2
echo "=== pgrep ==="
pgrep -a mediator_phase3 || echo "WARNING: mediator not running"
echo "=== first lines of new log ==="
head -20 /tmp/mediator.log
```

**Note:** Step 4 uses **`>>`** so if `nohup` also creates output, behavior is consistent; if you prefer a **fully empty** file before start, use **`> /tmp/mediator.log`** only in `nohup`:

```bash
cd /root/phase3
nohup ./mediator_phase3 > /tmp/mediator.log 2>&1 &
```

(use **one** of the two start styles, not both).

---

## After this

1. **VM:** trigger load only **after** mediator shows **Initialized** in the log.
2. **Host:** `tail -f /tmp/mediator.log` during the run.
3. **Fresh FATBIN:** `ls -la /tmp/fail401312.bin` after a **`401312`** line — **mtime** should be **new**.

---

## Do not do during an active VM load

Restarting the mediator **invalidates** the guest transport session. Stop loads first, then run this block.
