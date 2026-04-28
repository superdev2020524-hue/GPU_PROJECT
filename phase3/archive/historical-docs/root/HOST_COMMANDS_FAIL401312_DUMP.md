# Host commands — enable `/tmp/fail401312.bin` dumps (mediator)

The repo **`phase3/src/cuda_executor.c`** now includes the **`fopen("/tmp/fail401312.bin")`** block (after **`memcpy`**, before **`cuModuleLoadFatBinary`**). Copy it to **dom0**, rebuild, restart the mediator.

---

## 1) Copy updated `cuda_executor.c` to the host

**From your workstation** (adjust paths if needed):

```bash
scp -o StrictHostKeyChecking=no \
  /home/david/Downloads/gpu/phase3/src/cuda_executor.c \
  root@10.25.33.10:/root/phase3/src/cuda_executor.c
```

---

## 2) On dom0 (SSH as `root`)

```bash
cd /root/phase3

# Backup
cp -a src/cuda_executor.c "src/cuda_executor.c.bak.$(date +%Y%m%d_%H%M%S)"

# Confirm patch is present
grep -n 'fail401312' src/cuda_executor.c

# Rebuild
make -j"$(nproc)"

# Restart mediator (log continues to append)
pkill -f mediator_phase3 2>/dev/null || true
sleep 2
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
nohup ./mediator_phase3 2>>/tmp/mediator.log </dev/null &
disown
sleep 1
pgrep -af mediator_phase3
```

---

## 3) After the next VM load that hits **401312**

```bash
stat /tmp/fail401312.bin
# expect new mtime

grep 'dumped /tmp/fail401312' /tmp/mediator.log | tail -3

cuobjdump -elf /tmp/fail401312.bin \| grep -E 'arch ='
```

---

## Permissions

Host **edit / build / restart** = **you** (see **`ASSISTANT_PERMISSIONS.md`**).
