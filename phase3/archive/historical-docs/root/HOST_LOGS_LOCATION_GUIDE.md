# Host Logs Location Guide

## QEMU Logs Location

QEMU logs are typically **not in journalctl** for XCP-ng. They may be:
1. Redirected to stderr (captured by systemd or Xen)
2. Written to a file specified in QEMU startup
3. In Xen's log directory

### Finding QEMU Logs

**Option 1: Check Xen log directory**
```bash
# Check Xen's log location
ls -la /var/log/xen/
ls -la /var/log/xen/qemu-dm-176.log
```

**Option 2: Check if QEMU stderr is captured**
```bash
# Check systemd journal for QEMU
journalctl -u xen* --since '10 minutes ago' | grep -E 'vgpu|CUDA|DOORBELL'

# Or check all systemd units
journalctl --since '10 minutes ago' | grep -E '\[vgpu\].*CUDA|\[vgpu\].*DOORBELL'
```

**Option 3: Check QEMU process stderr**
```bash
# Find QEMU process
ps aux | grep qemu-dm-176

# Check if stderr is redirected (look at process file descriptors)
ls -la /proc/1775599/fd/ | grep -E 'log|err|out'
```

**Option 4: Check if logging goes to syslog**
```bash
# Check syslog
tail -100 /var/log/messages | grep -E 'vgpu|qemu'
tail -100 /var/log/syslog | grep -E 'vgpu|qemu'
```

**Option 5: QEMU may not have logging enabled**
- If QEMU was rebuilt with `vgpu-stub-enhanced.c` that has logging, it should appear
- If QEMU wasn't rebuilt, or logging code isn't active, you won't see logs
- The logging code uses `fprintf(stderr, ...)` so it should appear somewhere

## Mediator Logs Location

**Primary location:**
```bash
tail -200 /tmp/mediator.log
```

**Alternative locations:**
```bash
tail -200 /var/log/mediator_phase3.log
journalctl -u mediator.service --since '10 minutes ago'
```

## Quick Verification Commands

### 1. Check Mediator is Receiving CUDA Calls
```bash
# On host (root@10.25.33.10)
tail -200 /tmp/mediator.log | grep -E 'CUDA_CALL|vm_id=111|vm=111|Total processed|SOCKET|CONNECTION|PERSIST'
```

**What to look for:**
- `[SOCKET] New connection` - VGPU-STUB connected
- `[CONNECTION] New connection` - Connection established
- `[PERSIST] fd=XX registered` - Persistent connection
- `[cuda-executor] CUDA_CALL_MEM_ALLOC` - Memory allocation received
- `Total processed: X` - Should increase

### 2. Check for Recent Activity
```bash
# Check last 5 minutes
tail -100 /tmp/mediator.log | tail -50

# Or monitor real-time
tail -f /tmp/mediator.log | grep -E 'CUDA|vm_id=111|processed'
```

### 3. Check Socket Connection
```bash
# Verify socket exists
ls -la /var/xen/qemu/root-176/tmp/vgpu-mediator.sock

# Check if mediator is listening
ss -lx | grep vgpu-mediator
```

### 4. Check Physical GPU Activity
```bash
# Monitor GPU
watch -n 1 nvidia-smi

# Or check once
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

## If QEMU Logs Not Found

**This is OK!** The important thing is:
1. ✅ **Mediator receives calls** (check `/tmp/mediator.log`)
2. ✅ **CUDA executor processes them** (check executor logs in mediator.log)
3. ✅ **Physical GPU shows activity** (check `nvidia-smi`)

QEMU/VGPU-STUB logs are helpful for debugging but not required if mediator is receiving calls.

## Expected Mediator Log Pattern

When VM sends CUDA call, you should see in mediator.log:

```
[SOCKET] New connection on /var/xen/qemu/root-176/tmp/vgpu-mediator.sock
[CONNECTION] New connection from (fd=XX)
[PERSIST] fd=XX registered for persistent polling
[cuda-executor] CUDA_CALL_MEM_ALLOC vm=111 — allocating 545947648 bytes
[cuda-executor] cuMemAlloc SUCCESS: allocated 0x... on physical GPU
Total processed: X
```

## Troubleshooting

### No logs in /tmp/mediator.log
```bash
# Check if mediator is writing elsewhere
ps aux | grep mediator_phase3
# Check process stderr/stdout
ls -la /proc/$(pgrep mediator_phase3)/fd/
```

### Mediator not receiving calls
```bash
# Check socket exists
ls -la /var/xen/qemu/root-176/tmp/vgpu-mediator.sock

# Check mediator is listening
ss -lx | grep vgpu-mediator

# Check connection count
tail -100 /tmp/mediator.log | grep CONNECTION
```
