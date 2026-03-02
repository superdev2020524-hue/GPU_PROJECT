# Host-Side Verification Instructions

## Purpose

Verify that VGPU-STUB (in QEMU) is receiving doorbell rings from the guest and forwarding CUDA calls to the mediator, and that the mediator is processing them.

## Step 1: Check QEMU Logs for VGPU-STUB Activity

**On the host (root@10.25.33.10):**

```bash
# Find QEMU process for VM 111 (test-11)
ps aux | grep qemu | grep root-176

# Check QEMU logs for VGPU-STUB doorbell activity
# (Location depends on how QEMU was started - check systemd or journalctl)
journalctl -u qemu* --since '10 minutes ago' | grep -E '\[vgpu\].*CUDA|\[vgpu\].*DOORBELL|vm_id=111'

# Or if QEMU logs to a file:
# (You may need to find where QEMU logs are redirected)
tail -100 /var/log/qemu/vm-*.log | grep -E '\[vgpu\].*CUDA|\[vgpu\].*DOORBELL'
```

**What to look for:**
- ✅ `[vgpu] vm_id=111: CUDA DOORBELL RING (op=0x0030, seq=1, pid=130074)`
- ✅ `[vgpu] vm_id=111: PROCESSING CUDA DOORBELL (op=0x0030, seq=1, data_len=0)`
- ✅ `[vgpu] vm_id=111: SENDING CUDA CALL to mediator (op=0x0030, seq=1, sent=XX)`

**If you don't see these logs:**
- QEMU may not have been rebuilt with the logging changes
- QEMU logs may be in a different location
- The logging code may not be active

## Step 2: Check Mediator Logs for Received CUDA Calls

**On the host:**

```bash
# Check mediator logs for CUDA calls from VM 111
tail -200 /tmp/mediator.log | grep -E 'CUDA_CALL|vm_id=111|vm=111|Total processed'

# Or check for connection activity
tail -200 /tmp/mediator.log | grep -E 'SOCKET|CONNECTION|PERSIST|vm_id=111'
```

**What to look for:**
- ✅ `[SOCKET] New connection on /var/xen/qemu/root-176/tmp/vgpu-mediator.sock`
- ✅ `[CONNECTION] New connection from (fd=XX)`
- ✅ `[PERSIST] fd=XX registered for persistent polling`
- ✅ `[cuda-executor] CUDA_CALL_INIT vm=111 — pipeline live`
- ✅ `Total processed: X` (should increase when guest sends calls)

**If you see "Total processed: 0":**
- VGPU-STUB may not be forwarding to mediator
- Socket connection may not be established
- Check if mediator socket exists: `ls -l /var/xen/qemu/root-176/tmp/vgpu-mediator.sock`

## Step 3: Check CUDA Executor Logs

**On the host:**

```bash
# Check for CUDA executor activity
tail -200 /tmp/mediator.log | grep -E '\[cuda-executor\]|cuMemAlloc|cuMemcpy|cuLaunchKernel|CUDA_CALL_MEM'
```

**What to look for:**
- ✅ `[cuda-executor] cuMemAlloc: allocating X bytes on physical GPU (vm=111)`
- ✅ `[cuda-executor] cuMemAlloc SUCCESS: allocated 0x... on physical GPU (vm=111)`
- ✅ `[cuda-executor] cuMemcpyHtoD: copying X bytes (vm=111)`
- ✅ `[cuda-executor] cuLaunchKernel: launching kernel (vm=111)`

**If you don't see executor logs:**
- CUDA calls may not be reaching the executor
- The calls may be different types (unified memory APIs)
- Check what call_id values are being sent

## Step 4: Monitor Real-Time Activity

**On the host (run this while VM is sending CUDA calls):**

```bash
# Monitor mediator logs in real-time
tail -f /tmp/mediator.log | grep -E 'CUDA|vm_id=111|processed|SOCKET|CONNECTION'
```

**Then trigger CUDA activity from the VM** (I'll do this from the VM side)

**What to watch for:**
- New connections appearing
- "Total processed" counter increasing
- CUDA executor logs appearing
- Any error messages

## Step 5: Check Physical GPU Activity

**On the host:**

```bash
# Monitor GPU utilization
watch -n 1 nvidia-smi

# Or check once
nvidia-smi

# Check GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

**What to look for:**
- GPU memory usage increasing when VM sends CUDA calls
- GPU utilization percentage > 0% during operations
- Processes using GPU memory

## Troubleshooting

### Problem: No VGPU-STUB logs in QEMU

**Possible causes:**
- QEMU wasn't rebuilt with logging changes
- QEMU logs are in a different location
- Logging code isn't active

**Solution:**
- Check if QEMU has vgpu-cuda device: `/usr/lib64/xen/bin/qemu-system-i386 -device help | grep vgpu`
- If device exists but no logs, logging may need to be enabled
- Check QEMU source: `vgpu-stub-enhanced.c` should have the logging code

### Problem: Mediator shows "Total processed: 0"

**Possible causes:**
- VGPU-STUB not forwarding to mediator
- Socket connection not established
- CUDA calls not reaching mediator

**Solution:**
- Check mediator socket exists: `ls -l /var/xen/qemu/root-176/tmp/vgpu-mediator.sock`
- Check mediator is listening: `tail -50 /tmp/mediator.log | grep SOCKET`
- Verify connection: `tail -50 /tmp/mediator.log | grep CONNECTION`

### Problem: CUDA executor not logging

**Possible causes:**
- Different CUDA call types (unified memory vs traditional)
- Calls not reaching executor
- Executor logging not enabled

**Solution:**
- Check what call_id values are being sent (from VM logs)
- Verify executor is initialized: `tail -100 /tmp/mediator.log | grep 'CUDA executor ready'`
- Check if calls match executor's supported call types

## Expected Flow

1. **Guest VM** → Shim intercepts CUDA call
2. **Shim** → Sends to VGPU-STUB via MMIO (doorbell)
3. **VGPU-STUB (QEMU)** → Receives doorbell, forwards to mediator via socket
4. **Mediator** → Receives CUDA call, forwards to CUDA executor
5. **CUDA Executor** → Replays call on physical GPU
6. **Physical GPU** → Executes operation
7. **Result flows back** → Executor → Mediator → VGPU-STUB → Shim → Guest

## Verification Checklist

- [ ] QEMU logs show VGPU-STUB receiving doorbell rings
- [ ] QEMU logs show VGPU-STUB forwarding to mediator
- [ ] Mediator logs show new connection from VGPU-STUB
- [ ] Mediator logs show CUDA calls being received
- [ ] Mediator "Total processed" counter increases
- [ ] CUDA executor logs show operations on physical GPU
- [ ] Physical GPU shows memory/utilization activity

## Next Steps After Verification

Once you confirm the host-side is receiving and processing:
1. We can investigate why Ollama crashes (likely shim response handling)
2. We can verify end-to-end data flow
3. We can test with different CUDA operations
