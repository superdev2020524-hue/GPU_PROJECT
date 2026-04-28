# Host Quick Check Commands

## Run These Commands on Host (root@10.25.33.10)

### 1. Check Mediator Logs for CUDA Calls (MOST IMPORTANT)
```bash
tail -200 /tmp/mediator.log | grep -E 'CUDA_CALL|vm_id=111|vm=111|Total processed|SOCKET|CONNECTION|PERSIST'
```

**Expected output:**
- Connection from VGPU-STUB
- CUDA_CALL_MEM_ALLOC received
- Total processed counter

### 2. Check Recent Mediator Activity
```bash
tail -50 /tmp/mediator.log
```

**Look for:**
- Any CUDA-related messages
- Connection activity
- Error messages

### 3. Monitor Real-Time (While VM Sends Calls)
```bash
tail -f /tmp/mediator.log | grep -E 'CUDA|vm_id=111|processed|SOCKET|CONNECTION'
```

**Then I'll trigger activity from VM** - watch for new messages appearing

### 4. Check Socket Connection
```bash
ls -la /var/xen/qemu/root-176/tmp/vgpu-mediator.sock
ss -lx | grep vgpu-mediator
```

**Expected:**
- Socket file exists
- Mediator is listening

### 5. Check Physical GPU
```bash
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
```

**Look for:**
- Memory usage > 0
- Utilization > 0%

## What VM is Sending Right Now

**Call type:** `CUDA_CALL_MEM_ALLOC` (call_id=0x0030)
**Size:** 545,947,648 bytes (520 MB)
**Frequency:** Multiple calls when Ollama runs

**VM logs show:**
```
[cuda-transport] SENDING to VGPU-STUB: call_id=0x0030 seq=1
[cuda-transport] RINGING DOORBELL: MMIO write to VGPU-STUB
[cuda-transport] RECEIVED from VGPU-STUB: status=DONE
```

## Timing

**Current time on VM:** Check VM logs for timestamps around:
- 08:32:28
- 08:34:04
- And when I trigger new activity

**Adjust for timezone** when checking host logs.

## Report Back

After running the commands above, please tell me:

1. **Did you see CUDA calls in mediator.log?**
   - Yes/No
   - What call_id values?
   - What is "Total processed" count?

2. **Did you see connection activity?**
   - SOCKET/CONNECTION messages?
   - PERSIST messages?

3. **Did you see CUDA executor logs?**
   - cuMemAlloc messages?
   - Physical GPU operations?

4. **Any errors?**
   - Connection errors?
   - CUDA errors?
   - Other issues?
