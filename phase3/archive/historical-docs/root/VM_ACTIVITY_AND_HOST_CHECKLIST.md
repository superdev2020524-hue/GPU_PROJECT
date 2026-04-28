# VM Activity Summary & Host Verification Checklist

## VM Side: Confirmed Activity

### ✅ CUDA Calls Being Intercepted:
- `cudaMalloc()` - **545,947,648 bytes** (520 MB) - **SENT TO VGPU-STUB** ✅
- `cudaMallocHost()` - Host memory allocation
- `cuMemCreate()` - Unified memory (returns dummy, not sent)
- `cuMemMap()` - Unified memory mapping (returns dummy, not sent)
- `cudaGetDevice()` - Device queries
- `cublasSgemm_v2()` - Matrix operations

### ✅ Transport Activity:
```
[cuda-transport] SENDING to VGPU-STUB: call_id=0x0030 seq=1 args=4 data_len=0
[cuda-transport] RINGING DOORBELL: MMIO write to VGPU-STUB
[cuda-transport] RECEIVED from VGPU-STUB: call_id=0x0030 seq=1 status=DONE
```

**Note:** `call_id=0x0030` = `CUDA_CALL_MEM_ALLOC` (memory allocation)

### 📊 Statistics:
- **4 transport calls** sent in the last 5 minutes
- **All calls**: `call_id=0x0030` (CUDA_CALL_MEM_ALLOC)
- **All responses**: `status=DONE`

## Host-Side Verification Checklist

**Instructions:** See `HOST_SIDE_VERIFICATION_INSTRUCTIONS.md` for detailed steps.

### Quick Verification Commands:

#### 1. Check QEMU/VGPU-STUB Logs
```bash
# On host (root@10.25.33.10)
journalctl -u qemu* --since '10 minutes ago' | grep -E '\[vgpu\].*CUDA|\[vgpu\].*DOORBELL|vm_id=111'
```

**Expected:**
- `[vgpu] vm_id=111: CUDA DOORBELL RING (op=0x0030, seq=1)`
- `[vgpu] vm_id=111: PROCESSING CUDA DOORBELL`
- `[vgpu] vm_id=111: SENDING CUDA CALL to mediator`

#### 2. Check Mediator Logs
```bash
tail -200 /tmp/mediator.log | grep -E 'CUDA_CALL|vm_id=111|vm=111|Total processed|SOCKET|CONNECTION'
```

**Expected:**
- `[SOCKET] New connection` or `[CONNECTION] New connection`
- `[cuda-executor] CUDA_CALL_MEM_ALLOC vm=111` or similar
- `Total processed: X` (should be > 0)

#### 3. Check CUDA Executor
```bash
tail -200 /tmp/mediator.log | grep -E '\[cuda-executor\]|cuMemAlloc|CUDA_CALL_MEM_ALLOC'
```

**Expected:**
- `[cuda-executor] cuMemAlloc: allocating 545947648 bytes on physical GPU (vm=111)`
- `[cuda-executor] cuMemAlloc SUCCESS: allocated 0x... on physical GPU`

#### 4. Monitor Real-Time (While VM Sends Calls)
```bash
tail -f /tmp/mediator.log | grep -E 'CUDA|vm_id=111|processed'
```

**Then trigger activity from VM** (I'll do this when you're ready)

## Timing Information

**Last VM activity triggered at:** `Sun Mar  1 10:33:56 PM JST 2026`

**When checking host logs, look for activity around:**
- 08:33:56 - 08:34:04 (VM time)
- Adjust for timezone difference if needed

## What to Report

After checking host-side logs, please report:

1. **QEMU/VGPU-STUB:**
   - [ ] Did you see doorbell ring logs?
   - [ ] Did you see "SENDING CUDA CALL to mediator"?
   - [ ] If not, what did you see?

2. **Mediator:**
   - [ ] Did you see connection from VGPU-STUB?
   - [ ] Did you see CUDA_CALL_MEM_ALLOC received?
   - [ ] What is "Total processed" count?
   - [ ] Any error messages?

3. **CUDA Executor:**
   - [ ] Did you see cuMemAlloc logs?
   - [ ] Did you see physical GPU operations?
   - [ ] Any errors?

4. **Physical GPU:**
   - [ ] Did GPU memory/utilization increase?
   - [ ] Any processes using GPU?

## Next Steps Based on Results

### If Host Receives Calls ✅:
- Investigate why Ollama crashes (likely shim response handling)
- Verify end-to-end data flow
- Test with more CUDA operations

### If Host Doesn't Receive ❌:
- Check QEMU/VGPU-STUB connection to mediator
- Verify socket paths
- Check if QEMU was rebuilt with logging

### If Executor Not Processing ⚠️:
- Check call_id mapping
- Verify executor supports CUDA_CALL_MEM_ALLOC
- Check if responses are being sent back
