# Next Steps: Host-Side Verification

## Current Status

### ✅ Confirmed Working (VM Side):
1. **Shim libraries loaded** in Ollama process
2. **CUDA calls intercepted** - Unified memory APIs (cuMemCreate, cuMemMap, etc.)
3. **Transport connected** to VGPU-STUB at `0000:00:05.0`
4. **Commands sent** to VGPU-STUB via MMIO doorbell
5. **Responses received** from VGPU-STUB (status=DONE)

### ⚠️ Findings:
- **Only one call type sent**: `call_id=0x0030` (CUDA_CALL_INIT)
- **Unified memory APIs not sent**: `cuMemCreate`, `cuMemMap` return dummy values locally
- **Ollama crashes**: Runner process terminates with `exit status 2`
- **No cuMemAlloc/cuMemcpy calls**: Ollama uses unified memory instead

## Host-Side Verification Instructions

**File:** `HOST_SIDE_VERIFICATION_INSTRUCTIONS.md` (created)

### Quick Checklist:

1. **Check QEMU/VGPU-STUB logs:**
   ```bash
   # On host (root@10.25.33.10)
   journalctl -u qemu* --since '10 minutes ago' | grep -E '\[vgpu\].*CUDA|\[vgpu\].*DOORBELL|vm_id=111'
   ```
   **Look for:** Doorbell rings, CUDA call forwarding

2. **Check Mediator logs:**
   ```bash
   tail -200 /tmp/mediator.log | grep -E 'CUDA_CALL|vm_id=111|vm=111|Total processed|SOCKET|CONNECTION'
   ```
   **Look for:** 
   - Connection from VGPU-STUB
   - CUDA_CALL_INIT received
   - "Total processed" counter increasing

3. **Check CUDA Executor:**
   ```bash
   tail -200 /tmp/mediator.log | grep -E '\[cuda-executor\]|cuMemAlloc|CUDA_CALL_MEM'
   ```
   **Look for:** Physical GPU operations

4. **Monitor real-time:**
   ```bash
   tail -f /tmp/mediator.log | grep -E 'CUDA|vm_id=111|processed'
   ```
   Then trigger activity from VM (I'll do this)

## What We Expect to See

### If Everything Works:
- ✅ QEMU logs: `[vgpu] vm_id=111: CUDA DOORBELL RING (op=0x0030, seq=1)`
- ✅ QEMU logs: `[vgpu] vm_id=111: SENDING CUDA CALL to mediator`
- ✅ Mediator logs: `[SOCKET] New connection` or `[CONNECTION] New connection`
- ✅ Mediator logs: `[cuda-executor] CUDA_CALL_INIT vm=111`
- ✅ Mediator logs: `Total processed: X` (increasing)

### If Something's Missing:
- **No QEMU logs**: QEMU may not have logging enabled or wasn't rebuilt
- **No mediator connection**: Socket path issue or VGPU-STUB not connecting
- **No executor logs**: CUDA calls not reaching executor or different call types

## VM Side Activity

I'll trigger more CUDA activity from the VM to help you verify on the host side. The logs will show:
- Transport sending commands to VGPU-STUB
- Doorbell rings
- Responses received

**Timing:** When I trigger the activity, watch your host-side logs in real-time to see if the messages appear.

## Next Steps After Host Verification

Once you confirm host-side reception:

1. **If host receives calls:**
   - Investigate why only CUDA_CALL_INIT is sent
   - Check if unified memory APIs need transport implementation
   - Fix Ollama crash (likely invalid handle returns)

2. **If host doesn't receive:**
   - Check QEMU/VGPU-STUB connection to mediator
   - Verify socket paths
   - Check mediator is listening on correct socket

3. **If executor not processing:**
   - Check call_id mapping
   - Verify executor supports the call types
   - Check if unified memory APIs need executor support
