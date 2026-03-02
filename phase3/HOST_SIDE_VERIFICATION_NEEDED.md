# Host-Side Verification Needed

## Current Status

### ✅ Guest-Side (VM) - CONFIRMED WORKING
The shim is successfully sending commands to VGPU-STUB:

```
[libvgpu-cudart] ABOUT TO CALL transport: call_id=0x0030 size=545947648 transport=0x715b40028300 (pid=129294)
[cuda-transport] cuda_transport_call() INVOKED: call_id=0x0030 data_len=0 tp=0x715b40028300 bar0=0x715b95824000 (pid=129294)
[cuda-transport] SENDING to VGPU-STUB: call_id=0x0030 seq=1 args=4 data_len=0 (pid=129294)
[cuda-transport] RINGING DOORBELL: MMIO write to VGPU-STUB (call_id=0x0030, pid=129294)
[libvgpu-cudart] AFTER transport call: result=0 status=0 num_results=1 (pid=129294)
```

**Key Observations:**
1. Transport is initialized (`bar0=0x715b95824000` - valid MMIO mapping)
2. Commands are being sent (`SENDING to VGPU-STUB`)
3. Doorbell is being rung (`RINGING DOORBELL`)
4. Calls are completing successfully (`result=0 status=0 num_results=1`)

### ❓ Host-Side (Physical Machine) - NEEDS VERIFICATION

The VGPU-STUB device (part of QEMU) should:
1. Receive the MMIO doorbell write
2. Process the CUDA call via `vgpu_process_cuda_doorbell()`
3. Forward the request to the mediator daemon via Unix socket
4. Wait for the mediator's response
5. Write the response back to MMIO registers

## What to Check on the Host

### 1. QEMU Logs
Check if QEMU (which runs VGPU-STUB) is logging doorbell events:
```bash
# On the host, check QEMU logs
journalctl -u qemu* --since '5 minutes ago' | grep -E 'vgpu|CUDA|doorbell'
```

Or if QEMU is running in a terminal, check for:
- `[vgpu] vm_id=X: CUDA doorbell received`
- `[vgpu] vm_id=X: Connected to mediator`
- `[vgpu] vm_id=X: Sending CUDA call to mediator`

### 2. Mediator Daemon Logs
The mediator daemon should be receiving CUDA calls:
```bash
# On the host, check mediator logs
journalctl -u mediator.service --since '5 minutes ago' | grep -E 'CUDA|cuMemAlloc|call_id'
```

Or check `/tmp/mediator.log` if it exists:
```bash
tail -f /tmp/mediator.log | grep -E 'CUDA|cuMemAlloc'
```

### 3. Mediator Service Status
Verify the mediator is running:
```bash
systemctl status mediator.service
```

### 4. Unix Socket Connection
Verify the socket exists and is accessible:
```bash
ls -l /tmp/vgpu-mediator.sock
# Or check the path defined in VGPU_SOCKET_PATH
```

### 5. CUDA Executor Logs
Check if the CUDA executor is receiving and processing calls:
```bash
# Look for logs from cuda_executor.c
journalctl -u mediator.service | grep -E '\[cuda-executor\]|cuMemAlloc|CUDA_CALL_MEM_ALLOC'
```

## Expected Flow

1. **Guest**: `cudaMalloc()` → `cuda_transport_call()` → MMIO write to doorbell
2. **VGPU-STUB (QEMU)**: Receives MMIO write → `vgpu_process_cuda_doorbell()` → Sends to mediator via socket
3. **Mediator**: Receives socket message → `cuda_executor_call()` → Calls real CUDA API on physical GPU
4. **Mediator**: Returns result → Writes to socket
5. **VGPU-STUB**: Receives socket response → Writes to MMIO registers
6. **Guest**: Polls MMIO status → Reads result → Returns to `cudaMalloc()`

## Next Steps

1. **Add logging to VGPU-STUB** to confirm it receives doorbell rings
2. **Check mediator daemon** is running and receiving messages
3. **Verify socket connection** between VGPU-STUB and mediator
4. **Check CUDA executor** is processing calls on the physical GPU

## Files to Check

- `phase3/src/vgpu-stub-enhanced.c`: VGPU-STUB MMIO handler
- `phase3/src/mediator_phase3.c`: Mediator daemon main loop
- `phase3/src/cuda_executor.c`: CUDA API replay on physical GPU
