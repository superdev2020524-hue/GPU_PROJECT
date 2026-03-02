# VGPU-STUB Logging Added

## Changes Made

Added comprehensive logging to `vgpu-stub-enhanced.c` to track CUDA doorbell processing:

1. **Doorbell Reception**: Logs when MMIO doorbell register is written
2. **Doorbell Processing**: Logs when `vgpu_process_cuda_doorbell()` is called
3. **Mediator Connection**: Logs when connecting to mediator
4. **Message Sending**: Logs when sending CUDA call to mediator via socket

## Log Messages Added

### 1. Doorbell Ring Detection
```c
[vgpu] vm_id=%u: CUDA DOORBELL RING: call_id=0x%04x seq=%u (addr=0x%lx)
```

### 2. Doorbell Processing Start
```c
[vgpu] vm_id=%u: PROCESSING CUDA DOORBELL: call_id=0x%04x seq=%u args=%u data_len=%u
```

### 3. Mediator Connection Attempt
```c
[vgpu] vm_id=%u: Connecting to mediator for CUDA call 0x%04x
```

### 4. Mediator Connection Error
```c
[vgpu] vm_id=%u: ERROR: Cannot connect to mediator (call_id=0x%04x)
```

### 5. Message Sending
```c
[vgpu] vm_id=%u: SENDING CUDA CALL to mediator: call_id=0x%04x seq=%u total_bytes=%zu (fd=%d)
```

### 6. Message Sent Successfully
```c
[vgpu] vm_id=%u: CUDA CALL SENT to mediator: %zd bytes (call_id=0x%04x seq=%u)
```

## Next Steps

1. **Rebuild QEMU** with the updated `vgpu-stub-enhanced.c`
2. **Restart QEMU** to load the new VGPU-STUB code
3. **Check QEMU logs** on the host for the new log messages
4. **Verify mediator receives** the CUDA calls

## How to Check Logs

### On the Host (where QEMU runs):
```bash
# Check QEMU process logs
journalctl -u qemu* --since '5 minutes ago' | grep -E '\[vgpu\].*CUDA|\[vgpu\].*DOORBELL|\[vgpu\].*SENDING'

# Or if QEMU runs in a terminal, check that terminal output
# Or check QEMU's stderr if redirected to a file
```

### Expected Flow in Logs:
1. `[vgpu] vm_id=X: CUDA DOORBELL RING: call_id=0x0030 seq=1`
2. `[vgpu] vm_id=X: PROCESSING CUDA DOORBELL: call_id=0x0030 seq=1 args=4 data_len=0`
3. `[vgpu] vm_id=X: SENDING CUDA CALL to mediator: call_id=0x0030 seq=1 total_bytes=...`
4. `[vgpu] vm_id=X: CUDA CALL SENT to mediator: ... bytes`

If you see steps 1-4, then VGPU-STUB is successfully receiving and forwarding CUDA calls to the mediator.

If you see step 1 but not step 2, there's an issue with doorbell processing.
If you see steps 1-2 but not step 3, there's an issue with mediator connection.
If you see step 3 but not step 4, there's an issue with socket communication.
