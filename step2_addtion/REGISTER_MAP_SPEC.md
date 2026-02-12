# vGPU Stub MMIO Register Map Specification

**Version:** 2.0 (MMIO Communication)  
**Date:** February 12, 2026  
**Status:** Design Specification

---

## Overview

This document specifies the complete MMIO register layout for the enhanced vGPU stub device that supports direct guest-to-host communication via PCI MMIO, replacing the previous NFS-based transport.

## PCI Device Identification

```
Vendor ID:    0x1AF4 (Red Hat, Inc.)
Device ID:    0x1111 (vGPU Stub)
Class Code:   0x1200 (Processing Accelerator)
Revision:     0x02 (incremented from 0x01 to indicate MMIO comm support)
BAR0:         4KB MMIO region (64-bit prefetchable or 32-bit non-prefetchable)
```

---

## Complete MMIO Register Map

### Control Register Block (0x000 - 0x03F)

| Offset | Name | Size | Access | Reset | Description |
|--------|------|------|--------|-------|-------------|
| 0x000 | DOORBELL | 4B | R/W | 0x00000000 | Guest writes 1 to submit request, auto-clears |
| 0x004 | STATUS | 4B | RO | 0x00000000 | Device status (see STATUS codes below) |
| 0x008 | POOL_ID | 4B | RO | 'A' (0x41) | GPU pool ID ('A'=0x41, 'B'=0x42) |
| 0x00C | PRIORITY | 4B | RO | 0x00000001 | Priority level (0=low, 1=med, 2=high) |
| 0x010 | VM_ID | 4B | RO | 0x00000000 | VM identifier (0-4294967295) |
| 0x014 | ERROR_CODE | 4B | RO | 0x00000000 | Last error code (see ERROR codes below) |
| 0x018 | REQUEST_LEN | 4B | R/W | 0x00000000 | Guest writes request length in bytes (max 1024) |
| 0x01C | RESPONSE_LEN | 4B | RO | 0x00000000 | Host writes response length in bytes |
| 0x020 | PROTOCOL_VER | 4B | RO | 0x00010000 | Protocol version (major<<16 \| minor) = v1.0 |
| 0x024 | CAPABILITIES | 4B | RO | 0x00000001 | Feature bits (bit 0: basic request/response) |
| 0x028 | INTERRUPT_CTRL | 4B | R/W | 0x00000000 | Interrupt control (bit 0: enable completion IRQ) |
| 0x02C | INTERRUPT_STATUS | 4B | RW1C | 0x00000000 | Interrupt status (bit 0: response ready) |
| 0x030 | REQUEST_ID | 4B | R/W | 0x00000000 | Request ID for tracking (guest-assigned) |
| 0x034 | TIMESTAMP_LO | 4B | RO | varies | Request completion timestamp (low 32 bits) |
| 0x038 | TIMESTAMP_HI | 4B | RO | varies | Request completion timestamp (high 32 bits) |
| 0x03C | SCRATCH | 4B | R/W | 0x00000000 | Scratch register for testing |

**Notes:**
- RW1C = Read/Write-1-to-Clear (write 1 to clear bit, write 0 has no effect)
- All multi-byte values are little-endian
- Reserved bits read as 0, writes ignored

### Request Buffer (0x040 - 0x43F)

| Offset | Name | Size | Access | Description |
|--------|------|------|--------|-------------|
| 0x040 | REQUEST_BUF | 1024B | R/W | Guest writes request payload here |

**Usage:**
1. Guest writes request data to this region
2. Guest writes length to REQUEST_LEN (0x018)
3. Guest writes 1 to DOORBELL (0x000)
4. Host reads request from this buffer

### Response Buffer (0x440 - 0x83F)

| Offset | Name | Size | Access | Description |
|--------|------|------|--------|-------------|
| 0x440 | RESPONSE_BUF | 1024B | RO | Host writes response payload here |

**Usage:**
1. Host processes request
2. Host writes response data to this region
3. Host writes length to RESPONSE_LEN (0x01C)
4. Host updates STATUS to DONE (0x02)
5. Guest reads response from this buffer

### Reserved Region (0x840 - 0xFFF)

| Offset | Name | Size | Access | Description |
|--------|------|------|--------|-------------|
| 0x840 | RESERVED | 1976B | RO | Reserved for future use (reads as 0) |

---

## Register Definitions

### STATUS Register (0x004)

| Value | Name | Description |
|-------|------|-------------|
| 0x00 | IDLE | Device idle, ready for new request |
| 0x01 | BUSY | Processing request |
| 0x02 | DONE | Request completed successfully, response ready |
| 0x03 | ERROR | Request failed, check ERROR_CODE register |

**State Machine:**
```
IDLE --[doorbell=1]--> BUSY --[success]--> DONE --[read response]--> IDLE
                         |
                         +--[failure]--> ERROR --[read error]--> IDLE
```

### ERROR_CODE Register (0x014)

| Value | Name | Description |
|-------|------|-------------|
| 0x00 | NO_ERROR | No error |
| 0x01 | INVALID_REQUEST | Request format invalid |
| 0x02 | REQUEST_TOO_LARGE | Request exceeds 1024 bytes |
| 0x03 | MEDIATOR_UNAVAILABLE | Cannot connect to mediator |
| 0x04 | TIMEOUT | Request processing timeout |
| 0x05 | CUDA_ERROR | CUDA execution failed |
| 0x06 | INVALID_POOL | Pool ID not available |
| 0x07 | QUEUE_FULL | Request queue full |
| 0x08 | UNSUPPORTED_OPERATION | Operation not supported |
| 0x09-0xEF | RESERVED | Reserved for future errors |
| 0xF0-0xFF | CUDA_SPECIFIC | CUDA-specific error codes |

### PRIORITY Register (0x00C)

| Value | Name | Description |
|-------|------|-------------|
| 0 | LOW | Low priority queue |
| 1 | MEDIUM | Medium priority queue (default) |
| 2 | HIGH | High priority queue |

### CAPABILITIES Register (0x024)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | BASIC_REQUEST | Basic request/response supported |
| 1 | INTERRUPT | Interrupt notification supported |
| 2 | DMA | DMA transfers supported (future) |
| 3 | MULTI_REQUEST | Multiple outstanding requests (future) |
| 4-31 | RESERVED | Reserved for future capabilities |

### INTERRUPT_CTRL Register (0x028)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | ENABLE_COMPLETION | Enable interrupt on request completion |
| 1-31 | RESERVED | Reserved |

### INTERRUPT_STATUS Register (0x02C)

| Bit | Name | Description |
|-----|------|-------------|
| 0 | COMPLETION | Request completed (write 1 to clear) |
| 1-31 | RESERVED | Reserved |

---

## Request/Response Protocol

### Request Format (in REQUEST_BUF)

```c
struct vgpu_request {
    uint32_t version;        // Protocol version (0x00010000 for v1.0)
    uint32_t opcode;         // Operation code (see opcodes below)
    uint32_t flags;          // Request flags (see flags below)
    uint32_t param_count;    // Number of parameters
    uint32_t data_offset;    // Offset to data section (bytes from start)
    uint32_t data_length;    // Length of data section
    uint32_t reserved[2];    // Reserved, must be 0
    uint32_t params[];       // Variable-length parameters
    // uint8_t data[];       // Variable-length data follows params
};
```

**Maximum Request Size:** 1024 bytes (including header, params, and data)

### Response Format (in RESPONSE_BUF)

```c
struct vgpu_response {
    uint32_t version;        // Protocol version (0x00010000 for v1.0)
    uint32_t status;         // Response status (0=success, non-zero=error)
    uint32_t result_count;   // Number of result values
    uint32_t data_offset;    // Offset to data section
    uint32_t data_length;    // Length of data section
    uint32_t exec_time_us;   // Execution time in microseconds
    uint32_t reserved[2];    // Reserved
    uint32_t results[];      // Variable-length results
    // uint8_t data[];       // Variable-length data follows results
};
```

**Maximum Response Size:** 1024 bytes

### Operation Codes

| Opcode | Name | Description |
|--------|------|-------------|
| 0x0000 | NOP | No operation (for testing) |
| 0x0001 | CUDA_KERNEL | Execute CUDA kernel |
| 0x0002 | MEMORY_ALLOC | Allocate GPU memory |
| 0x0003 | MEMORY_FREE | Free GPU memory |
| 0x0004 | MEMORY_COPY | Copy memory |
| 0x0005 | GET_DEVICE_INFO | Query device information |
| 0x0006 | SYNCHRONIZE | Synchronize device |
| 0x0100-0x0FFF | RESERVED | Reserved for future operations |
| 0x1000-0xFFFF | CUSTOM | Custom/vendor-specific operations |

### Request Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | ASYNC | Asynchronous execution (don't wait for completion) |
| 1 | HIGH_PRIORITY | Boost priority for this request |
| 2-31 | RESERVED | Reserved |

---

## Communication Flow

### Basic Request/Response Flow

```
1. Guest checks STATUS == IDLE
2. Guest writes request to REQUEST_BUF (0x040)
3. Guest writes request length to REQUEST_LEN (0x018)
4. Guest optionally writes request ID to REQUEST_ID (0x030)
5. Guest writes 1 to DOORBELL (0x000)
6. STATUS changes to BUSY (0x01)

   [Host processing...]

7. Host writes response to RESPONSE_BUF (0x440)
8. Host writes response length to RESPONSE_LEN (0x01C)
9. Host writes timestamps to TIMESTAMP_LO/HI (0x034/0x038)
10. Host writes STATUS to DONE (0x02) or ERROR (0x03)
11. If interrupts enabled, host raises interrupt

12. Guest polls STATUS or receives interrupt
13. Guest reads RESPONSE_LEN (0x01C)
14. Guest reads response from RESPONSE_BUF (0x440)
15. STATUS returns to IDLE (0x00)
```

### Error Handling Flow

```
1-6. Same as basic flow

   [Host encounters error...]

7. Host writes error details to RESPONSE_BUF (optional)
8. Host writes RESPONSE_LEN (if error details provided)
9. Host writes ERROR_CODE (0x014)
10. Host writes STATUS to ERROR (0x03)

11. Guest detects ERROR status
12. Guest reads ERROR_CODE
13. Guest reads error details from RESPONSE_BUF (if RESPONSE_LEN > 0)
14. Guest handles error
15. STATUS returns to IDLE
```

---

## Example Usage

### Example 1: Simple NOP Request (Testing)

**Guest Code:**
```c
volatile uint32_t *mmio = /* mapped BAR0 */;

// Wait for idle
while (mmio[0x004/4] != 0) { /* spin */ }

// Prepare NOP request
struct vgpu_request req = {
    .version = 0x00010000,
    .opcode = 0x0000,  // NOP
    .flags = 0,
    .param_count = 0,
    .data_offset = 0,
    .data_length = 0
};

// Write request
memcpy((void*)(mmio + 0x040/4), &req, sizeof(req));
mmio[0x018/4] = sizeof(req);  // REQUEST_LEN
mmio[0x030/4] = 12345;         // REQUEST_ID (arbitrary)

// Ring doorbell
mmio[0x000/4] = 1;

// Poll for completion
while (mmio[0x004/4] == 0x01) { /* busy */ }

// Check status
if (mmio[0x004/4] == 0x02) {  // DONE
    // Read response
    uint32_t resp_len = mmio[0x01C/4];
    struct vgpu_response resp;
    memcpy(&resp, (void*)(mmio + 0x440/4), resp_len);
    printf("NOP succeeded, exec time: %u us\n", resp.exec_time_us);
} else {  // ERROR
    uint32_t err = mmio[0x014/4];
    printf("Error: %u\n", err);
}
```

### Example 2: CUDA Kernel Execution

**Guest Code:**
```c
// Prepare CUDA kernel request
struct vgpu_request req = {
    .version = 0x00010000,
    .opcode = 0x0001,     // CUDA_KERNEL
    .flags = 0,
    .param_count = 3,
    .data_offset = sizeof(struct vgpu_request) + 3 * sizeof(uint32_t),
    .data_length = kernel_code_len
};

// Build request in buffer
uint8_t buf[1024];
memcpy(buf, &req, sizeof(req));

// Add parameters (e.g., grid size, block size, shared mem)
uint32_t *params = (uint32_t*)(buf + sizeof(req));
params[0] = grid_dim;
params[1] = block_dim;
params[2] = shared_mem_bytes;

// Add kernel code
memcpy(buf + req.data_offset, kernel_code, kernel_code_len);

// Write to MMIO
uint32_t total_len = req.data_offset + kernel_code_len;
memcpy((void*)(mmio + 0x040/4), buf, total_len);
mmio[0x018/4] = total_len;
mmio[0x000/4] = 1;  // Ring doorbell

// Wait and check response...
```

---

## Performance Considerations

### Latency Components
- **MMIO write latency:** ~100-500 ns (VM exit to hypervisor)
- **Doorbell processing:** ~1-10 µs (QEMU handler)
- **Socket to mediator:** ~10-100 µs (IPC)
- **CUDA execution:** varies (µs to ms)
- **Response path:** ~10-100 µs (reverse path)

**Total overhead:** ~20-210 µs (excluding CUDA execution)

**Comparison to NFS:**
- NFS write/read: ~100-1000 µs per operation
- NFS has 2 file operations (write request + read response)
- **Expected speedup:** 10-100x for small requests

### Optimization Tips
1. Batch multiple operations into single request when possible
2. Use async flag for fire-and-forget operations
3. Enable interrupts instead of polling for lower CPU usage
4. Keep requests < 1KB to avoid fragmentation

---

## Compatibility

### Backward Compatibility
- Protocol version field allows detection of supported features
- Old guests can be detected by PROTOCOL_VER register
- New features gated by CAPABILITIES register

### Forward Compatibility
- Reserved fields must be set to 0 by guest
- Host must ignore unknown flags/opcodes gracefully
- New opcodes use previously reserved ranges

---

## Security Considerations

1. **Input Validation:** Host must validate all request fields
2. **Buffer Bounds:** Enforce 1KB max request/response size
3. **DOS Prevention:** Rate limit requests per VM
4. **Isolation:** CUDA execution must not access other VM data
5. **Memory Safety:** No buffer overflows in request/response handling

---

## Future Extensions (v2.0+)

- **DMA Support:** Direct memory access for large transfers
- **Multiple Queues:** Separate queues for different operation types
- **MSI-X Interrupts:** Multiple interrupt vectors for different events
- **Extended Buffers:** > 1KB via BAR1 or DMA
- **Shared Memory:** Persistent shared memory regions
- **Event Notifications:** Asynchronous event delivery

---

**Document Status:** Ready for Implementation  
**Next Step:** Implement in vgpu-stub.c based on this specification
