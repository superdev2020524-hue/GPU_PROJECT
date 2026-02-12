# vGPU Stub MMIO Communication Implementation Roadmap

**Date:** February 12, 2026  
**Based on:** Existing vGPU stub (complete.txt) + MMIO transition plan (core.txt)  
**Goal:** Replace NFS-based communication with MMIO-based PCI communication

---

## Phase 1: Extend vGPU Stub MMIO Layout

### Current MMIO Map (4KB BAR0)
```
Offset | Size | Access | Current Use
-------|------|--------|----------------------------------
0x000  | 4B   | R/W    | Command register
0x004  | 4B   | RO     | Status register
0x008  | 4B   | RO     | Pool ID (ASCII char)
0x00C  | 4B   | RO     | Priority (0/1/2)
0x010  | 4B   | RO     | VM ID
0x014+ | -    | -      | Unused (3.96KB available)
```

### Target MMIO Map (Extended)
```
Offset  | Size | Access | New Purpose
--------|------|--------|----------------------------------
0x000   | 4B   | R/W    | Doorbell register (guest writes to notify)
0x004   | 4B   | RO     | Status register (0=idle, 1=busy, 2=done, 3=error)
0x008   | 4B   | RO     | Pool ID (ASCII char - keep existing)
0x00C   | 4B   | RO     | Priority (0/1/2 - keep existing)
0x010   | 4B   | RO     | VM ID (keep existing)
0x014   | 4B   | RO     | Error code register (0=no error, 1-255=error codes)
0x018   | 4B   | R/W    | Request length (guest writes size of request)
0x01C   | 4B   | RO     | Response length (host writes size of response)
0x020   | 4B   | RO     | Protocol version (hardcoded 0x00010000 = v1.0)
0x024   | 28B  | -      | Reserved for future control registers
        |      |        |
0x040   | 1KB  | R/W    | Request buffer (guest writes command/data here)
0x440   | 1KB  | RO     | Response buffer (host writes result here)
0x840   | 1.9K | -      | Reserved for future use
```

### Files to Modify
- **File:** `~/vgpu-build/rpmbuild/SOURCES/vgpu-stub.c`
- **Changes:**
  1. Update `VGPUStubState` structure
  2. Extend `vgpu_mmio_read()` handler
  3. Extend `vgpu_mmio_write()` handler
  4. Add request/response buffer handling

---

## Phase 2: Add Request/Response Processing

### Host-Side Processing Flow
```
Guest writes request → MMIO write handler → Push to mediator queue
                                                ↓
Guest reads response ← MMIO read handler ← Mediator completes work
```

### Implementation Tasks

#### Task 2.1: Add Request Queue Interface
**File:** `vgpu-stub.c` (new section)
- Add queue structure or socket connection to mediator
- Function: `vgpu_submit_request(VGPUStubState *s, uint8_t *data, uint32_t len)`
- Function: `vgpu_receive_response(VGPUStubState *s, uint8_t *buf, uint32_t *len)`

#### Task 2.2: Implement Doorbell Handler
**File:** `vgpu-stub.c` → `vgpu_mmio_write()`
```c
case 0x000:  /* Doorbell register */
    if (val == 1) {
        // Guest rang doorbell
        uint32_t req_len = s->request_length;
        // Read from request buffer (0x040-0x43F)
        // Push to mediator via socket/queue
        vgpu_submit_request(s, s->request_buffer, req_len);
        s->status_reg = 1;  // Set to "busy"
    }
    break;
```

#### Task 2.3: Add Response Notification
- When mediator completes request, callback updates:
  - Response buffer (0x440-0x83F)
  - Response length register (0x01C)
  - Status register (0x004) = 2 (done) or 3 (error)
  - Error code register (0x014) if applicable

---

## Phase 3: Create Guest Client Library

### Current NFS-based Client (ASSUMPTION)
```c
// Current approach (via NFS)
write_file("/mnt/vgpu/pool_A/request_vm200.json", request_data);
read_file("/mnt/vgpu/pool_A/response_vm200.json", response_data);
```

### Target MMIO-based Client
```c
// New approach (via MMIO)
vgpu_send_request(mmio_base, request_data, req_len);
vgpu_wait_response(mmio_base, response_data, &resp_len);
```

### Files to Create
**New file:** `~/vgpu-build/guest_client/vgpu_mmio_client.c`
**New file:** `~/vgpu-build/guest_client/vgpu_mmio_client.h`

### Client Functions
```c
// Initialize: map MMIO region
void* vgpu_init(const char *pci_device_path);

// Send request: write to MMIO + ring doorbell
int vgpu_send_request(void *mmio_base, void *data, uint32_t len);

// Poll status: read status register
int vgpu_poll_status(void *mmio_base, uint32_t timeout_ms);

// Read response: copy from response buffer
int vgpu_read_response(void *mmio_base, void *buf, uint32_t *len);

// Combined: send + wait + receive
int vgpu_execute_request(void *mmio_base, void *req, uint32_t req_len,
                         void *resp, uint32_t *resp_len, uint32_t timeout_ms);

// Cleanup
void vgpu_cleanup(void *mmio_base);
```

---

## Phase 4: Update Host Mediator

### Current Mediator (ASSUMPTION - need actual code)
```python
# Watches /mnt/vgpu/pool_A/ for new request files
# Reads JSON request
# Executes CUDA work
# Writes JSON response file
```

### Target Mediator Interface
```python
# Receives requests from vgpu-stub via Unix socket or shared memory
# Executes CUDA work (same as before)
# Sends response back to vgpu-stub
```

### Files to Modify
**File:** `<mediator_daemon_path>/mediator.py` (or similar - need actual path)

### Changes Needed
1. Replace file watching with socket/queue listening
2. Parse binary request format (instead of JSON files)
3. Send response via socket (instead of writing file)
4. Keep existing scheduling and CUDA logic unchanged

### Communication Channel Options
**Option A: Unix Domain Socket**
- vgpu-stub creates `/tmp/vgpu-stub-<vm_id>.sock`
- Mediator connects and receives requests
- Pros: Simple, standard IPC
- Cons: Need to handle socket lifecycle

**Option B: Shared Memory Queue**
- vgpu-stub and mediator use shared memory segment
- Lock-free queue for low latency
- Pros: Faster, no context switch
- Cons: More complex implementation

**Recommendation:** Start with Unix socket (Option A) for prototype

---

## Phase 5: Testing Strategy

### Test Level 1: MMIO Register Access
```bash
# In guest VM
# Verify extended register map
sudo ./test_mmio_registers
```

### Test Level 2: Request/Response Round-Trip
```bash
# In guest VM
# Send simple request, verify response
sudo ./test_request_response
```

### Test Level 3: End-to-End CUDA Execution
```bash
# In guest VM
# Send actual CUDA kernel request
sudo ./test_cuda_execution
```

### Test Level 4: Load and Priority Testing
```bash
# Multiple VMs sending requests
# Verify priority scheduling works
./test_multi_vm_priority.sh
```

---

## Phase 6: Remove NFS Dependencies

### Cleanup Tasks
1. Remove NFS mount from VM configuration
2. Remove file-based request/response code
3. Update documentation
4. Archive old NFS-based code

---

## Directory Structure for Implementation

```
/home/david/Downloads/gpu/step2_addtion/
├── core.txt                          # Original plan (exists)
├── IMPLEMENTATION_ROADMAP.md         # This file
├── vgpu_stub_enhanced/
│   ├── vgpu-stub.c                   # Modified QEMU device
│   ├── vgpu-stub.h                   # Header with register definitions
│   ├── vgpu-protocol.h               # Shared protocol definitions
│   ├── Makefile.patch                # Patch for hw/misc/Makefile.objs
│   └── qemu.spec.patch               # Patch for RPM spec file
├── guest_client/
│   ├── vgpu_mmio_client.c            # Client library implementation
│   ├── vgpu_mmio_client.h            # Client library header
│   ├── test_mmio_registers.c         # Test: read all registers
│   ├── test_request_response.c       # Test: simple request/response
│   ├── test_cuda_execution.c         # Test: actual CUDA work
│   ├── Makefile                      # Build guest utilities
│   └── README.md                     # Guest client usage guide
├── host_mediator/
│   ├── mediator_socket_adapter.py    # New socket interface
│   ├── mediator_legacy_file.py       # Old NFS interface (for reference)
│   ├── protocol_parser.py            # Binary protocol parser
│   └── README.md                     # Mediator update guide
├── testing/
│   ├── test_multi_vm.sh              # Multi-VM test script
│   ├── test_priority.sh              # Priority scheduling test
│   └── benchmark.sh                  # Performance comparison
├── docs/
│   ├── PROTOCOL_SPEC.md              # Binary protocol specification
│   ├── REGISTER_MAP.md               # Complete MMIO register map
│   ├── MIGRATION_GUIDE.md            # NFS → MMIO migration guide
│   └── API_REFERENCE.md              # Client library API docs
└── build_scripts/
    ├── build_qemu.sh                 # Automated QEMU build
    ├── install_qemu.sh               # Install custom QEMU
    ├── deploy_guest_client.sh        # Deploy client to VMs
    └── rollback.sh                   # Restore original QEMU
```

---

## Next Steps - What I Need From You

To proceed with implementation, please provide:

### 1. **Current Mediator Code**
- Location of mediator daemon source code
- How it currently reads NFS files
- Request/response format (JSON structure)
- CUDA execution interface

### 2. **Guest Client Code** (if exists)
- How guest VMs currently submit requests
- Current NFS mount configuration
- Request submission flow

### 3. **Request/Response Protocol**
- What data is sent in requests? (kernel name, parameters, buffer pointers?)
- What data is returned? (results, timings, errors?)
- Maximum request/response sizes

### 4. **Priority Scheduling Details**
- How does mediator implement priority queues?
- How are pool_id values used?
- VM scheduling algorithm

### 5. **CUDA Execution Context**
- What CUDA operations are performed?
- Memory management approach
- Error handling

---

## Implementation Order (Recommended)

### Sprint 1: Extend vGPU Stub (1-2 days)
- Modify vgpu-stub.c with new register map
- Add request/response buffers
- Implement basic doorbell handler
- Test: verify new registers accessible from guest

### Sprint 2: Create Guest Client Library (2-3 days)
- Implement MMIO client functions
- Create test programs
- Test: send/receive dummy data via MMIO

### Sprint 3: Add Socket Interface (2-3 days)
- Add Unix socket to vgpu-stub
- Create socket adapter for mediator
- Test: end-to-end dummy request flow

### Sprint 4: Integrate with Mediator (3-4 days)
- Update mediator to receive from socket
- Implement binary protocol parser
- Keep existing CUDA execution logic
- Test: actual CUDA execution via MMIO

### Sprint 5: Testing & Validation (2-3 days)
- Multi-VM testing
- Priority scheduling verification
- Performance benchmarking
- Bug fixes

### Sprint 6: Cleanup & Documentation (1-2 days)
- Remove NFS dependencies
- Update all documentation
- Create migration guide

**Total Estimated Time: 11-17 days**

---

## Risk Mitigation

### Backup Strategy
1. Keep NFS path working during development
2. Support both paths via compile flag initially
3. Gradual migration VM by VM

### Rollback Plan
1. Original vgpu-stub.c backed up
2. Can reinstall original QEMU
3. NFS infrastructure remains available

### Testing Checkpoints
- Each sprint ends with working functionality
- Never break existing VMs
- Incremental validation at each step

---

## Success Criteria

✅ **Phase 1 Complete:** Extended MMIO registers readable from guest  
✅ **Phase 2 Complete:** Doorbell rings and triggers host-side handler  
✅ **Phase 3 Complete:** Guest client can send/receive via MMIO  
✅ **Phase 4 Complete:** Mediator processes requests from MMIO path  
✅ **Phase 5 Complete:** CUDA execution works end-to-end via MMIO  
✅ **Phase 6 Complete:** NFS removed, all VMs using MMIO path  

---

**I'm ready to start implementing! Please provide the mediator code and current client code, and I'll begin with Sprint 1.**
