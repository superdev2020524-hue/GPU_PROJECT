# Migration Plan: NFS → MMIO Communication

**Date:** February 12, 2026  
**Source System:** `/home/david/Downloads/gpu/step2_test/` (NFS-based)  
**Target System:** `/home/david/Downloads/gpu/step2_addtion/` (MMIO-based)

---

## Executive Summary

This document provides a **concrete migration plan** to transition your working NFS-based system to MMIO-based communication, preserving all functionality while achieving 10-100x performance improvement.

**Approach:** Adapt existing code rather than rewriting from scratch.

---

## 1. Migration Strategy

### Three-Phase Approach

```
Phase 1: Enhance vGPU Stub        Phase 2: Update MEDIATOR         Phase 3: Update VM Client
    (2-3 days)                        (3-4 days)                       (2-3 days)
         ↓                                   ↓                                ↓
┌─────────────────────┐          ┌─────────────────────┐          ┌─────────────────────┐
│ vgpu-stub.c         │          │ mediator_mmio.c     │          │ vm_client_mmio.c    │
│ • Extended registers│   -->    │ • Socket listener   │   -->    │ • MMIO write/read   │
│ • Request/response  │          │ • Binary parser     │          │ • Doorbell ring     │
│ • Doorbell handler  │          │ • Keep CUDA logic   │          │ • Keep properties   │
│ • Socket to mediator│          │ • Keep queue logic  │          │ • Remove NFS        │
└─────────────────────┘          └─────────────────────┘          └─────────────────────┘
         ↓                                   ↓                                ↓
    Test with dummy                Test with socket                 Test end-to-end
    register reads                  communication                     CUDA execution
```

---

## 2. Code Mapping

### 2.1 VM Client Migration

**Current (vm_client_vector.c)** → **Target (vm_client_mmio.c)**

| Function | Current Implementation | New Implementation | Change Level |
|----------|----------------------|-------------------|--------------|
| `find_vgpu_device()` | Lines 54-150 | **Keep unchanged** | ✅ None |
| `get_vgpu_properties()` | Lines 156-194 | **Keep unchanged** | ✅ None |
| `send_cuda_request()` | Lines 199-237 (NFS write) | **Replace with MMIO** | 🔄 Major |
| `wait_for_response()` | Lines 243-284 (NFS poll) | **Replace with MMIO** | 🔄 Major |

**Migration Steps:**

#### Step 1: Keep Property Reading (Lines 156-194)
```c
// NO CHANGES NEEDED - Already uses MMIO
int get_vgpu_properties(VGPUProperties *props) {
    // ... existing code ...
    props->pool_id = (char)mmio[0x008/4];
    props->priority = mmio[0x00C/4];
    props->vm_id = mmio[0x010/4];
    // ... existing code ...
}
```

#### Step 2: Replace send_cuda_request() - NFS → MMIO
```c
// OLD (vm_client_vector.c:199-237)
int send_cuda_request(...) {
    // Format request string
    snprintf(request_data, sizeof(request_data),
             "%c:%u:%u:%d:%d", pool_id, priority, vm_id, num1, num2);
    
    // Write to NFS file
    fp = fopen("/mnt/vgpu/vmX/request.txt", "w");
    fprintf(fp, "%s\n", request_data);
    fflush(fp);
    fsync(fileno(fp));
    fclose(fp);
}

// NEW (vm_client_mmio.c)
int send_cuda_request(volatile uint32_t *mmio, int num1, int num2) {
    // Wait for device idle
    while (mmio[0x004/4] != 0) { /* STATUS != IDLE */ }
    
    // Build binary request
    struct vgpu_request req = {
        .version = 0x00010000,
        .opcode = 0x0001,  // CUDA_KERNEL
        .param_count = 2,
        .params = {num1, num2}
    };
    
    // Write request to MMIO buffer (0x040-0x43F)
    uint32_t *req_buf = (uint32_t*)(mmio + 0x040/4);
    memcpy(req_buf, &req, sizeof(req));
    
    // Set request length
    mmio[0x018/4] = sizeof(req);
    
    // Ring doorbell
    mmio[0x000/4] = 1;
    
    return 0;
}
```

#### Step 3: Replace wait_for_response() - NFS Poll → MMIO Poll
```c
// OLD (vm_client_vector.c:243-284)
int wait_for_response(uint32_t vm_id, int *result) {
    // Poll NFS file
    while (elapsed < RESPONSE_TIMEOUT) {
        fp = fopen("/mnt/vgpu/vmX/response.txt", "r");
        if (fp && fgets(line, sizeof(line), fp)) {
            sscanf(line, "%d", result);
            // Found response
            return 0;
        }
        usleep(POLL_INTERVAL);
    }
}

// NEW (vm_client_mmio.c)
int wait_for_response(volatile uint32_t *mmio, int *result, 
                      uint32_t timeout_sec) {
    struct timespec start, now;
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Poll STATUS register
    while (1) {
        uint32_t status = mmio[0x004/4];
        
        if (status == 0x02) {  // DONE
            // Read response length
            uint32_t resp_len = mmio[0x01C/4];
            
            // Read response from MMIO buffer (0x440-0x83F)
            uint32_t *resp_buf = (uint32_t*)(mmio + 0x440/4);
            struct vgpu_response resp;
            memcpy(&resp, resp_buf, resp_len);
            
            // Extract result
            if (resp.result_count > 0) {
                *result = resp.results[0];
            }
            
            return 0;
        }
        else if (status == 0x03) {  // ERROR
            uint32_t error_code = mmio[0x014/4];
            fprintf(stderr, "CUDA error: %u\n", error_code);
            return -1;
        }
        
        // Check timeout
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - start.tv_sec) + 
                        (now.tv_nsec - start.tv_nsec) / 1e9;
        if (elapsed >= timeout_sec) {
            return -1;  // Timeout
        }
        
        usleep(10000);  // 10ms (100x faster than NFS polling!)
    }
}
```

#### Step 4: Update main() Function
```c
// OLD
int main(int argc, char *argv[]) {
    // ... get properties from MMIO ...
    // Send via NFS
    send_cuda_request(vm_id, pool_id, priority, num1, num2);
    // Wait via NFS
    wait_for_response(vm_id, &result);
}

// NEW
int main(int argc, char *argv[]) {
    // Map MMIO (existing code, lines 156-194)
    int fd = open(device_path, O_RDWR | O_SYNC);
    volatile uint32_t *mmio = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd, 0);
    
    // Get properties (existing code - unchanged)
    VGPUProperties props;
    get_vgpu_properties(mmio, &props);
    
    // Send via MMIO (new)
    send_cuda_request(mmio, num1, num2);
    
    // Wait via MMIO (new)
    wait_for_response(mmio, &result, 30);
    
    // Cleanup
    munmap((void*)mmio, 4096);
    close(fd);
}
```

**Summary:**
- **Keep:** PCI scanning, MMIO mapping, property reading (50%)
- **Replace:** Request sending, response polling (50%)
- **Result:** ~300 lines, 60% reused code

---

### 2.2 MEDIATOR Migration

**Current (mediator_async.c)** → **Target (mediator_mmio.c)**

| Component | Current Implementation | New Implementation | Change Level |
|-----------|----------------------|-------------------|--------------|
| Queue structures | Lines 43-66 | **Keep unchanged** | ✅ None |
| `enqueue_request()` | Lines 104-155 | **Keep unchanged** | ✅ None |
| `dequeue_request()` | Lines 160-175 | **Keep unchanged** | ✅ None |
| `process_request()` | Lines 254-279 | **Keep unchanged** | ✅ None |
| `cuda_result_callback()` | Lines 196-249 (NFS write) | **Adapt for socket** | 🔄 Moderate |
| `poll_requests()` | Lines 316-421 (NFS poll) | **Replace with socket** | 🔄 Major |
| CUDA integration | Uses cuda_vector_add.c | **Keep unchanged** | ✅ None |

**Migration Steps:**

#### Step 1: Keep Queue Logic (Lines 43-175)
```c
// NO CHANGES NEEDED - Queue logic is perfect
typedef struct Request { ... } Request;
typedef struct MediatorState { ... } MediatorState;

void enqueue_request(MediatorState *state, Request *new_req) {
    // ... existing priority sorting logic ...
}

Request* dequeue_request(MediatorState *state) {
    // ... existing dequeue logic ...
}
```

#### Step 2: Replace poll_requests() - File Polling → Socket Listening
```c
// OLD (mediator_async.c:316-421)
void poll_requests(MediatorState *state) {
    DIR *dir = opendir("/var/vgpu");
    
    while ((entry = readdir(dir)) != NULL) {
        // Read request.txt
        fp = fopen(request_file, "r");
        fgets(line, sizeof(line), fp);
        
        // Parse: "pool_id:priority:vm_id:num1:num2"
        parse_request(line, &pool_id, &priority, &vm_id, &num1, &num2);
        
        // Create and enqueue request
        Request *req = malloc(sizeof(Request));
        req->pool_id = pool_id;
        req->priority = priority;
        req->vm_id = vm_id;
        req->num1 = num1;
        req->num2 = num2;
        enqueue_request(state, req);
    }
}

// NEW (mediator_mmio.c)
void socket_listener(MediatorState *state, int sock_fd) {
    // Accept connections from vgpu-stub devices
    int client_fd = accept(sock_fd, NULL, NULL);
    
    while (1) {
        // Read binary request packet
        struct {
            uint32_t vm_id;
            uint32_t request_id;
            uint32_t length;
            char pool_id;
            uint8_t priority;
            uint8_t reserved[2];
            uint8_t data[1024];
        } packet;
        
        ssize_t n = recv(client_fd, &packet, sizeof(packet), 0);
        if (n <= 0) break;
        
        // Parse binary request
        struct vgpu_request *req_data = (struct vgpu_request*)packet.data;
        
        // Create request (SAME structure as before)
        Request *req = malloc(sizeof(Request));
        req->pool_id = packet.pool_id;
        req->priority = packet.priority;
        req->vm_id = packet.vm_id;
        req->num1 = req_data->params[0];
        req->num2 = req_data->params[1];
        req->timestamp = time(NULL);
        
        // Enqueue (SAME function as before)
        enqueue_request(state, req);
    }
}
```

#### Step 3: Adapt cuda_result_callback() - File Write → Socket Write
```c
// OLD (mediator_async.c:196-249)
void cuda_result_callback(int result, void *user_data) {
    Request *req = (Request *)user_data;
    
    // Write to NFS file
    snprintf(response_file, sizeof(response_file), 
             "/var/vgpu/vm%u/response.txt", req->vm_id);
    FILE *fp = fopen(response_file, "w");
    fprintf(fp, "%d\n", result);
    fflush(fp);
    fsync(fileno(fp));
    fclose(fp);
    
    // ... statistics and cleanup ...
}

// NEW (mediator_mmio.c)
void cuda_result_callback(int result, void *user_data) {
    Request *req = (Request *)user_data;
    MediatorState *state = &g_state;
    
    // Build binary response
    struct vgpu_response resp = {
        .version = 0x00010000,
        .status = 0,  // Success
        .result_count = 1,
        .results = {result}
    };
    
    // Send response via socket (back to vgpu-stub)
    // vgpu-stub will write to MMIO response buffer
    struct {
        uint32_t vm_id;
        uint32_t request_id;
        uint32_t length;
        uint8_t data[1024];
    } response_packet;
    
    response_packet.vm_id = req->vm_id;
    response_packet.request_id = req->request_id;
    response_packet.length = sizeof(resp);
    memcpy(response_packet.data, &resp, sizeof(resp));
    
    send(req->socket_fd, &response_packet, 
         sizeof(response_packet) - 1024 + sizeof(resp), 0);
    
    // ... statistics and cleanup (SAME as before) ...
    state->total_processed++;
    if (req->pool_id == 'A') state->pool_a_processed++;
    else state->pool_b_processed++;
    
    state->cuda_busy = 0;
    free(req);
    
    // Process next request (SAME as before)
    Request *next_req = dequeue_request(state);
    if (next_req) {
        process_request(state, next_req);
    }
}
```

#### Step 4: Keep CUDA Integration
```c
// NO CHANGES NEEDED
void process_request(MediatorState *state, Request *req) {
    state->cuda_busy = 1;
    
    // Call CUDA (SAME as before)
    cuda_vector_add_async(req->num1, req->num2, 
                         cuda_result_callback, req);
}
```

**Summary:**
- **Keep:** Queue logic, CUDA integration, statistics (70%)
- **Replace:** Input (file → socket), Output (file → socket) (30%)
- **Result:** ~450 lines, 70% reused code

---

### 2.3 vGPU Stub Enhancement

**Current (complete.txt vgpu-stub.c)** → **Target (vgpu-stub-enhanced.c)**

This is new code (not a migration), but informed by existing protocol.

**Key additions (as per VGPU_STUB_CHANGES.md):**
1. Extended register map (16 registers)
2. Request buffer (1KB at 0x040)
3. Response buffer (1KB at 0x440)
4. Doorbell handler
5. Socket connection to mediator

See VGPU_STUB_CHANGES.md for complete details.

---

## 3. Protocol Mapping

### 3.1 Request Translation

**NFS Protocol (ASCII text):**
```
Format: "pool_id:priority:vm_id:num1:num2\n"
Example: "A:2:1:100:200\n"
Size: ~15 bytes
```

**MMIO Protocol (Binary):**
```c
struct vgpu_request {
    uint32_t version;       // 0x00010000 = v1.0
    uint32_t opcode;        // 0x0001 = CUDA_KERNEL
    uint32_t flags;         // 0
    uint32_t param_count;   // 2
    uint32_t data_offset;   // sizeof(header)
    uint32_t data_length;   // 0
    uint32_t reserved[2];   // 0
    uint32_t params[2];     // [100, 200]
};
// Size: 40 bytes
```

**Conversion Logic:**
```c
// NFS → MMIO (in VM client)
// OLD: sprintf(buf, "%c:%u:%u:%d:%d", pool_id, priority, vm_id, num1, num2);
// NEW:
struct vgpu_request req = {
    .version = 0x00010000,
    .opcode = 0x0001,
    .flags = 0,
    .param_count = 2,
    .data_offset = sizeof(struct vgpu_request) - 8,  // Before params
    .data_length = 0,
    .reserved = {0, 0},
    .params = {num1, num2}
};
```

---

### 3.2 Response Translation

**NFS Protocol (ASCII text):**
```
Format: "result\n"
Example: "300\n"
Size: ~4 bytes
```

**MMIO Protocol (Binary):**
```c
struct vgpu_response {
    uint32_t version;       // 0x00010000
    uint32_t status;        // 0 = success
    uint32_t result_count;  // 1
    uint32_t data_offset;   // sizeof(header)
    uint32_t data_length;   // 0
    uint32_t exec_time_us;  // CUDA execution time
    uint32_t reserved[2];   // 0
    uint32_t results[1];    // [300]
};
// Size: 36 bytes
```

**Conversion Logic:**
```c
// MEDIATOR → vgpu-stub → VM (binary)
// OLD: fprintf(fp, "%d\n", result);
// NEW:
struct vgpu_response resp = {
    .version = 0x00010000,
    .status = 0,
    .result_count = 1,
    .data_offset = sizeof(struct vgpu_response) - 4,  // Before results
    .data_length = 0,
    .exec_time_us = execution_time,
    .reserved = {0, 0},
    .results = {result}
};
```

---

## 4. Performance Comparison

### Latency Breakdown

| Operation | NFS (Current) | MMIO (Target) | Improvement |
|-----------|--------------|---------------|-------------|
| VM → Host communication | 1-10 ms (file write) | 0.1-0.5 ms (MMIO write) | **10-20x** |
| Host polling/notification | 0-1000 ms (poll interval) | 0.001-0.01 ms (doorbell) | **100-1000x** |
| Parse request | 0.05 ms (ASCII) | 0.001 ms (binary) | **50x** |
| CUDA execution | 1-5 ms (same) | 1-5 ms (same) | 1x |
| Host → VM response | 1-10 ms (file write) | 0.1-0.5 ms (MMIO write) | **10-20x** |
| VM polling | 0-100 ms (poll interval) | 0-10 ms (poll interval) | **10x** |
| **TOTAL (typical)** | **500-1100 ms** | **2-22 ms** | **50-500x** |
| **TOTAL (best case)** | **3-26 ms** | **2-11 ms** | **2-10x** |

**Expected Overall Improvement: 10-100x faster**

---

## 5. Migration Timeline

### Week 1: Infrastructure (Days 1-5)

**Day 1-2: Enhance vGPU Stub**
- [ ] Modify vgpu-stub.c per VGPU_STUB_CHANGES.md
- [ ] Add extended registers
- [ ] Add request/response buffers
- [ ] Add doorbell handler (dummy for now)
- [ ] Build and install QEMU
- [ ] Test: Read new registers from guest

**Day 3: Add Socket Interface to vGPU Stub**
- [ ] Add Unix socket creation in vgpu-stub.c
- [ ] Add socket connection logic
- [ ] Test: Socket can be created and connected

**Day 4-5: Test Infrastructure**
- [ ] Create test VM
- [ ] Verify all registers accessible
- [ ] Verify doorbell can be rung
- [ ] Verify status changes

---

### Week 2: MEDIATOR & Client (Days 6-10)

**Day 6-7: Adapt MEDIATOR**
- [ ] Copy mediator_async.c → mediator_mmio.c
- [ ] Keep queue logic unchanged
- [ ] Replace poll_requests() with socket listener
- [ ] Adapt cuda_result_callback() for socket
- [ ] Build and test socket communication

**Day 8-9: Adapt VM Client**
- [ ] Copy vm_client_vector.c → vm_client_mmio.c
- [ ] Keep PCI scanning and property reading
- [ ] Replace send_cuda_request() with MMIO version
- [ ] Replace wait_for_response() with MMIO version
- [ ] Build and deploy to test VM

**Day 10: Integration Testing**
- [ ] Test single VM end-to-end
- [ ] Test request/response via MMIO
- [ ] Verify CUDA execution works
- [ ] Compare latency vs NFS

---

### Week 3: Multi-VM & Validation (Days 11-15)

**Day 11-12: Multi-VM Testing**
- [ ] Deploy to 3 VMs (different priorities)
- [ ] Test priority scheduling
- [ ] Test concurrent requests
- [ ] Verify queue ordering

**Day 13: Performance Testing**
- [ ] Benchmark latency (NFS vs MMIO)
- [ ] Benchmark throughput
- [ ] Test under load
- [ ] Collect statistics

**Day 14: Documentation**
- [ ] Update all docs
- [ ] Create migration guide
- [ ] Document new protocol
- [ ] Create troubleshooting guide

**Day 15: Validation & Cleanup**
- [ ] Final testing with 7 VMs
- [ ] Verify all features working
- [ ] Code cleanup
- [ ] Review and polish

---

## 6. Testing Strategy

### Phase 1: Component Testing

**Test 1.1: vGPU Stub Registers**
```bash
# In guest VM
sudo ./test_mmio_registers
# Expected: All 16 registers readable, correct values
```

**Test 1.2: Doorbell Mechanism**
```bash
# In guest VM
sudo ./test_doorbell
# Expected: STATUS changes IDLE → BUSY → (ERROR if no mediator)
```

---

### Phase 2: Integration Testing

**Test 2.1: Single VM, Simple Request**
```bash
# Start mediator on host
sudo ./mediator_mmio

# In VM
sudo ./vm_client_mmio 100 200
# Expected: Result = 300, latency < 50ms
```

**Test 2.2: Priority Ordering**
```bash
# Send 3 requests with different priorities
# Expected: High processed first, then medium, then low
```

---

### Phase 3: Performance Testing

**Test 3.1: Latency Comparison**
```bash
# NFS version
time ./vm_client_vector 100 200
# MMIO version
time ./vm_client_mmio 100 200
# Expected: 10-100x faster
```

**Test 3.2: Throughput Test**
```bash
# 100 requests
for i in {1..100}; do ./vm_client_mmio $i $((i+100)); done
# Measure total time and requests/second
```

---

## 7. Rollback Plan

### If Migration Fails

**Option 1: Keep NFS as Fallback**
- Compile both versions
- VM can choose which to use
- Gradual migration

**Option 2: Complete Rollback**
1. Restore original QEMU
2. Keep using NFS-based system
3. Re-evaluate approach

**Backup Steps:**
```bash
# Before starting
cp /usr/lib64/xen/bin/qemu-system-i386 qemu-system-i386.backup
cp /usr/lib64/xen/bin/qemu-wrapper qemu-wrapper.backup
tar czf step2_test_backup.tar.gz /home/david/Downloads/gpu/step2_test/

# To rollback
cp qemu-system-i386.backup /usr/lib64/xen/bin/qemu-system-i386
cp qemu-wrapper.backup /usr/lib64/xen/bin/qemu-wrapper
# Restart VMs
```

---

## 8. Success Criteria

### Phase 1 Success
- [ ] Enhanced vGPU stub compiles without errors
- [ ] All 16 registers readable from guest
- [ ] Doorbell triggers status change
- [ ] No regressions in existing VMs

### Phase 2 Success
- [ ] Single VM can send request via MMIO
- [ ] MEDIATOR receives request via socket
- [ ] CUDA executes and returns result
- [ ] VM receives response via MMIO
- [ ] Latency < 50ms (vs 500-1000ms with NFS)

### Phase 3 Success
- [ ] 3+ VMs work simultaneously
- [ ] Priority scheduling correct
- [ ] 10-100x performance improvement measured
- [ ] All existing features preserved
- [ ] Ready for 7-VM deployment

---

## 9. Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| QEMU build fails | Low | High | Use working base from complete.txt |
| Socket communication issues | Medium | Medium | Test incrementally with simple echo server |
| MMIO timing issues | Low | Low | Use same polling as NFS initially |
| Data corruption | Low | High | Extensive validation, CRC checks |
| Performance not as expected | Low | Medium | Profile and optimize, keep NFS fallback |
| VM compatibility issues | Medium | Medium | Test on multiple VM types |

---

## 10. Next Action Items

### Immediate (This Week)
1. **Review this migration plan** - Confirm approach
2. **Backup current system** - Tar up step2_test/
3. **Start Phase 1** - Enhance vGPU stub
4. **Create test environment** - Prepare test VM

### Short Term (Next 2 Weeks)
5. **Implement MMIO communication** - Follow timeline
6. **Test incrementally** - Each component separately
7. **Integrate components** - End-to-end testing
8. **Measure performance** - Compare to NFS baseline

### Medium Term (Weeks 3-4)
9. **Multi-VM deployment** - Scale to 7 VMs
10. **Production testing** - Real workloads
11. **Documentation** - Complete all docs
12. **Handover** - System ready for use

---

## Conclusion

**You have everything needed to succeed:**

✅ **Working reference implementation** (NFS-based)  
✅ **Proven protocol** (request/response format)  
✅ **Solid CUDA integration** (async execution)  
✅ **Tested queue logic** (priority scheduling)  
✅ **Complete documentation** (architecture, guides)  
✅ **Clear migration path** (this document)

**Recommendation:** Proceed with Phase 1 (vGPU Stub enhancement) immediately.

**Expected Outcome:** 10-100x performance improvement while preserving all functionality.

**Status:** ✅ Ready to begin migration
