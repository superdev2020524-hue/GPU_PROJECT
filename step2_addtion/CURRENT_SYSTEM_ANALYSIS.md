# Current System Analysis - NFS-Based Implementation

**Date:** February 12, 2026  
**Source:** `/home/david/Downloads/gpu/step2_test/`  
**Status:** ✅ Working system with verified code

---

## Executive Summary

You have a **fully functional GPU sharing system** using NFS-based communication. This document analyzes your current implementation to inform the transition to MMIO-based communication.

---

## 1. Current Architecture Overview

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    VM Layer (Guest OS)                       │
├─────────────────────────────────────────────────────────────┤
│  vm_client_vector.c (391 lines)                            │
│  ├─ Reads vGPU properties from MMIO:                        │
│  │  • pool_id from register 0x008 ('A' or 'B')            │
│  │  • priority from register 0x00C (0/1/2)                 │
│  │  • vm_id from register 0x010                            │
│  ├─ Formats request: "pool_id:priority:vm_id:num1:num2"   │
│  ├─ Writes to: /mnt/vgpu/vm<id>/request.txt               │
│  └─ Polls: /mnt/vgpu/vm<id>/response.txt                  │
└─────────────────────────────────────────────────────────────┘
                              ↓
                      NFS Communication
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    NFS Layer (Shared FS)                     │
├─────────────────────────────────────────────────────────────┤
│  Dom0 exports: /var/vgpu                                    │
│  VM mounts: /mnt/vgpu                                       │
│                                                             │
│  /var/vgpu/                                                 │
│  ├── vm1/                                                   │
│  │   ├── request.txt  ← VM writes, MEDIATOR reads         │
│  │   └── response.txt ← MEDIATOR writes, VM reads         │
│  ├── vm2/                                                   │
│  └── ... (vm3-vm7)                                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
                      File Polling
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                MEDIATOR Daemon (Dom0)                       │
├─────────────────────────────────────────────────────────────┤
│  mediator_async.c (535 lines)                              │
│  ├─ Polls /var/vgpu/vm*/request.txt every 1 second        │
│  ├─ Parses: "pool_id:priority:vm_id:num1:num2"            │
│  ├─ Single priority queue (spans Pool A + Pool B)          │
│  │  Priority order: High(2) → Medium(1) → Low(0)          │
│  │  FIFO within same priority                              │
│  ├─ Sends to CUDA asynchronously                           │
│  ├─ Callback writes to: /var/vgpu/vm<id>/response.txt     │
│  └─ Clears request file after processing                   │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    Async CUDA Execution
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                    CUDA Layer                                │
├─────────────────────────────────────────────────────────────┤
│  cuda_vector_add.c (349 lines)                             │
│  ├─ Asynchronous kernel execution                          │
│  ├─ Thread-safe operation                                  │
│  ├─ Callback mechanism: cuda_result_callback()             │
│  └─ Returns result to MEDIATOR                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Current Protocol Specification

### 2.1 Request Format (ASCII Text)

**File:** `/mnt/vgpu/vm<id>/request.txt` (VM writes)

**Format:**
```
pool_id:priority:vm_id:num1:num2\n
```

**Example:**
```
A:2:1:100:200
```

**Parsing (mediator_async.c:285-312):**
```c
sscanf(line, "%c:%u:%u:%d:%d", &pool, &prio, &vm, &n1, &n2)
```

**Validation:**
- `pool_id`: Must be 'A' or 'B'
- `priority`: Must be 0 (low), 1 (medium), or 2 (high)
- `vm_id`: 1-20
- `num1`, `num2`: Any integer

---

### 2.2 Response Format (ASCII Text)

**File:** `/var/vgpu/vm<id>/response.txt` (MEDIATOR writes)

**Format:**
```
result\n
```

**Example:**
```
300
```

**Writing (mediator_async.c:203-217):**
```c
FILE *fp = fopen(response_file, "w");
fprintf(fp, "%d\n", result);
fflush(fp);
fsync(fileno(fp));  // Force NFS sync
fclose(fp);
```

---

### 2.3 Communication Flow

```
1. VM: Read properties from vGPU MMIO registers
   └─ pool_id = mmio[0x008/4]  // 'A' or 'B'
   └─ priority = mmio[0x00C/4]  // 0, 1, or 2
   └─ vm_id = mmio[0x010/4]     // VM identifier

2. VM: Format request string
   └─ sprintf(buf, "%c:%u:%u:%d:%d", pool_id, priority, vm_id, num1, num2)

3. VM: Write to NFS
   └─ write to /mnt/vgpu/vm<id>/request.txt
   └─ fflush + fsync for NFS synchronization

4. MEDIATOR: Poll files (every 1 second)
   └─ opendir("/var/vgpu")
   └─ for each vm*/request.txt:
      └─ if file not empty: parse and enqueue

5. MEDIATOR: Clear request file immediately
   └─ fopen(request_file, "w") + fclose()  // Truncate

6. MEDIATOR: Process queue
   └─ Dequeue highest priority request
   └─ Send to CUDA asynchronously

7. CUDA: Execute kernel (takes ~milliseconds)
   └─ Call callback with result

8. MEDIATOR: Write response
   └─ fprintf to /var/vgpu/vm<id>/response.txt
   └─ fflush + fsync

9. VM: Poll for response (every 0.1 seconds)
   └─ while (timeout < 30s):
      └─ if response.txt not empty: read result

10. VM: Clear response file after reading
    └─ fopen(response_file, "w") + fclose()
```

---

## 3. Current Implementation Details

### 3.1 VM Client (vm_client_vector.c)

**Key Functions:**
```c
// Line 54-150: Find vGPU device by scanning PCI
char* find_vgpu_device(void)
  └─ Scans /sys/bus/pci/devices/
  └─ Matches vendor=0x1af4, device=0x1111, class=0x120000
  └─ Returns path to resource0

// Line 156-194: Read vGPU properties from MMIO
int get_vgpu_properties(VGPUProperties *props)
  └─ Maps resource0 to memory
  └─ Reads pool_id from offset 0x008
  └─ Reads priority from offset 0x00C
  └─ Reads vm_id from offset 0x010

// Line 199-237: Send request via NFS
int send_cuda_request(uint32_t vm_id, char pool_id, uint32_t priority, 
                      int num1, int num2)
  └─ Formats: "pool_id:priority:vm_id:num1:num2"
  └─ Writes to /mnt/vgpu/vm<id>/request.txt
  └─ Uses fflush + fsync for NFS sync

// Line 243-284: Wait for response
int wait_for_response(uint32_t vm_id, int *result)
  └─ Polls /mnt/vgpu/vm<id>/response.txt
  └─ 30 second timeout
  └─ 0.1 second poll interval
```

**Execution:**
```bash
sudo ./vm_client_vector 100 200
# Output:
# [SCAN] Found vGPU stub at 0000:00:06.0
# [PROPERTIES] Pool ID: A, Priority: 2, VM ID: 1
# [REQUEST] Sending: A:2:1:100:200
# [POLLING] Waiting for response...
# [SUCCESS] Result: 300
```

---

### 3.2 MEDIATOR Daemon (mediator_async.c)

**Key Data Structures:**
```c
// Line 43-51: Request structure
typedef struct Request {
    char pool_id;           // 'A' or 'B' (metadata only)
    uint32_t priority;      // 2=high, 1=medium, 0=low
    uint32_t vm_id;
    int num1, num2;
    time_t timestamp;       // For FIFO ordering
    void *user_data;
    struct Request *next;   // Linked list
} Request;

// Line 57-66: Mediator state
typedef struct {
    Request *queue_head;     // Single priority queue
    pthread_mutex_t queue_lock;
    int running;
    int cuda_busy;
    Request *current_request;
    uint64_t total_processed;
    uint64_t pool_a_processed;
    uint64_t pool_b_processed;
} MediatorState;
```

**Key Functions:**
```c
// Line 104-155: Enqueue with priority sorting
void enqueue_request(MediatorState *state, Request *new_req)
  └─ Insert sorted by: priority DESC, timestamp ASC
  └─ Higher priority = earlier in queue
  └─ Same priority = FIFO by timestamp

// Line 160-175: Dequeue highest priority
Request* dequeue_request(MediatorState *state)
  └─ Return head of queue (highest priority, earliest timestamp)

// Line 196-249: CUDA result callback
void cuda_result_callback(int result, void *user_data)
  └─ Write to /var/vgpu/vm<id>/response.txt
  └─ Clear /var/vgpu/vm<id>/request.txt
  └─ Update statistics
  └─ Mark CUDA idle
  └─ Process next request

// Line 254-279: Process request (send to CUDA)
void process_request(MediatorState *state, Request *req)
  └─ Mark CUDA busy
  └─ Call cuda_vector_add_async(num1, num2, callback, req)

// Line 316-421: Poll for new requests
void poll_requests(MediatorState *state)
  └─ opendir("/var/vgpu")
  └─ for each vm<id>/:
      └─ Clear old response.txt
      └─ Read request.txt
      └─ Parse and enqueue
      └─ Clear request.txt

// Line 456-493: Main loop
void run_mediator(MediatorState *state)
  └─ while (running):
      └─ poll_requests()
      └─ process_queue()
      └─ sleep(1 second)
```

**Execution:**
```bash
sudo ./mediator_async
# Output:
# [MEDIATOR] Initialized
# [MEDIATOR] Starting main loop...
# [MEDIATOR] Polling /var/vgpu every 1 seconds
# [ENQUEUE] Pool A: vm=1, prio=2, 100+200
# [PROCESS] Pool A: vm=1, prio=2, 100+200
# [RESULT] Pool A: vm=1, result=300
# [RESPONSE] Sent to vm1: 300
```

---

### 3.3 CUDA Implementation (cuda_vector_add.c)

**Key Functions:**
```c
// Line 60-65: CUDA kernel
__global__ void vector_add_kernel(int *a, int *b, int *c)
  └─ c[0] = a[0] + b[0]  // Simple addition

// Line 70-160: Async worker thread
void* async_worker(void *arg)
  └─ Allocate GPU memory
  └─ Copy to GPU
  └─ Launch kernel
  └─ Copy result back
  └─ Call callback
  └─ Free GPU memory

// Line 165-190: Public async API
int cuda_vector_add_async(int num1, int num2, 
                         cuda_callback_t callback, void *user_data)
  └─ Create AsyncContext
  └─ pthread_create(async_worker)
  └─ Return immediately (non-blocking)
```

---

## 4. Performance Characteristics

### 4.1 Latency Breakdown (Estimated)

| Component | Latency | Notes |
|-----------|---------|-------|
| VM → NFS write | 1-10 ms | File write + sync |
| MEDIATOR poll interval | 0-1000 ms | Worst case 1 second |
| MEDIATOR parse/enqueue | <1 ms | In-memory operations |
| CUDA execution | 1-5 ms | GPU kernel |
| MEDIATOR → NFS write | 1-10 ms | File write + sync |
| VM poll interval | 0-100 ms | Worst case 0.1 second |
| **Total (typical)** | **500-1100 ms** | **Dominated by polling** |
| **Total (best case)** | **3-26 ms** | **No polling wait** |

**Bottleneck:** MEDIATOR polling interval (1 second) causes significant latency.

---

### 4.2 Throughput Analysis

**Single VM throughput:**
- Limited by round-trip latency: ~1-2 requests/second

**Multi-VM throughput:**
- Queue allows batching of requests
- CUDA processing is fast (~1-5 ms per request)
- Bottleneck is NFS polling, not CUDA

**Optimization potential:**
- Reduce polling interval (but increases CPU usage)
- Use inotify for file watching (still slower than MMIO)
- **Best solution:** MMIO-based communication (10-100x faster)

---

## 5. Testing Infrastructure

### 5.1 Test Client (test_mediator_client.c)

**Features:**
- Simulates multiple VMs
- Two test modes:
  - **Simultaneous:** All VMs submit at once
  - **Sequential:** VMs submit with delays
- Real-time visualization
- Statistics and timing

**Example usage:**
```bash
# Test 3 VMs simultaneously
./test_mediator_client simultaneous --vms "1:A:2:100:200,2:A:1:150:250,3:B:2:50:75"

# Test sequential arrivals
./test_mediator_client sequential --vms "1:A:2,2:A:2,3:A:2" --delay 0.5
```

---

## 6. Strengths of Current System

✅ **Proven and Working**
- All code compiled and tested
- Clear separation of concerns
- Robust error handling

✅ **Simple Protocol**
- ASCII text format (easy to debug)
- Human-readable request/response files
- Standard file I/O

✅ **Priority Scheduling**
- Single queue with priority sorting
- FIFO within same priority
- Works across both pools

✅ **Asynchronous CUDA**
- Non-blocking execution
- Callback mechanism
- Thread-safe

✅ **Comprehensive Testing**
- Test client with multiple modes
- Visualization tools
- Statistics tracking

---

## 7. Limitations of Current System

❌ **High Latency**
- NFS file I/O: 1-10 ms per operation
- Polling interval: up to 1 second delay
- Total latency: 500-1100 ms typical

❌ **CPU Overhead**
- Continuous polling (every 1 second)
- Directory scanning
- File open/close/read operations

❌ **Scalability**
- More VMs = more files to poll
- NFS server load increases
- File system overhead

❌ **Not Real Hardware-Like**
- Real GPUs use MMIO, not files
- Doesn't match actual PCI device behavior
- Not suitable for CloudStack integration

❌ **Reliability**
- NFS can have sync issues
- File locking complications
- Race conditions possible

---

## 8. Transition Path to MMIO

### What Can Be Reused?

✅ **Keep Unchanged:**
1. **CUDA Implementation** (cuda_vector_add.c)
   - Same async execution
   - Same callback mechanism
   - No changes needed

2. **Priority Queue** (mediator_async.c: enqueue/dequeue)
   - Same sorting logic
   - Same FIFO within priority
   - Same data structures

3. **Request Parsing Logic**
   - Same validation
   - Same protocol (but binary instead of ASCII)

4. **Statistics and Logging**
   - Same counters
   - Same display format

✅ **Adapt:**
1. **MEDIATOR Input** (mediator_async.c: poll_requests)
   - Replace file polling → socket listening
   - Replace file parsing → binary parsing
   - Same enqueue after parsing

2. **MEDIATOR Output** (mediator_async.c: cuda_result_callback)
   - Replace file write → socket write
   - Binary response instead of ASCII

3. **VM Client** (vm_client_vector.c: send/wait)
   - Replace file write → MMIO write
   - Replace file poll → MMIO poll
   - Same property reading (already uses MMIO)

❌ **Remove:**
1. NFS setup and configuration
2. File polling loop
3. Directory scanning
4. File sync operations

---

## 9. Key Insights for MMIO Design

### Protocol Compatibility

**Current ASCII:**
```
Request:  "A:2:1:100:200\n"
Response: "300\n"
```

**Future Binary (MMIO):**
```c
struct vgpu_request {
    uint32_t version;      // 0x00010000
    uint32_t opcode;       // 0x0001 (CUDA_KERNEL)
    uint32_t flags;        // 0
    uint32_t param_count;  // 2
    uint32_t params[2];    // [100, 200]
};

struct vgpu_response {
    uint32_t version;      // 0x00010000
    uint32_t status;       // 0 = success
    uint32_t result_count; // 1
    uint32_t results[1];   // [300]
};
```

**Advantages:**
- Smaller size (32 bytes vs ~15 bytes text, but with version info)
- Type-safe
- No parsing overhead
- Extensible

---

## 10. Files to Create for MMIO Transition

Based on your current code structure, here's what needs to be created:

```
step2_addtion/implementation/
├── vgpu_stub_enhanced/
│   └── vgpu-stub.c              ← Enhanced from complete.txt
│
├── host_mediator/
│   ├── mediator_mmio.c          ← Adapted from mediator_async.c
│   ├── socket_server.c          ← New: Unix socket listener
│   └── protocol_parser.c        ← New: Binary protocol parser
│
├── guest_client/
│   ├── vm_client_mmio.c         ← Adapted from vm_client_vector.c
│   └── vgpu_mmio_lib.c          ← New: MMIO helper functions
│
└── testing/
    ├── test_mmio_client.c       ← Adapted from test_mediator_client.c
    └── compare_nfs_vs_mmio.sh   ← New: Performance comparison
```

---

## 11. Summary Statistics

### Current Codebase

| Component | File | Lines | Status |
|-----------|------|-------|--------|
| VM Client | vm_client_vector.c | 391 | ✅ Working |
| MEDIATOR | mediator_async.c | 535 | ✅ Working |
| CUDA | cuda_vector_add.c | 349 | ✅ Working |
| CUDA Header | cuda_vector_add.h | 74 | ✅ Working |
| Test Client | test_mediator_client.c | 709 | ✅ Working |
| Build System | Makefile | 167 | ✅ Working |
| **Total** | **6 files** | **2,225 lines** | **100% functional** |

### Documentation

| Document | Lines | Status |
|----------|-------|--------|
| Architecture Diagram | 209 | ✅ Complete |
| NFS Setup Guide | 692 | ✅ Complete |
| Project Status | 637 | ✅ Complete |
| Implementation Plan | 328 | ✅ Complete |
| Various others | ~3,000 | ✅ Complete |
| **Total** | **~5,000 lines** | **Comprehensive** |

---

## 12. Next Steps

### Option A: Keep NFS, Enhance Features
- Add more CUDA operations
- Optimize polling
- Add more VMs
- **Pro:** Already working
- **Con:** Still slow, not production-ready

### Option B: Transition to MMIO (Recommended)
- Implement enhanced vGPU stub
- Create socket-based mediator
- Create MMIO-based client
- **Pro:** 10-100x faster, production-ready
- **Con:** Requires implementation effort

### Option C: Hybrid Approach
- Support both NFS and MMIO
- Gradual migration
- Keep NFS as fallback
- **Pro:** Risk mitigation
- **Con:** Complexity

---

## Recommendation

**Proceed with Option B: Full MMIO Transition**

**Reasoning:**
1. You have a **working reference implementation** (NFS-based)
2. Protocol is well-understood and tested
3. CUDA integration is solid
4. Priority scheduling is proven
5. MMIO will provide 10-100x performance improvement
6. Necessary for production/CloudStack integration

**Timeline:** 11-17 days (as per IMPLEMENTATION_ROADMAP.md)

**Starting Point:** Use your current code as the reference specification for the MMIO implementation.

---

**Status:** ✅ Analysis Complete - Ready to proceed with MMIO implementation
