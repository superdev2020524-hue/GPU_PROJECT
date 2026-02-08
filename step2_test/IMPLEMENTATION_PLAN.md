# IMPLEMENTATION PLAN
**Date:** 2026-02-08  
**Approach:** Three-file implementation (CUDA, MEDIATOR, VM Client)

---

## IMPLEMENTATION STRATEGY

### File Structure:
```
step2_test/
├── cuda_vector_add.c      # CUDA GPU implementation
├── mediator_async.c       # MEDIATOR daemon (async)
└── vm_client_vector.c      # VM client application
```

### Why Three Files?
1. **Separation of Concerns:** Each component has distinct responsibility
2. **Independent Testing:** Can test each component separately
3. **Clear Dependencies:** Easy to understand data flow
4. **Maintainability:** Changes to one component don't affect others

---

## FILE 1: CUDA Vector Addition (`cuda_vector_add.c`)

### Purpose:
- CUDA kernel for vector addition
- Asynchronous execution interface
- Callback mechanism for result delivery

### Key Functions:
```c
// Initialize CUDA
int cuda_init();

// Execute vector addition asynchronously
int cuda_vector_add_async(int num1, int num2, 
                          void (*callback)(int result, void *user_data),
                          void *user_data);

// Cleanup
void cuda_cleanup();
```

### Features:
- ✅ CUDA kernel: `__global__ void vector_add_kernel(int *a, int *b, int *c)`
- ✅ Asynchronous execution (non-blocking)
- ✅ Callback when result ready
- ✅ Error handling
- ✅ GPU memory management

### Dependencies:
- CUDA Toolkit
- NVIDIA GPU driver
- H100 GPU accessible

---

## FILE 2: MEDIATOR Daemon (`mediator_async.c`)

### Purpose:
- Polls NFS directory for VM requests
- Maintains single priority queue (spans Pool A + Pool B)
- Manages asynchronous CUDA execution
- Handles file I/O and initialization

### Key Functions:
```c
// Initialize mediator
int mediator_init();

// Main processing loop
void mediator_run();

// Queue management
void enqueue_request(char pool_id, int priority, int vm_id, int num1, int num2);
Request* dequeue_next_request();

// CUDA result callback
void cuda_result_callback(int result, void *user_data);

// File management
void send_response_to_vm(int vm_id, int result);
void initialize_vm_files(int vm_id);
```

### Data Structures:
```c
// Single priority queue (not per-pool)
typedef struct Request {
    char pool_id;           // 'A' or 'B' (metadata)
    uint32_t priority;      // 2=high, 1=medium, 0=low
    uint32_t vm_id;
    int num1, num2;
    time_t timestamp;
    void *user_data;        // For callback
    struct Request *next;
} Request;

typedef struct {
    Request *queue_head;    // Priority-sorted queue
    pthread_mutex_t lock;
    int cuda_busy;          // CUDA processing flag
    Request *current_request; // Currently processing
} MediatorState;
```

### Features:
- ✅ Continuous polling of `/var/vgpu/vm*/request.txt`
- ✅ Single priority queue (priority DESC, timestamp ASC)
- ✅ Asynchronous CUDA integration
- ✅ File initialization after response
- ✅ Thread-safe queue operations
- ✅ Statistics and logging

### Processing Flow:
```
1. Poll request files
2. Parse: pool_id:priority:vm_id:num1:num2
3. Insert into priority queue (sorted)
4. If CUDA idle: pop queue → send to CUDA
5. While CUDA busy: continue accepting requests
6. On CUDA result: write response → clear files → process next
```

---

## FILE 3: VM Client (`vm_client_vector.c`)

### Purpose:
- Reads vGPU properties from MMIO
- Sends vector addition request to MEDIATOR
- Waits for and displays result

### Key Functions:
```c
// Read vGPU properties
int read_vgpu_properties(char *pool_id, int *priority, int *vm_id);

// Send vector addition request
int send_vector_add_request(int num1, int num2);

// Wait for result
int wait_for_result(int *result);
```

### Request Format:
```
"pool_id:priority:vm_id:num1:num2"
Example: "A:2:1:100:200"
```

### Response Format:
```
"result"
Example: "300"
```

### Features:
- ✅ Reads pool_id, priority, vm_id from vGPU MMIO
- ✅ Formats and sends request to NFS
- ✅ Polls for response with timeout
- ✅ User-friendly output
- ✅ Error handling

---

## IMPLEMENTATION ORDER

### Phase 1: CUDA Implementation
**File:** `cuda_vector_add.c`
**Goal:** Working CUDA vector addition with async interface

**Steps:**
1. Create CUDA kernel for vector addition
2. Implement async execution wrapper
3. Add callback mechanism
4. Test with simple values

**Validation:**
- CUDA kernel compiles
- Async execution works
- Callback receives result

---

### Phase 2: MEDIATOR Implementation
**File:** `mediator_async.c`
**Goal:** Complete mediation daemon with async CUDA integration

**Steps:**
1. Implement request polling
2. Implement single priority queue (corrected understanding)
3. Integrate CUDA async interface
4. Implement file I/O and initialization
5. Add result callback handling

**Validation:**
- Polls request files correctly
- Queue maintains priority order (single queue)
- CUDA integration works asynchronously
- Files initialized after response

---

### Phase 3: VM Client Implementation
**File:** `vm_client_vector.c`
**Goal:** VM application that sends vector addition requests

**Steps:**
1. Read vGPU properties from MMIO
2. Format and send request
3. Wait for and display result
4. Error handling

**Validation:**
- Reads properties correctly
- Sends request in correct format
- Receives and displays result

---

## INTEGRATION TESTING

### Test Scenario 1: Single VM
```
VM-1 (Pool A, High) sends: 100+200
Expected: Result 300
```

### Test Scenario 2: Priority Ordering
```
VM-1 (Pool A, High) sends: 100+200
VM-4 (Pool B, High) sends: 150+250
VM-2 (Pool A, Medium) sends: 50+75

Expected order:
1. VM-1 processed first (High, earlier)
2. VM-4 processed second (High, later)
3. VM-2 processed third (Medium)
```

### Test Scenario 3: Async Processing
```
VM-1 sends request → CUDA starts processing
VM-2 sends request → Queued (CUDA busy)
VM-1 receives result → Files cleared
VM-2 request processed → CUDA starts processing
VM-2 receives result
```

### Test Scenario 4: Pool Equality
```
VM-1 (Pool A, High) sends: 100+200
VM-4 (Pool B, High) sends: 150+250

Expected: Processed in FIFO order (not pool order)
```

---

## FILE LOCATIONS

### Development Location:
```
/home/david/Downloads/gpu/step2_test/
├── cuda_vector_add.c
├── mediator_async.c
└── vm_client_vector.c
```

### Deployment Location:
```
Dom0:
  /usr/local/bin/mediator_async
  /usr/local/lib/libcuda_vector_add.so (if library)

VM:
  /usr/local/bin/vm_client_vector
```

---

## BUILD REQUIREMENTS

### CUDA File:
```bash
nvcc -o cuda_vector_add cuda_vector_add.c -lcudart
```

### MEDIATOR File:
```bash
gcc -o mediator_async mediator_async.c -lpthread -lcudart -L/usr/local/cuda/lib64
```

### VM Client File:
```bash
gcc -o vm_client_vector vm_client_vector.c
```

---

## NEXT STEPS

1. ✅ **Documentation Complete** - Priority system clarified
2. ⏳ **Implementation Plan Created** - This document
3. ⏳ **Await Confirmation** - User review of plan
4. ⏳ **Implement Phase 1** - CUDA implementation
5. ⏳ **Implement Phase 2** - MEDIATOR implementation
6. ⏳ **Implement Phase 3** - VM client implementation
7. ⏳ **Integration Testing** - End-to-end validation

---

## CONFIRMATION CHECKLIST

Before implementation, please confirm:
- [ ] Priority system understanding is correct (single queue, Pool A = Pool B)
- [ ] Three-file approach is acceptable
- [ ] Implementation order (CUDA → MEDIATOR → VM) is correct
- [ ] File locations are appropriate
- [ ] Ready to proceed with implementation

---

**Ready to implement once confirmed!**
