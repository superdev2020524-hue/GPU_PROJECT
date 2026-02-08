# IMPLEMENTATION UNDERSTANDING
**Date:** 2026-02-08  
**Purpose:** Confirm understanding of CUDA vector addition implementation

---

## SYSTEM ARCHITECTURE

### VM Assignment
```
Pool A: VM-1, VM-2, VM-3
Pool B: VM-4, VM-5, VM-6
VM-7:   Any pool, any priority (configurable)
```

### Request Format
Each VM sends:
```
pool_id:priority:vm_id:number1:number2
Example: "A:2:1:100:200"  (Pool A, high priority, VM-1, add 100+200)
```

---

## DATA FLOW DIAGRAM

```
┌─────────┐
│  VM-1   │  Sends: "A:2:1:100:200"
│ Pool A  │  ──────────────────┐
│ High    │                     │
└────┬────┘                     │
     │                          │
     │  NFS: /mnt/vgpu/vm1/     │
     │  request.txt             │
     │                          │
     └──────────────────────────┼──┐
                                │  │
┌─────────┐                     │  │
│  VM-2   │  Sends: "A:1:2:50:75"│  │
│ Pool A  │  ───────────────────┼──┼──┐
│ Medium  │                     │  │  │
└────┬────┘                     │  │  │
     │                          │  │  │
     │  NFS: /mnt/vgpu/vm2/     │  │  │
     │  request.txt             │  │  │
     │                          │  │  │
     └──────────────────────────┼──┼──┼──┐
                                │  │  │  │
                                ▼  ▼  ▼  ▼
                    ┌─────────────────────────┐
                    │   MEDIATOR DAEMON        │
                    │                         │
                    │  ┌──────────┐           │
                    │  │ Pool A   │           │
                    │  │ Queue    │           │
                    │  │ (sorted) │           │
                    │  └─────┬────┘           │
                    │        │                │
                    │  ┌──────────┐           │
                    │  │ Pool B   │           │
                    │  │ Queue    │           │
                    │  │ (sorted) │           │
                    │  └─────┬────┘           │
                    │        │                │
                    │  ┌──────────┐           │
                    │  │ CUDA     │           │
                    │  │ Executor │◄──────────┘
                    │  │ (async)  │  Next request
                    │  └─────┬────┘           │
                    │        │                │
                    │        │ Result: 300     │
                    │        │                │
                    └────────┼────────────────┘
                             │
                             │ Write to
                             │ /var/vgpu/vm1/response.txt
                             │ "300"
                             │
                             │ Initialize files:
                             │ - Clear request.txt
                             │ - Clear response.txt
                             │
                             ▼
                    ┌─────────────────────────┐
                    │   CUDA GPU (H100)       │
                    │   Vector Addition       │
                    │   number1 + number2     │
                    └─────────────────────────┘
```

---

## PROCESSING FLOW

### Step 1: VM Sends Request
```
VM-1 writes to /mnt/vgpu/vm1/request.txt:
"A:2:1:100:200"
```

### Step 2: MEDIATOR Polls and Enqueues
```
MEDIATOR reads request.txt files
Parses: pool_id='A', priority=2, vm_id=1, num1=100, num2=200
Inserts into Pool A queue (priority-sorted, FIFO within priority)
```

### Step 3: MEDIATOR Processes Queue
```
If CUDA is idle:
  - Pop highest priority request from appropriate pool
  - Send to CUDA: execute_vector_add(100, 200)
  - Mark CUDA as busy
  - Continue polling for new requests

If CUDA is busy:
  - Continue accepting new requests
  - Insert into queue (priority-sorted)
  - Wait for CUDA result
```

### Step 4: CUDA Executes (Asynchronously)
```
CUDA receives: (100, 200)
Performs: vector addition on GPU
Returns: 300
```

### Step 5: MEDIATOR Receives Result
```
MEDIATOR receives result: 300
Identifies requesting VM: vm_id=1
Writes to /var/vgpu/vm1/response.txt: "300"
```

### Step 6: File Initialization
```
MEDIATOR clears:
  - /var/vgpu/vm1/request.txt (empty or delete)
  - /var/vgpu/vm1/response.txt (empty or delete)
```

### Step 7: Process Next Request
```
MEDIATOR pops next request from queue
Sends to CUDA (if available)
Repeats process
```

---

## QUEUE MANAGEMENT RULES

### Priority Ordering
```
High Priority (2) → Processed first
Medium Priority (1) → Processed after high
Low Priority (0) → Processed last
```

### FIFO Within Same Priority
```
If two requests have same priority:
  - Earlier timestamp → Processed first
  - Later timestamp → Processed after
```

### Pool Separation (CORRECTED)
```
Pool A and Pool B share the same priority system:
  - Single priority queue spans both pools
  - Pool A High = Pool B High (same priority level)
  - Pool ID is metadata only (for tracking/logging)
  - Processing order: Priority → FIFO (pool doesn't matter)
```

### Queue Re-ordering
```
While CUDA is processing:
  - New requests continue to arrive
  - Inserted into queue (priority-sorted)
  - Queue is re-evaluated when CUDA becomes available
  - Next highest priority request is selected
```

---

## ASYNCHRONOUS PROCESSING MODEL

### Current State (Synchronous - NOT what you want)
```
Request → Queue → CUDA (blocking) → Result → Next Request
```

### Desired State (Asynchronous - What you want)
```
Request 1 → Queue → CUDA (async) ──┐
Request 2 → Queue (while CUDA busy)│
Request 3 → Queue (while CUDA busy)│
...                                 │
Result 1 ←──────────────────────────┘
  ↓
Send Result 1 to VM-1
Initialize files
  ↓
Request 2 → CUDA (async) ──┐
Request 4 → Queue (while busy)│
...
Result 2 ←───────────────────┘
  ↓
Send Result 2 to VM-2
...
```

---

## IMPLEMENTATION REQUIREMENTS

### 1. Request Format
```
Format: "pool_id:priority:vm_id:number1:number2"
Example: "A:2:1:100:200"
```

### 2. Response Format
```
Format: "result"
Example: "300"
```

### 3. File Management
```
After sending response:
  - Clear /var/vgpu/vm<id>/request.txt
  - Clear /var/vgpu/vm<id>/response.txt
  (Or delete and recreate)
```

### 4. CUDA Integration
```
CUDA function signature:
  int cuda_vector_add(int num1, int num2, int *result);
  
Execution:
  - Non-blocking (async)
  - Returns immediately
  - Callback when result ready
```

### 5. Queue Structure
```
Per Pool:
  - Priority-sorted linked list
  - FIFO within same priority (timestamp-based)
  - Thread-safe (mutex locks)
```

### 6. MEDIATOR State
```
- Pool A queue
- Pool B queue
- CUDA busy flag
- Current CUDA request (vm_id, numbers)
- Callback handler for CUDA results
```

---

## KEY DIFFERENCES FROM CURRENT IMPLEMENTATION

### Current mediator.c:
- ❌ Synchronous CUDA execution (blocks)
- ❌ Processes one request at a time
- ❌ No async handling

### Required Implementation:
- ✅ Asynchronous CUDA execution
- ✅ Continuous request acceptance
- ✅ Queue re-ordering while CUDA busy
- ✅ File initialization after response
- ✅ Callback-based result handling

---

## CONFIRMATION QUESTIONS

1. **VM Assignment:** VMs 1-3 → Pool A, VMs 4-6 → Pool B, VM-7 → any pool?
2. **Request Format:** `pool_id:priority:vm_id:number1:number2`?
3. **Response Format:** Just the result number (e.g., "300")?
4. **File Initialization:** Clear both request.txt and response.txt after sending result?
5. **Async Processing:** MEDIATOR continues accepting requests while CUDA is busy?
6. **Queue Re-ordering:** Queue is re-evaluated when CUDA becomes available?

---

**Please confirm if this understanding is correct before I proceed with implementation.**
