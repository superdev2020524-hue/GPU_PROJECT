# ARCHITECTURE DIAGRAM - CUDA Vector Addition System

## COMPLETE SYSTEM FLOW

```
┌─────────────────────────────────────────────────────────────────────┐
│                         VM LAYER (7 VMs)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐│
│  │  VM-1    │  │  VM-2    │  │  VM-3    │  │  VM-4    │  │  VM-5    ││
│  │ Pool A   │  │ Pool A   │  │ Pool A   │  │ Pool B   │  │ Pool B   ││
│  │ Priority │  │ Priority │  │ Priority │  │ Priority │  │ Priority ││
│  │ (any)    │  │ (any)    │  │ (any)    │  │ (any)    │  │ (any)    ││
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘│
│       │             │              │             │             │      │
│       │  ┌─────────┐│              │             │             │      │
│       │  │  VM-6    ││              │             │             │      │
│       │  │ Pool B   ││              │             │             │      │
│       │  │ Priority ││              │             │             │      │
│       │  │ (any)    ││              │             │             │      │
│       │  └────┬─────┘│              │             │             │      │
│       │       │      │              │             │             │      │
│       │       │  ┌───┴──────────────┴─────────────┴─────────────┴──┐ │
│       │       │  │  VM-7 (Any Pool, Any Priority)                    │ │
│       │       │  └──────────────────────────────────────────────────┘ │
│       │       │                                                       │
└───────┼───────┼───────────────────────────────────────────────────────┘
        │       │
        │       │  NFS Mount: /mnt/vgpu
        │       │  Per-VM directories: vm1/, vm2/, ..., vm7/
        │       │
        │       │  Request Format:
        │       │  "pool_id:priority:vm_id:number1:number2"
        │       │
        ▼       ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    NFS SHARED DIRECTORY                              │
│                    /var/vgpu (Dom0 Export)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  /var/vgpu/                                                         │
│  ├── vm1/                                                           │
│  │   ├── request.txt   ← VM writes: "A:2:1:100:200"               │
│  │   └── response.txt  ← MEDIATOR writes: "300"                    │
│  ├── vm2/                                                           │
│  │   ├── request.txt                                               │
│  │   └── response.txt                                              │
│  ├── ...                                                            │
│  └── vm7/                                                           │
│      ├── request.txt                                               │
│      └── response.txt                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
        │
        │  MEDIATOR polls request.txt files
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    MEDIATOR DAEMON (Dom0)                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  REQUEST POLLER (Continuous Loop)                            │  │
│  │  - Polls /var/vgpu/vm*/request.txt                           │  │
│  │  - Parses: pool_id, priority, vm_id, num1, num2             │  │
│  │  - Inserts into appropriate queue                            │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            │                                         │
│                            ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  POOL A QUEUE (Priority-Sorted, FIFO within priority)        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │  │
│  │  │ High(2)  │→ │ High(2)  │→ │ Med(1)    │→ ...              │  │
│  │  │ VM-1     │  │ VM-2     │  │ VM-3      │                  │  │
│  │  │ 100+200  │  │ 50+75    │  │ 200+300   │                  │  │
│  │  └──────────┘  └──────────┘  └──────────┘                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            │                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  POOL B QUEUE (Priority-Sorted, FIFO within priority)        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │  │
│  │  │ High(2)  │→ │ Med(1)    │→ │ Low(0)    │→ ...              │  │
│  │  │ VM-4     │  │ VM-5      │  │ VM-6      │                  │  │
│  │  │ 150+250  │  │ 80+120    │  │ 30+40     │                  │  │
│  │  └──────────┘  └──────────┘  └──────────┘                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            │                                         │
│                            ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  QUEUE MANAGER                                                │  │
│  │  - Selects highest priority request from either pool          │  │
│  │  - Sends to CUDA Executor (if CUDA idle)                      │  │
│  │  - Continues accepting new requests while CUDA busy           │  │
│  │  - Re-orders queue when CUDA becomes available                │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            │                                         │
│                            ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  CUDA EXECUTOR (Asynchronous)                                 │  │
│  │  ┌────────────────────────────────────────────────────────┐  │  │
│  │  │ Status: BUSY                                           │  │  │
│  │  │ Current: VM-1, Pool A, High, 100+200                   │  │  │
│  │  │                                                         │  │  │
│  │  │ While processing:                                      │  │  │
│  │  │ - New requests continue to arrive                      │  │  │
│  │  │ - Inserted into queues                                │  │  │
│  │  │ - Queue re-ordered                                    │  │  │
│  │  └────────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                            │                                         │
│                            │  Async call: cuda_vector_add(100, 200)
│                            │
│                            ▼
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  RESULT HANDLER                                               │  │
│  │  - Receives result from CUDA (callback)                      │  │
│  │  - Identifies requesting VM (vm_id)                         │  │
│  │  - Writes to /var/vgpu/vm<id>/response.txt                  │  │
│  │  - Clears request.txt and response.txt                       │  │
│  │  - Triggers next request processing                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
        │
        │  CUDA kernel execution
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    CUDA GPU (NVIDIA H100)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Vector Addition Kernel:                                            │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │  __global__ void vector_add(int *a, int *b, int *c)         │ │
│  │  {                                                           │ │
│  │      int idx = threadIdx.x;                                 │ │
│  │      c[idx] = a[idx] + b[idx];                              │ │
│  │  }                                                           │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  Execution:                                                         │
│  - Input: num1=100, num2=200                                       │
│  - Output: result=300                                               │
│  - Returns to MEDIATOR via callback                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## TIMELINE EXAMPLE

### T0: VM-1 sends request
```
VM-1 → request.txt: "A:2:1:100:200"
MEDIATOR → Enqueues to Pool A (High priority)
MEDIATOR → Sends to CUDA: vector_add(100, 200)
CUDA → Status: BUSY
```

### T1: VM-2 sends request (CUDA still busy)
```
VM-2 → request.txt: "A:1:2:50:75"
MEDIATOR → Enqueues to Pool A (Medium priority)
MEDIATOR → CUDA still busy, request queued
```

### T2: VM-4 sends request (CUDA still busy)
```
VM-4 → request.txt: "B:2:4:150:250"
MEDIATOR → Enqueues to Pool B (High priority)
MEDIATOR → CUDA still busy, request queued
```

### T3: CUDA completes VM-1 request
```
CUDA → Returns: 300
MEDIATOR → Writes to vm1/response.txt: "300"
MEDIATOR → Clears vm1/request.txt and response.txt
MEDIATOR → Selects next: VM-4 (Pool B, High priority)
MEDIATOR → Sends to CUDA: vector_add(150, 250)
```

### T4: CUDA completes VM-4 request
```
CUDA → Returns: 400
MEDIATOR → Writes to vm4/response.txt: "400"
MEDIATOR → Clears vm4/request.txt and response.txt
MEDIATOR → Selects next: VM-2 (Pool A, Medium priority)
MEDIATOR → Sends to CUDA: vector_add(50, 75)
```

---

## KEY FEATURES

1. **Asynchronous Processing:** CUDA operations don't block request acceptance
2. **Priority Queuing:** High priority processed before medium/low
3. **FIFO Within Priority:** Earlier requests processed first
4. **Pool Separation:** Pool A and Pool B independent
5. **Continuous Operation:** MEDIATOR always accepting new requests
6. **File Management:** Files cleared after response sent
7. **Queue Re-ordering:** Queue evaluated when CUDA becomes available

---

**This architecture supports your requirements. Please confirm if this matches your vision.**
