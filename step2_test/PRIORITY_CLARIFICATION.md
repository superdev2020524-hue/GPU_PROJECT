# PRIORITY SYSTEM CLARIFICATION
**Date:** 2026-02-08  
**Critical Understanding:** Pool A and Pool B share the same priority system

---

## CORRECTED PRIORITY UNDERSTANDING

### ❌ INCORRECT (Previous Understanding)
```
Pool A Queue (independent):
  High → Medium → Low

Pool B Queue (independent):
  High → Medium → Low

(Each pool processes independently)
```

### ✅ CORRECT (Actual Requirement)
```
Single Priority Queue (spans both pools):
  All High Priority (Pool A + Pool B) → 
  All Medium Priority (Pool A + Pool B) → 
  All Low Priority (Pool A + Pool B)

Within same priority level:
  - FIFO ordering (earlier request first)
  - Pool A and Pool B are equal
```

---

## PROCESSING ORDER EXAMPLE

### Scenario:
```
T0: VM-1 (Pool A, High) sends: "A:2:1:100:200"
T1: VM-4 (Pool B, High) sends: "B:2:4:150:250"
T2: VM-2 (Pool A, Medium) sends: "A:1:2:50:75"
T3: VM-5 (Pool B, High) sends: "B:2:5:80:120"
T4: VM-3 (Pool A, Low) sends: "A:0:3:200:300"
```

### Queue Order (Correct):
```
1. VM-1 (Pool A, High, T0)     ← Processed first
2. VM-4 (Pool B, High, T1)     ← Processed second (same priority, FIFO)
3. VM-5 (Pool B, High, T3)     ← Processed third (same priority, FIFO)
4. VM-2 (Pool A, Medium, T2)   ← Processed fourth (lower priority)
5. VM-3 (Pool A, Low, T4)      ← Processed last (lowest priority)
```

### Key Points:
- **Pool A High = Pool B High** (same priority level)
- **FIFO within same priority** (T0 before T1 before T3)
- **Priority determines order** (High before Medium before Low)
- **Pool ID is metadata only** (doesn't affect processing order)

---

## UPDATED QUEUE STRUCTURE

### Single Priority Queue (Not Per-Pool)
```
┌─────────────────────────────────────────────────┐
│  Priority Queue (Single, spans both pools)      │
├─────────────────────────────────────────────────┤
│                                                  │
│  High Priority (2):                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
│  │ Pool A   │→ │ Pool B   │→ │ Pool A   │→ ...│
│  │ VM-1     │  │ VM-4     │  │ VM-2     │     │
│  │ T0       │  │ T1       │  │ T5       │     │
│  └──────────┘  └──────────┘  └──────────┘     │
│                                                  │
│  Medium Priority (1):                           │
│  ┌──────────┐  ┌──────────┐                   │
│  │ Pool B   │→ │ Pool A   │→ ...               │
│  │ VM-6     │  │ VM-3     │                   │
│  │ T2       │  │ T4       │                   │
│  └──────────┘  └──────────┘                   │
│                                                  │
│  Low Priority (0):                             │
│  ┌──────────┐                                  │
│  │ Pool A   │→ ...                             │
│  │ VM-7     │                                  │
│  │ T6       │                                  │
│  └──────────┘                                  │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## IMPLEMENTATION IMPLICATIONS

### Queue Data Structure:
```c
// Single priority queue (not per-pool)
typedef struct Request {
    char pool_id;           // 'A' or 'B' (metadata only)
    uint32_t priority;      // 2=high, 1=medium, 0=low
    uint32_t vm_id;
    int num1, num2;         // Numbers to add
    time_t timestamp;       // For FIFO ordering
    struct Request *next;
} Request;

// Single queue head (not separate for each pool)
Request *priority_queue_head;  // Sorted by: priority DESC, then timestamp ASC
```

### Insertion Logic:
```c
// Insert request into single priority queue
// Sort by: priority (high to low), then timestamp (early to late)
void insert_request(Request *new_req) {
    // Find insertion point:
    // 1. Higher priority first
    // 2. Same priority: earlier timestamp first
    // Pool ID doesn't affect ordering
}
```

### Processing Logic:
```c
// Process requests in priority order
// Pool A and Pool B are equal at same priority level
Request *get_next_request() {
    // Return head of queue (highest priority, earliest timestamp)
    // Pool ID is just metadata for tracking
}
```

---

## UPDATED ARCHITECTURE

```
MEDIATOR
  │
  ├─ Single Priority Queue (spans Pool A + Pool B)
  │   │
  │   ├─ High Priority (2)
  │   │   ├─ Pool A requests (FIFO)
  │   │   └─ Pool B requests (FIFO)
  │   │
  │   ├─ Medium Priority (1)
  │   │   ├─ Pool A requests (FIFO)
  │   │   └─ Pool B requests (FIFO)
  │   │
  │   └─ Low Priority (0)
  │       ├─ Pool A requests (FIFO)
  │       └─ Pool B requests (FIFO)
  │
  └─ CUDA Executor (processes in queue order)
```

---

## CONFIRMATION

**Correct Understanding:**
- ✅ Pool A and Pool B have equal priority (not separate systems)
- ✅ High priority in Pool A = High priority in Pool B
- ✅ Single priority queue spans both pools
- ✅ Processing order: Priority (high→medium→low), then FIFO within priority
- ✅ Pool ID is metadata for tracking/logging only

**This is now correctly documented and will be implemented accordingly.**
