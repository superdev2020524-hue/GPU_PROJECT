# Test MEDIATOR Design Proposal

## Purpose

Create a test client that simulates multiple VMs sending requests to demonstrate:
1. **CUDA progress and response** to simultaneous requests from different VMs
2. **Scheduling behavior** when VMs arrive sequentially
3. **Queue state visualization** showing priority and FIFO ordering
4. **Real-time operation experience** for the user

## Key Requirements

1. **Same client functionality** as `vm_client_vector.c`:
   - Uses NFS to send requests
   - Reads/writes to same NFS directories (`/mnt/vgpu/vmX/`)
   - Same request/response format
   - Same file handling

2. **Testing capabilities**:
   - Simulate multiple VMs (different VM IDs, pools, priorities)
   - Control request timing (simultaneous vs sequential)
   - Display queue state and scheduling decisions
   - Show CUDA execution progress
   - Visualize request flow

3. **Won't run simultaneously** with `vm_client_vector.c`

## Proposed Design

### Architecture

```
test_mediator_client
├── VM Simulation Engine
│   ├── Simulate multiple VMs
│   ├── Control request timing
│   └── Track request state
├── Request Manager
│   ├── Send requests via NFS (same as vm_client_vector.c)
│   ├── Poll for responses
│   └── Track request lifecycle
├── Display System
│   ├── Real-time status updates
│   ├── Queue visualization
│   ├── Timeline display
│   └── Statistics
└── Test Scenarios
    ├── Simultaneous requests
    ├── Sequential requests
    └── Mixed scenarios
```

### Core Components

#### 1. VM Simulation
- **Purpose**: Simulate multiple VMs with different properties
- **Features**:
  - Define test VMs with: VM ID, Pool ID, Priority
  - Can use real vGPU properties OR simulated properties
  - Support for multiple concurrent VM simulations

#### 2. Request Sender
- **Purpose**: Send requests exactly like `vm_client_vector.c`
- **Implementation**:
  - Reuse NFS communication code from `vm_client_vector.c`
  - Same file format: `pool_id:priority:vm_id:num1:num2`
  - Same response polling mechanism
  - Track: submission time, response time, result

#### 3. Status Monitor
- **Purpose**: Monitor and display what's happening
- **Features**:
  - Real-time display of queue state (from MEDIATOR logs)
  - Request submission timeline
  - Response timeline
  - CUDA execution status
  - Scheduling decisions (why request X was chosen)

#### 4. Display System
- **Purpose**: Show user what's happening
- **Features**:
  - Color-coded output (different colors for different VMs/priorities)
  - Timeline view (when requests submitted, processed, completed)
  - Queue state visualization
  - Statistics summary

### Test Scenarios

#### Scenario 1: Simultaneous Requests
```
Purpose: Show how MEDIATOR handles burst of requests
Timing: All requests sent at same time (or very close)
VMs: Multiple VMs with different priorities
Expected: Show priority ordering, queue state, processing order
```

#### Scenario 2: Sequential Requests
```
Purpose: Show FIFO behavior within same priority
Timing: Requests sent one after another with delays
VMs: Multiple VMs with same priority
Expected: Show FIFO ordering, queue building up
```

#### Scenario 3: Mixed Priority + Sequential
```
Purpose: Show complete scheduling behavior
Timing: Mix of simultaneous and sequential
VMs: Different priorities, different pools
Expected: Show priority first, then FIFO, queue reordering
```

## Implementation Approach

### Option A: Single Process with Threads
**Pros:**
- Easier to coordinate display
- Single process managing all VM simulations
- Easier to track state

**Cons:**
- More complex threading
- Need to coordinate NFS access

### Option B: Single Process, Sequential Simulation
**Pros:**
- Simpler implementation
- No threading complexity
- Easier to control timing

**Cons:**
- Less realistic (not truly simultaneous)
- May not catch all race conditions

### Option C: Multiple Processes (Fork)
**Pros:**
- Most realistic (truly independent VMs)
- Each process is like a real VM client
- Better for testing concurrent access

**Cons:**
- Harder to coordinate display
- More complex to manage

## Recommended Approach: **Option A (Threads) with Sequential Fallback**

### Why This Approach?

1. **Realistic Testing**: Threads can send requests simultaneously (or very close)
2. **Coordinated Display**: Single process can show unified view
3. **Flexible Timing**: Can control delays between requests
4. **State Tracking**: Easier to track all requests in one place

### Implementation Details

#### Structure
```c
typedef struct {
    uint32_t vm_id;
    char pool_id;
    uint32_t priority;
    int num1, num2;
    time_t submit_time;
    time_t response_time;
    int result;
    int status;  // PENDING, PROCESSING, COMPLETED, ERROR
} TestRequest;

typedef struct {
    TestRequest *requests;
    int count;
    pthread_mutex_t lock;
} TestState;
```

#### Main Flow
```
1. Parse test scenario configuration
2. Initialize test state
3. Start display thread (updates screen periodically)
4. For each test request:
   a. Create thread to simulate VM
   b. Thread sends request via NFS (like vm_client_vector.c)
   c. Thread polls for response
   d. Thread updates test state when done
5. Display results and statistics
```

#### Display Format
```
================================================================================
                    TEST MEDIATOR CLIENT - Scenario: Simultaneous
================================================================================

Test Configuration:
  VMs: 5 (VM-1, VM-2, VM-3, VM-4, VM-5)
  Timing: Simultaneous (all at T=0)
  
┌─────────────────────────────────────────────────────────────────────────┐
│ Timeline                                                               │
├─────────────────────────────────────────────────────────────────────────┤
│ T=0.0s  [VM-1] Pool A, High   → Request: 100+200                      │
│ T=0.0s  [VM-2] Pool A, Medium → Request: 150+250                      │
│ T=0.0s  [VM-3] Pool A, Low    → Request: 50+75                        │
│ T=0.0s  [VM-4] Pool B, High   → Request: 80+120                       │
│ T=0.0s  [VM-5] Pool B, Medium → Request: 200+300                       │
│                                                                         │
│ T=0.1s  [MEDIATOR] Queue State:                                        │
│         1. VM-1 (Pool A, High, T=0.0)                                  │
│         2. VM-4 (Pool B, High, T=0.0)  ← Same priority, FIFO          │
│         3. VM-2 (Pool A, Medium, T=0.0)                                 │
│         4. VM-5 (Pool B, Medium, T=0.0)                                │
│         5. VM-3 (Pool A, Low, T=0.0)                                   │
│                                                                         │
│ T=0.2s  [CUDA] Processing VM-1: 100+200                                 │
│ T=0.5s  [CUDA] Completed VM-1: Result=300                              │
│ T=0.5s  [VM-1] Received response: 300                                  │
│ T=0.6s  [CUDA] Processing VM-4: 80+120                                  │
│ ...                                                                     │
└─────────────────────────────────────────────────────────────────────────┘

Statistics:
  Total Requests: 5
  Completed: 5
  Average Response Time: 0.8s
  Queue Max Size: 5
  Priority Distribution:
    High: 2 requests
    Medium: 2 requests
    Low: 1 request
```

## Key Design Decisions

### 1. Reuse vm_client_vector.c Code
**Decision**: Extract common functions into shared code or copy relevant parts
**Reason**: Maintain consistency with actual VM client behavior

### 2. How to Monitor MEDIATOR Queue State
**Options**:
- A. Parse MEDIATOR logs (if running with verbose output)
- B. Query MEDIATOR via separate interface (not currently available)
- C. Infer from request/response timing
- D. Display what we know (our requests, their timing, responses)

**Recommended**: **Option D** - Display what we can observe:
- When we submit requests
- When we receive responses
- Infer queue state from timing
- Show scheduling decisions based on priority/FIFO rules

### 3. Real-time vs Post-Processing Display
**Decision**: Real-time updates with periodic refresh
**Reason**: Better user experience, shows system behavior as it happens

### 4. Configuration Method
**Options**:
- A. Command-line arguments
- B. Configuration file
- C. Interactive mode

**Recommended**: **Command-line with preset scenarios**
```bash
# Simultaneous requests
./test_mediator_client simultaneous --vms "1:A:2:100:200,4:B:2:150:250,2:A:1:50:75"

# Sequential requests
./test_mediator_client sequential --vms "1:A:2,2:A:2,3:A:2" --delay 0.5

# Preset scenarios
./test_mediator_client preset1  # Predefined test scenario
```

## Why This Design?

### 1. Maintains Client Behavior
- Uses same NFS communication
- Same file formats
- Same error handling
- **Result**: Tests reflect real VM behavior

### 2. Shows Scheduling Clearly
- Visual timeline shows request order
- Queue state shows priority ordering
- Response timing shows processing order
- **Result**: User can see how scheduling works

### 3. Flexible Testing
- Can test simultaneous requests
- Can test sequential requests
- Can test mixed scenarios
- **Result**: Comprehensive testing capability

### 4. Good User Experience
- Real-time updates
- Clear visualization
- Statistics summary
- **Result**: Easy to understand system behavior

## Implementation Plan

### Phase 1: Core Infrastructure
1. Extract/copy NFS communication code from `vm_client_vector.c`
2. Create test request structure
3. Implement basic request sending
4. Implement response polling

### Phase 2: VM Simulation
1. Implement VM simulation (threaded or sequential)
2. Support multiple concurrent requests
3. Track request lifecycle

### Phase 3: Display System
1. Implement real-time status display
2. Create timeline visualization
3. Add statistics calculation

### Phase 4: Test Scenarios
1. Implement simultaneous request scenario
2. Implement sequential request scenario
3. Implement mixed scenario
4. Add preset test configurations

### Phase 5: Polish
1. Add color coding
2. Improve formatting
3. Add error handling
4. Documentation

## Questions for Discussion

1. **Threading vs Sequential**: Do you prefer truly simultaneous requests (threads) or controlled sequential (easier to follow)?

2. **Display Detail**: How much detail do you want?
   - Just request/response timeline?
   - Full queue state visualization?
   - CUDA execution progress?

3. **Configuration**: Command-line, config file, or interactive?

4. **Real MEDIATOR Integration**: Should we try to query MEDIATOR state, or just display what we observe?

5. **Timing Control**: Should we be able to control exact timing of requests, or just "simultaneous" vs "sequential"?

## Expected Benefits

1. **Visual Understanding**: See how priority and FIFO scheduling works
2. **Testing**: Verify scheduling behavior is correct
3. **Debugging**: Identify issues with queue management
4. **Documentation**: Demonstrate system behavior to others
5. **Performance**: See response times, queue buildup, etc.

---

**Please review this design and let me know:**
1. Does this match what you want?
2. Any changes or additions?
3. Preferred implementation approach?
4. Any specific features you want to see?

Once confirmed, I'll proceed with implementation.
