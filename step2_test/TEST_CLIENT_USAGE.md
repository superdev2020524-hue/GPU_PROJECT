# Test MEDIATOR Client - Usage Guide

## Overview

The `test_mediator_client` is a testing tool that simulates multiple VMs sending requests to the MEDIATOR daemon. It demonstrates:
- How CUDA processes simultaneous requests
- How scheduling works (priority + FIFO)
- Queue state visualization
- Real-time operation experience

## Building

```bash
cd /home/david/Downloads/gpu/step2_test
make vm
```

This builds both `vm_client_vector` and `test_mediator_client` in the `build-vm/` directory.

## Prerequisites

1. **MEDIATOR daemon must be running** on the host (Dom0)
2. **NFS must be mounted** at `/mnt/vgpu` on the VM
3. **VM directories must exist**: `/mnt/vgpu/vm1/`, `/mnt/vgpu/vm2/`, etc.

## Usage

### Scenario 1: Simultaneous Requests

Send all requests at the same time to see priority ordering:

```bash
./test_mediator_client simultaneous --vms "1:A:2:100:200,4:B:2:150:250,2:A:1:50:75"
```

**Format**: `vm_id:pool:priority:num1:num2`

**Example**:
- `1:A:2:100:200` = VM-1, Pool A, High priority (2), 100+200
- `4:B:2:150:250` = VM-4, Pool B, High priority (2), 150+250
- `2:A:1:50:75` = VM-2, Pool A, Medium priority (1), 50+75

**Expected Behavior**:
- All requests submitted simultaneously
- Queue ordered by priority (High → Medium → Low)
- Within same priority: FIFO ordering
- Shows how MEDIATOR schedules requests

### Scenario 2: Sequential Requests

Send requests one after another with a delay to see FIFO behavior:

```bash
./test_mediator_client sequential --vms "1:A:2,2:A:2,3:A:2" --delay 0.5 --nums "100:200,150:250,50:75"
```

**Arguments**:
- `--vms`: VM specifications (can omit numbers, will use defaults)
- `--delay`: Delay between requests in seconds (default: 0.5)
- `--nums`: Optional, specify numbers for each VM: `num1:num2,num1:num2,...`

**Example**:
```bash
./test_mediator_client sequential \
  --vms "1:A:2,4:B:2,2:A:1,5:B:1,3:A:0" \
  --delay 1.0 \
  --nums "100:200,150:250,50:75,80:120,200:300"
```

**Expected Behavior**:
- Requests submitted sequentially with delay
- Shows queue building up
- Demonstrates FIFO within same priority

### Scenario 3: Preset Tests

Run predefined test scenarios:

```bash
./test_mediator_client preset1
```

**Preset 1**: Mixed priorities, simultaneous
- VM-1 (Pool A, High): 100+200
- VM-4 (Pool B, High): 150+250
- VM-2 (Pool A, Medium): 50+75
- VM-5 (Pool B, Medium): 80+120
- VM-3 (Pool A, Low): 200+300

## Output Format

The test client displays:

### 1. Timeline
Shows when each request is:
- Submitted
- Processing
- Completed (with result and timing)

### 2. Queue State
Shows inferred queue state based on:
- Priority ordering (High → Medium → Low)
- FIFO within same priority
- Current processing status

### 3. Statistics
- Total requests
- Completed/errors
- Average/min/max response times
- Priority distribution
- Pool distribution

## Example Output

```
================================================================================
                    TEST MEDIATOR CLIENT - Simultaneous Requests
================================================================================

Test Configuration:
  VMs: 5
  Timing: Simultaneous (all at T=0)

┌─────────────────────────────────────────────────────────────────────────┐
│ Timeline                                                               │
├─────────────────────────────────────────────────────────────────────────┤
│ T=0.00   [VM-1] Pool A, High   → Request: 100+200 (submitted at T=0.00)
│ T=0.00   [VM-4] Pool B, High   → Request: 150+250 (submitted at T=0.00)
│ T=0.00   [VM-2] Pool A, Medium → Request: 50+75 (submitted at T=0.00)
│            → Response: 125 (received at T=0.52, total=0.52s)
│ T=0.00   [VM-5] Pool B, Medium → Request: 80+120 (submitted at T=0.00)
│            → Response: 200 (received at T=0.55, total=0.55s)
│ T=0.00   [VM-3] Pool A, Low    → Request: 200+300 (submitted at T=0.00)
│            → Response: 500 (received at T=1.10, total=1.10s)
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Inferred Queue State (Priority → FIFO)                                │
├─────────────────────────────────────────────────────────────────────────┤
│ 1. VM-1 (Pool A, High, T=0.00) → 100+200 ✓ Result: 300
│ 2. VM-4 (Pool B, High, T=0.00) → 150+250 ✓ Result: 400
│ 3. VM-2 (Pool A, Medium, T=0.00) → 50+75 ✓ Result: 125
│ 4. VM-5 (Pool B, Medium, T=0.00) → 80+120 ✓ Result: 200
│ 5. VM-3 (Pool A, Low, T=0.00) → 200+300 ✓ Result: 500
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Statistics                                                             │
├─────────────────────────────────────────────────────────────────────────┤
│ Total Requests:    5
│ Completed:          5
│ Errors:            0
│ Average Response:  0.65s
│ Min Response:      0.52s
│ Max Response:      1.10s
│ Priority Distribution:
│   High:   2 requests
│   Medium: 2 requests
│   Low:    1 requests
│ Pool Distribution:
│   Pool A: 3 requests
│   Pool B: 2 requests
└─────────────────────────────────────────────────────────────────────────┘
```

## Tips

1. **Start MEDIATOR first**: Make sure the MEDIATOR daemon is running on the host before starting tests

2. **Check NFS mount**: Verify `/mnt/vgpu` is mounted and accessible

3. **VM directories**: Ensure directories exist:
   ```bash
   ls /mnt/vgpu/vm1/ /mnt/vgpu/vm2/ /mnt/vgpu/vm3/ ...
   ```

4. **Watch real-time**: The display updates every 0.5 seconds showing progress

5. **Interpret queue state**: The queue state is inferred from request timing and priority rules, not directly queried from MEDIATOR

## Troubleshooting

### "Failed to open request file"
- Check NFS mount: `mount | grep vgpu`
- Verify directory exists: `ls /mnt/vgpu/vmX/`
- Check permissions

### "Timeout waiting for response"
- Verify MEDIATOR is running on host
- Check MEDIATOR logs for errors
- Verify NFS is working correctly

### "vGPU stub device not found"
- This tool doesn't need vGPU device (it simulates VMs)
- If you see this, you might be running `vm_client_vector` instead

## Differences from vm_client_vector

| Feature | vm_client_vector | test_mediator_client |
|---------|------------------|---------------------|
| Reads vGPU properties | Yes (from MMIO) | No (simulated) |
| Single request | Yes | No (multiple) |
| Real VM | Yes | Simulated |
| Testing/Visualization | No | Yes |
| Display timeline | No | Yes |
| Queue visualization | No | Yes |

## Use Cases

1. **Verify scheduling**: Confirm priority and FIFO ordering works correctly
2. **Performance testing**: Measure response times under different loads
3. **Debugging**: Identify issues with queue management
4. **Demonstration**: Show how the system works to others
5. **Regression testing**: Ensure changes don't break scheduling

---

**Note**: This tool runs on VMs (not Dom0) and does NOT require CUDA. It only needs NFS access to communicate with the MEDIATOR daemon.
