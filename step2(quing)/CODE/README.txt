================================================================================
                        CODE DIRECTORY
================================================================================

This directory contains source code for Phase 2: Queue-Based Mediation Layer

FILES:
------
1. vm_client.c    - VM client program (runs in guest VM)
2. mediator.c     - Mediation daemon (runs in Dom0)
3. README.txt     - This file

================================================================================
                        BUILD INSTRUCTIONS
================================================================================

ON DOM0 (Host):
--------------
gcc -o mediator mediator.c -lpthread
chmod +x mediator

ON VM (Guest):
-------------
gcc -o vm_client vm_client.c
chmod +x vm_client

================================================================================
                        USAGE
================================================================================

1. Start Daemon (Dom0):
   sudo ./mediator

2. Send Request (VM):
   sudo ./vm_client VECTOR_ADD

================================================================================
                        PREREQUISITES
================================================================================

Before running:

1. ✅ Phase 1 complete (vGPU stub device working)
2. ✅ NFS setup complete (/var/vgpu exported, /mnt/vgpu mounted)
3. ✅ Per-VM directories created (/var/vgpu/vm1, /var/vgpu/vm2, etc.)
4. ✅ VM has vgpu-stub device attached with pool_id and priority

================================================================================
                        HOW IT WORKS
================================================================================

VM Client (vm_client.c):
-----------------------
1. Reads pool_id, priority, vm_id from vGPU stub MMIO registers
2. Formats request: "pool_id:priority:vm_id:command"
3. Writes to /mnt/vgpu/vm<id>/request.txt
4. Polls /mnt/vgpu/vm<id>/response.txt for result
5. Displays response

Mediator Daemon (mediator.c):
-----------------------------
1. Polls /var/vgpu/vm*/request.txt for new requests
2. Parses request format
3. Inserts into correct pool queue (A or B)
4. Maintains priority ordering within each queue
5. Processes highest priority request
6. Executes GPU workload (placeholder for now)
7. Writes response to /var/vgpu/vm<id>/response.txt

Queue Management:
----------------
- Two independent queues (Pool A and Pool B)
- Priority-sorted insertion (high=2, medium=1, low=0)
- FIFO tie-breaking within same priority
- Thread-safe with mutex locks

================================================================================
                        TESTING
================================================================================

Test 1: Single VM Request
-------------------------
[Dom0] ./mediator
[VM1]  sudo ./vm_client VECTOR_ADD

Expected: VM1 receives response

Test 2: Priority Ordering
-------------------------
Setup: 3 VMs with different priorities in same pool
Submit: low, then high, then medium
Expected: Processed in order: high, medium, low

Test 3: Pool Separation
-----------------------
Setup: VM1 in Pool A, VM2 in Pool B
Submit: Both simultaneously
Expected: Both processed independently, no interference

Test 4: Concurrency
-------------------
[VM1] while true; do sudo ./vm_client TEST1; sleep 1; done
[VM2] while true; do sudo ./vm_client TEST2; sleep 1; done
Expected: Both receive responses, no crashes

================================================================================
                        NEXT STEPS
================================================================================

1. Test basic communication (non-CUDA)
2. Add CUDA integration (replace execute_gpu_workload placeholder)
3. Test with real CUDA workloads
4. Performance tuning

================================================================================
End of README
================================================================================
