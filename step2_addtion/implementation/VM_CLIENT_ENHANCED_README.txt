================================================================
  Enhanced VM Client - MMIO Communication
================================================================

The enhanced VM client replaces NFS file I/O with direct MMIO
register and buffer communication with the vGPU stub device.

================================================================
  WHAT CHANGED
================================================================

  OLD (vm_client_vector.c)        NEW (vm_client_enhanced.c)
  ─────────────────────────────   ──────────────────────────────
  Reads properties from MMIO      Reads properties from MMIO
  Writes to NFS file               Writes to MMIO buffer (0x040)
  /mnt/vgpu/vm*/request.txt        Rings doorbell register
  Polls NFS file                   Polls status register
  /mnt/vgpu/vm*/response.txt       Reads from MMIO buffer (0x440)
  Text format                      Binary protocol (VGPURequest/Response)

================================================================
  BUILD INSTRUCTIONS
================================================================

  Build (inside guest VM):
    gcc -O2 -Wall -o vm_client_enhanced vm_client_enhanced.c

  Or use Makefile:
    make vm_client_enhanced

================================================================
  USAGE
================================================================

  Run the enhanced client:
    sudo ./vm_client_enhanced <num1> <num2>

  Example:
    sudo ./vm_client_enhanced 100 200

  The client will:
    1. Find vGPU stub PCI device
    2. Read properties (pool_id, priority, vm_id) from MMIO
    3. Write VGPURequest to MMIO buffer (0x040)
    4. Set request length
    5. Ring doorbell register
    6. Poll status register until DONE or ERROR
    7. Read VGPUResponse from MMIO buffer (0x440)
    8. Display result

================================================================
  PROTOCOL FLOW
================================================================

  1. Client finds vGPU stub device via PCI scan
  2. Client maps MMIO BAR0 (4KB)
  3. Client reads properties from registers:
     - POOL_ID (0x008)
     - PRIORITY (0x00C)
     - VM_ID (0x010)
  4. Client builds VGPURequest:
     - version: VGPU_PROTOCOL_VERSION
     - opcode: VGPU_OP_CUDA_KERNEL
     - param_count: 2
     - params[0]: num1
     - params[1]: num2
  5. Client writes request to buffer (0x040)
  6. Client sets REQUEST_LEN register (0x018)
  7. Client writes 1 to DOORBELL register (0x000)
  8. vGPU stub receives doorbell, sends to mediator via socket
  9. Mediator processes, sends response back
  10. vGPU stub writes response to buffer (0x440), sets STATUS=DONE
  11. Client polls STATUS register until DONE
  12. Client reads VGPUResponse from buffer (0x440)
  13. Client extracts result and displays it

================================================================
  REQUEST FORMAT
================================================================

  MMIO Buffer at offset 0x040:
    VGPURequest (32 bytes):
      version: 0x00010000
      opcode: 0x0001 (CUDA_KERNEL)
      flags: 0
      param_count: 2
      data_offset: 40
      data_length: 0
      reserved: [0, 0]
    
    Parameters (8 bytes):
      params[0]: num1 (uint32_t)
      params[1]: num2 (uint32_t)

  Total size: 40 bytes

================================================================
  RESPONSE FORMAT
================================================================

  MMIO Buffer at offset 0x440:
    VGPUResponse (32 bytes):
      version: 0x00010000
      status: 0 (success)
      result_count: 1
      data_offset: 36
      data_length: 0
      exec_time_us: execution time (microseconds)
      reserved: [0, 0]
    
    Results (4 bytes):
      results[0]: sum (uint32_t)

  Total size: 36 bytes

================================================================
  TESTING
================================================================

  1. Ensure enhanced vGPU stub is installed in QEMU
  2. Start enhanced mediator on host:
     sudo ./mediator_enhanced
  3. Start VM with vgpu-stub device
  4. Inside VM, build and run client:
     gcc -O2 -Wall -o vm_client_enhanced vm_client_enhanced.c
     sudo ./vm_client_enhanced 100 200
  5. Expected output:
     [SCAN] Found vGPU stub at 0000:00:08.0
     [MMIO] Read vGPU properties:
       Pool ID: A
       Priority: 2 (high)
       VM ID: 1
     [DOORBELL] Submitting request...
     [WAIT] Polling for response...
     [RESPONSE] Received: 300
     Result: 100 + 200 = 300

================================================================
  TROUBLESHOOTING
================================================================

  Problem: "vGPU stub device not found"
  Solution: Check device is attached:
            lspci | grep "Processing accelerators"
            Check QEMU cmdline has: -device vgpu-stub,...

  Problem: "Failed to open vGPU device"
  Solution: Run as root: sudo ./vm_client_enhanced

  Problem: "Device error: code=3" (MEDIATOR_UNAVAILABLE)
  Solution: Start mediator daemon on host:
            sudo ./mediator_enhanced

  Problem: "Timeout waiting for response"
  Solution: Check mediator is running and processing requests
            Check QEMU log for [vgpu-stub] messages

  Problem: "Invalid response version"
  Solution: Protocol mismatch - ensure vgpu-stub and client
            use same protocol version

================================================================
  COMPARISON WITH OLD CLIENT
================================================================

  Feature              Old (NFS)          New (MMIO)
  ──────────────────────────────────────────────────────
  Request transport    NFS file           MMIO buffer
  Response transport   NFS file           MMIO buffer
  Format               Text string        Binary protocol
  Polling              File existence     Status register
  Dependencies         NFS mount          PCI device access
  Performance          Slower (NFS I/O)   Faster (MMIO)
  Latency              Higher             Lower

================================================================
  FILES
================================================================

  vm_client_enhanced.c  - Enhanced VM client source
  vgpu_protocol.h       - Protocol definitions (shared)
  Makefile              - Build configuration

================================================================
