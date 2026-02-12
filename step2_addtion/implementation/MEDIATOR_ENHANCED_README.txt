================================================================
  Enhanced Mediator Daemon - MMIO/Socket Communication
================================================================

The enhanced mediator daemon replaces NFS file polling with direct
Unix domain socket communication from vgpu-stub devices.

================================================================
  WHAT CHANGED
================================================================

  OLD (mediator_async.c)          NEW (mediator_enhanced.c)
  ─────────────────────────────   ──────────────────────────────
  Polls NFS files                 Listens on Unix socket
  /var/vgpu/vm*/request.txt        /tmp/vgpu-mediator.sock
  Text format: "A:2:1:10:20"      Binary: VGPUSocketHeader + VGPURequest
  Writes response.txt              Sends response via socket
  Clears request.txt               Keeps connection open for response

  Same priority queue logic        Same priority queue logic
  Same CUDA execution              Same CUDA execution
  Same statistics                  Same statistics

================================================================
  BUILD INSTRUCTIONS
================================================================

  Prerequisites:
    - CUDA toolkit installed
    - vgpu_protocol.h in current directory
    - cuda_vector_add.c from step2_test/

  Build:
    make mediator_enhanced

  Or manually:
    gcc -O2 -Wall -I. -I/usr/local/cuda/include \
        -c mediator_enhanced.c -o mediator_enhanced.o
    gcc -O2 -Wall -I. -I/usr/local/cuda/include \
        -c ../step2_test/cuda_vector_add.c -o cuda_vector_add.o
    gcc -o mediator_enhanced mediator_enhanced.o cuda_vector_add.o \
        -lpthread -L/usr/local/cuda/lib64 -lcudart -lcuda

================================================================
  USAGE
================================================================

  Start the mediator:
    sudo ./mediator_enhanced

  The mediator will:
    1. Create Unix socket at /tmp/vgpu-mediator.sock
    2. Accept connections from vgpu-stub devices
    3. Receive binary requests (VGPUSocketHeader + VGPURequest)
    4. Parse vector addition parameters (num1, num2)
    5. Enqueue by priority (high → medium → low, then FIFO)
    6. Execute CUDA kernel asynchronously
    7. Send response back via socket (VGPUSocketHeader + VGPUResponse)

  Stop the mediator:
    Press Ctrl+C or send SIGTERM

================================================================
  PROTOCOL FLOW
================================================================

  1. vgpu-stub connects to /tmp/vgpu-mediator.sock
  2. vgpu-stub sends:
     [VGPUSocketHeader (20 bytes)]
     [VGPURequest (32 bytes)]
     [params: num1 (4 bytes), num2 (4 bytes)]
  3. Mediator parses, enqueues, processes
  4. CUDA executes vector addition
  5. Mediator sends response:
     [VGPUSocketHeader (20 bytes)]
     [VGPUResponse (32 bytes)]
     [result: sum (4 bytes)]
  6. vgpu-stub receives, writes to MMIO response buffer
  7. Connection closes

================================================================
  MESSAGE FORMAT
================================================================

  Request (vgpu-stub → mediator):
    VGPUSocketHeader:
      magic: 0x56475055 ("VGPU")
      msg_type: VGPU_MSG_REQUEST (0x01)
      vm_id: VM identifier
      request_id: Request tracking ID
      pool_id: 'A' or 'B'
      priority: 0=low, 1=medium, 2=high
      payload_len: 40 (32 + 8 for params)

    VGPURequest (payload):
      version: VGPU_PROTOCOL_VERSION (0x00010000)
      opcode: VGPU_OP_CUDA_KERNEL (0x0001)
      flags: 0
      param_count: 2
      data_offset: 40
      data_length: 0
      reserved: [0, 0]
      params[0]: num1 (uint32_t)
      params[1]: num2 (uint32_t)

  Response (mediator → vgpu-stub):
    VGPUSocketHeader:
      magic: 0x56475055
      msg_type: VGPU_MSG_RESPONSE (0x02)
      vm_id: Same as request
      request_id: Same as request
      pool_id: Same as request
      priority: Same as request
      payload_len: 36 (32 + 4 for result)

    VGPUResponse (payload):
      version: VGPU_PROTOCOL_VERSION
      status: 0 (success)
      result_count: 1
      data_offset: 36
      data_length: 0
      exec_time_us: Execution time (microseconds)
      reserved: [0, 0]
      results[0]: sum (uint32_t)

================================================================
  TESTING
================================================================

  1. Start mediator:
     sudo ./mediator_enhanced

  2. Start VM with vgpu-stub device:
     xe vm-param-set uuid=<VM_UUID> \
       platform:device-model-args="-device vgpu-stub,pool_id=B,priority=high,vm_id=200"

  3. Inside VM, run test program:
     sudo ./test_vgpu_enhanced

  4. Check mediator output for:
     [ENQUEUE] Pool B: vm=200, req_id=9999, prio=2, 10+20
     [PROCESS] Pool B: vm=200, req_id=9999, prio=2, 10+20
     [RESULT] Pool B: vm=200, req_id=9999, result=30
     [RESPONSE] Sent to vm200 (req_id=9999): 30

================================================================
  TROUBLESHOOTING
================================================================

  Problem: "bind: Address already in use"
  Solution: Another mediator is running or socket file exists
            rm -f /tmp/vgpu-mediator.sock

  Problem: "Failed to initialize CUDA"
  Solution: Check CUDA installation and GPU availability
            nvidia-smi

  Problem: No requests received
  Solution: Check vgpu-stub is connecting:
            ls -la /tmp/vgpu-mediator.sock
            Check QEMU log for [vgpu-stub] messages

  Problem: "Invalid magic" errors
  Solution: Protocol mismatch - ensure vgpu-stub and mediator
            use same vgpu_protocol.h version

================================================================
  FILES
================================================================

  mediator_enhanced.c    - Enhanced mediator daemon source
  vgpu_protocol.h        - Shared protocol definitions
  Makefile               - Build configuration
  cuda_vector_add.c      - CUDA implementation (from step2_test/)

================================================================
