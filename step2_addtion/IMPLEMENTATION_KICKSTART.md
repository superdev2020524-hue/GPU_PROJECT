# Implementation Kickstart - Start Here!

**Date:** February 12, 2026  
**For:** Immediate implementation start  
**Goal:** Get you coding in the next hour

---

## 🚀 You Are Ready to Start!

### What You Have

✅ **Working NFS-based system** (`step2_test/`)
- `mediator_async.c` - 535 lines, fully functional
- `vm_client_vector.c` - 391 lines, working
- `cuda_vector_add.c` - 349 lines, tested
- Protocol proven and documented

✅ **Complete specifications** (`step2_addtion/`)
- REGISTER_MAP_SPEC.md - Full MMIO layout
- VGPU_STUB_CHANGES.md - Exact code changes
- MIGRATION_PLAN.md - Step-by-step migration
- CURRENT_SYSTEM_ANALYSIS.md - Your system analyzed

✅ **Base vGPU stub** (`step2(quing)/vgpu-stub_enhance/complete.txt`)
- Working QEMU 4.2.1 build
- Basic vgpu-stub.c (250 lines)
- Tested on XCP-ng

---

## 📋 Quick Start Decision

### Choose Your Path

**Path A: I have 1 hour → Test the concept**
- Create enhanced vgpu-stub.c
- Build and install QEMU
- Test new registers from guest
- **Deliverable:** Verify MMIO extensions work

**Path B: I have 1 day → Complete vGPU stub**
- Full vgpu-stub.c with socket
- Build and install
- Test all registers and doorbell
- **Deliverable:** Working enhanced device

**Path C: I have 1 week → Full integration**
- Enhanced vgpu-stub.c
- Adapted mediator_mmio.c
- Adapted vm_client_mmio.c
- **Deliverable:** End-to-end MMIO system

---

## 🎯 Path A: 1-Hour Quick Start

### Step 1: Create Enhanced vgpu-stub.c (20 minutes)

I'll create the complete file for you. Just tell me and I'll generate:
- `step2_addtion/implementation/vgpu-stub-enhanced.c`

This file will include:
1. Extended register map (16 registers)
2. Request/response buffers (2x 1KB)
3. Doorbell handler (placeholder)
4. All MMIO read/write handlers

### Step 2: Build QEMU (30 minutes)

```bash
cd ~/vgpu-build/rpmbuild

# Replace vgpu-stub.c with enhanced version
cp /home/david/Downloads/gpu/step2_addtion/implementation/vgpu-stub-enhanced.c \
   SOURCES/vgpu-stub.c

# Build (takes 30-45 minutes)
rpmbuild -bb SPECS/qemu.spec

# Install
rpm -Uvh --nodeps --force RPMS/x86_64/qemu-4.2.1-*.rpm
```

### Step 3: Test Registers (10 minutes)

Create a simple test program in your test VM:

```c
// test_new_registers.c
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>

int main() {
    // Open vGPU device (adjust path if needed)
    int fd = open("/sys/bus/pci/devices/0000:00:06.0/resource0", 
                  O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("Failed to open device");
        return 1;
    }
    
    // Map MMIO region
    volatile uint32_t *mmio = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd, 0);
    if (mmio == MAP_FAILED) {
        perror("Failed to map MMIO");
        return 1;
    }
    
    printf("=== Testing New Registers ===\n\n");
    
    // Test new registers
    printf("STATUS register (0x004):        0x%08x\n", mmio[0x004/4]);
    printf("PROTOCOL_VER register (0x020):  0x%08x (expect 0x00010000)\n", 
           mmio[0x020/4]);
    printf("CAPABILITIES register (0x024):  0x%08x (expect 0x00000001)\n", 
           mmio[0x024/4]);
    printf("REQUEST_LEN register (0x018):   0x%08x\n", mmio[0x018/4]);
    printf("RESPONSE_LEN register (0x01C):  0x%08x\n", mmio[0x01C/4]);
    
    // Test doorbell (should cause STATUS change)
    printf("\n=== Testing Doorbell ===\n");
    printf("Before doorbell - STATUS: 0x%08x\n", mmio[0x004/4]);
    
    mmio[0x018/4] = 64;  // Set request length
    mmio[0x000/4] = 1;   // Ring doorbell
    
    usleep(10000);  // Wait 10ms
    printf("After doorbell - STATUS:  0x%08x\n", mmio[0x004/4]);
    printf("ERROR_CODE register:      0x%08x\n", mmio[0x014/4]);
    
    munmap((void*)mmio, 4096);
    close(fd);
    
    printf("\n✅ Test complete!\n");
    return 0;
}
```

Compile and run:
```bash
gcc test_new_registers.c -o test_new_registers
sudo ./test_new_registers
```

**Expected Output:**
```
=== Testing New Registers ===

STATUS register (0x004):        0x00000000
PROTOCOL_VER register (0x020):  0x00010000 (expect 0x00010000)
CAPABILITIES register (0x024):  0x00000001 (expect 0x00000001)
REQUEST_LEN register (0x018):   0x00000000
RESPONSE_LEN register (0x01C):  0x00000000

=== Testing Doorbell ===
Before doorbell - STATUS: 0x00000000
After doorbell - STATUS:  0x00000003
ERROR_CODE register:      0x00000003

✅ Test complete!
```

**What this means:**
- STATUS changed to 0x03 (ERROR) - Expected! No mediator connected yet
- ERROR_CODE = 0x03 (MEDIATOR_UNAVAILABLE) - Perfect!
- **All registers working correctly!**

---

## 🚀 Path B: 1-Day Complete vGPU Stub

### Hour 1-2: Enhanced vgpu-stub.c

Same as Path A + add socket connection logic.

### Hour 3-4: Socket Test

Create a simple test mediator (not full implementation):

```c
// test_socket_server.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>

int main() {
    struct sockaddr_un addr;
    int server_fd, client_fd;
    
    // Create socket
    server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    unlink("/tmp/vgpu-mediator-1.sock");  // VM ID 1
    
    // Bind
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strcpy(addr.sun_path, "/tmp/vgpu-mediator-1.sock");
    bind(server_fd, (struct sockaddr*)&addr, sizeof(addr));
    
    // Listen
    listen(server_fd, 5);
    printf("Waiting for vgpu-stub connection...\n");
    
    // Accept
    client_fd = accept(server_fd, NULL, NULL);
    printf("✅ Connected!\n");
    
    // Receive request
    uint8_t buf[2048];
    ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
    printf("Received %ld bytes\n", n);
    
    // Send dummy response
    struct {
        uint32_t vm_id;
        uint32_t request_id;
        uint32_t length;
        uint8_t data[36];  // vgpu_response size
    } response;
    
    response.vm_id = 1;
    response.request_id = 0;
    response.length = 36;
    
    // Build dummy vgpu_response
    uint32_t *resp_data = (uint32_t*)response.data;
    resp_data[0] = 0x00010000;  // version
    resp_data[1] = 0;            // status = success
    resp_data[2] = 1;            // result_count
    resp_data[3] = 32;           // data_offset
    resp_data[4] = 0;            // data_length
    resp_data[5] = 5000;         // exec_time_us
    resp_data[6] = 0;            // reserved
    resp_data[7] = 0;            // reserved
    resp_data[8] = 300;          // result (100+200)
    
    send(client_fd, &response, sizeof(response), 0);
    printf("✅ Sent response: result=300\n");
    
    close(client_fd);
    close(server_fd);
    return 0;
}
```

### Hour 5-6: Simple MMIO Client

```c
// simple_mmio_client.c
#include <stdio.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <string.h>
#include <unistd.h>

struct vgpu_request {
    uint32_t version;
    uint32_t opcode;
    uint32_t flags;
    uint32_t param_count;
    uint32_t data_offset;
    uint32_t data_length;
    uint32_t reserved[2];
    uint32_t params[2];
};

struct vgpu_response {
    uint32_t version;
    uint32_t status;
    uint32_t result_count;
    uint32_t data_offset;
    uint32_t data_length;
    uint32_t exec_time_us;
    uint32_t reserved[2];
    uint32_t results[1];
};

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <num1> <num2>\n", argv[0]);
        return 1;
    }
    
    int num1 = atoi(argv[1]);
    int num2 = atoi(argv[2]);
    
    // Open and map device
    int fd = open("/sys/bus/pci/devices/0000:00:06.0/resource0",
                  O_RDWR | O_SYNC);
    volatile uint32_t *mmio = mmap(NULL, 4096, PROT_READ | PROT_WRITE,
                                    MAP_SHARED, fd, 0);
    
    printf("[CLIENT] Sending: %d + %d\n", num1, num2);
    
    // Wait for idle
    while (mmio[0x004/4] != 0) {
        usleep(1000);
    }
    
    // Build request
    struct vgpu_request req = {
        .version = 0x00010000,
        .opcode = 0x0001,
        .flags = 0,
        .param_count = 2,
        .data_offset = sizeof(req) - 8,
        .data_length = 0,
        .reserved = {0, 0},
        .params = {num1, num2}
    };
    
    // Write to request buffer
    uint32_t *req_buf = (uint32_t*)(mmio + 0x040/4);
    memcpy(req_buf, &req, sizeof(req));
    
    // Set request length and ring doorbell
    mmio[0x018/4] = sizeof(req);
    mmio[0x000/4] = 1;
    
    printf("[CLIENT] Waiting for response...\n");
    
    // Poll for response
    while (1) {
        uint32_t status = mmio[0x004/4];
        
        if (status == 0x02) {  // DONE
            // Read response
            uint32_t resp_len = mmio[0x01C/4];
            uint32_t *resp_buf = (uint32_t*)(mmio + 0x440/4);
            
            struct vgpu_response resp;
            memcpy(&resp, resp_buf, resp_len);
            
            printf("[CLIENT] ✅ Result: %d\n", resp.results[0]);
            printf("[CLIENT] Execution time: %u μs\n", resp.exec_time_us);
            break;
        }
        else if (status == 0x03) {  // ERROR
            printf("[CLIENT] ❌ Error: %u\n", mmio[0x014/4]);
            break;
        }
        
        usleep(10000);  // 10ms
    }
    
    munmap((void*)mmio, 4096);
    close(fd);
    return 0;
}
```

### Hour 7-8: Integration Test

```bash
# Terminal 1 (on Dom0)
./test_socket_server

# Terminal 2 (on VM)
sudo ./simple_mmio_client 100 200
```

**Expected:**
```
Terminal 1:
Waiting for vgpu-stub connection...
✅ Connected!
Received 1064 bytes
✅ Sent response: result=300

Terminal 2:
[CLIENT] Sending: 100 + 200
[CLIENT] Waiting for response...
[CLIENT] ✅ Result: 300
[CLIENT] Execution time: 5000 μs
```

---

## 🎊 Path C: 1-Week Full Implementation

Follow MIGRATION_PLAN.md timeline:

**Week 1:**
- Day 1-2: Path B above
- Day 3-4: Adapt mediator_async.c → mediator_mmio.c
- Day 5: Integration testing

**Week 2:**
- Day 6-7: Multi-VM testing
- Day 8: Performance comparison
- Day 9-10: Documentation and cleanup

---

## 📦 What I Can Generate For You

Just say the word and I'll create:

### 1. Enhanced vgpu-stub.c (Complete)
- ✅ All 16 registers implemented
- ✅ Request/response buffers
- ✅ Doorbell handler with socket
- ✅ Ready to compile

### 2. mediator_mmio.c (Adapted)
- ✅ Keep your queue logic
- ✅ Keep your CUDA integration
- ✅ Replace NFS with socket
- ✅ Binary protocol parser

### 3. vm_client_mmio.c (Adapted)
- ✅ Keep your PCI scanning
- ✅ Keep your property reading
- ✅ Replace NFS with MMIO
- ✅ Binary protocol builder

### 4. Protocol Headers (Shared)
- ✅ vgpu_protocol.h
- ✅ Request/response structures
- ✅ Error codes
- ✅ Register definitions

### 5. Test Programs
- ✅ test_mmio_registers.c
- ✅ test_doorbell.c
- ✅ test_socket_communication.c
- ✅ benchmark_nfs_vs_mmio.c

### 6. Build Scripts
- ✅ build_qemu.sh (automated)
- ✅ deploy_to_vm.sh (automated)
- ✅ run_tests.sh (automated)

---

## ✅ Pre-Flight Checklist

Before starting, verify:

- [ ] I have read CURRENT_SYSTEM_ANALYSIS.md
- [ ] I have read MIGRATION_PLAN.md
- [ ] I understand my current NFS-based system
- [ ] I have backups of my current system
- [ ] I have a test VM available
- [ ] I have decided which path (A, B, or C)
- [ ] I'm ready to start coding!

---

## 🚀 Ready to Start?

Tell me which path you want:

**Option 1:** "Start with Path A - generate enhanced vgpu-stub.c"  
**Option 2:** "Start with Path B - generate all vgpu-stub files"  
**Option 3:** "Start with Path C - generate everything"  
**Option 4:** "Generate [specific file] first"

I'll create the complete, working code immediately!

---

## 💡 Pro Tips

### Tip 1: Test Incrementally
Don't try to build everything at once. Test each component:
1. Registers readable? ✅ → Move to doorbell
2. Doorbell working? ✅ → Move to socket
3. Socket connected? ✅ → Move to mediator
4. Mediator receiving? ✅ → Move to CUDA
5. CUDA executing? ✅ → Move to response
6. Response received? ✅ → System complete!

### Tip 2: Keep NFS Working
Don't remove NFS code until MMIO is proven. Keep both:
```bash
./vm_client_vector  # Old NFS version
./vm_client_mmio    # New MMIO version
```

### Tip 3: Use Logs Liberally
Add printf() everywhere during development:
```c
printf("[MMIO] Writing to register 0x%03x: 0x%08x\n", offset, value);
printf("[SOCKET] Received %zd bytes\n", n);
printf("[CUDA] Result: %d\n", result);
```

### Tip 4: Measure Everything
```c
struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC, &start);
// ... operation ...
clock_gettime(CLOCK_MONOTONIC, &end);
double elapsed_ms = (end.tv_sec - start.tv_sec) * 1000.0 +
                    (end.tv_nsec - start.tv_nsec) / 1000000.0;
printf("[PERF] Operation took %.2f ms\n", elapsed_ms);
```

---

## 📞 Ready When You Are!

**I'm prepared to generate:**
- Complete enhanced vgpu-stub.c (~500 lines)
- Adapted mediator_mmio.c (~450 lines)
- Adapted vm_client_mmio.c (~350 lines)
- Protocol headers (~200 lines)
- Test programs (~600 lines)
- Build scripts (~200 lines)

**Total:** ~2,300 lines of production-ready code

**Estimated time to generate:** 30-60 minutes  
**Estimated time for you to test:** 2-4 hours  
**Expected result:** Working MMIO-based GPU sharing system

---

**👉 NEXT:** Tell me which path to start with, and I'll generate the code!

Status: ⏳ Waiting for your go-ahead...
