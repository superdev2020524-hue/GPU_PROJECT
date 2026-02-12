# vGPU Stub Changes Required

**File:** `hw/misc/vgpu-stub.c` (currently in QEMU source)  
**Current Version:** From complete.txt (lines 109-362)  
**Target Version:** Enhanced with MMIO communication

---

## Summary of Changes

| Component | Current | New | Impact |
|-----------|---------|-----|--------|
| BAR size | 4KB | 4KB (same) | No change |
| Register count | 5 registers | 16 registers | +11 registers |
| Buffers | None | 2x 1KB buffers | Request/Response |
| Communication | None | Socket to mediator | New feature |
| Revision | 0x01 | 0x02 | Version bump |

---

## Change 1: Update VGPUStubState Structure

**Current (lines 145-159):**
```c
typedef struct VGPUStubState {
    PCIDevice parent_obj;
    
    MemoryRegion mmio;
    
    uint32_t command_reg;
    uint32_t status_reg;
    
    char *pool_id;
    char *priority;
    uint32_t vm_id;
} VGPUStubState;
```

**New (enhanced):**
```c
typedef struct VGPUStubState {
    PCIDevice parent_obj;
    
    MemoryRegion mmio;
    
    /* Control registers */
    uint32_t doorbell_reg;       // 0x000: Write 1 to submit request
    uint32_t status_reg;         // 0x004: IDLE/BUSY/DONE/ERROR
    uint32_t error_code;         // 0x014: Error code if STATUS==ERROR
    uint32_t request_len;        // 0x018: Guest writes request length
    uint32_t response_len;       // 0x01C: Host writes response length
    uint32_t interrupt_ctrl;     // 0x028: Interrupt control
    uint32_t interrupt_status;   // 0x02C: Interrupt status
    uint32_t request_id;         // 0x030: Request tracking ID
    uint64_t completion_time;    // 0x034/0x038: Timestamp
    uint32_t scratch;            // 0x03C: Scratch register
    
    /* Request/response buffers */
    uint8_t request_buffer[1024];   // 0x040-0x43F
    uint8_t response_buffer[1024];  // 0x440-0x83F
    
    /* Properties (unchanged) */
    char *pool_id;
    char *priority;
    uint32_t vm_id;
    
    /* Mediator communication */
    int mediator_fd;             // Unix socket to mediator
    QEMUTimer *response_timer;   // For async response handling
} VGPUStubState;
```

**Why:** Need to track new registers and buffers for communication protocol

---

## Change 2: Extend vgpu_mmio_read() Handler

**Current (lines 165-202):**
```c
static uint64_t vgpu_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    VGPUStubState *s = opaque;
    uint64_t val = 0;
    
    switch (addr) {
    case 0x000:  /* Command register */
        val = s->command_reg;
        break;
    case 0x004:  /* Status register */
        val = s->status_reg;
        break;
    case 0x008:  /* Pool ID register */
        // ... (existing code)
        break;
    case 0x00C:  /* Priority register */
        // ... (existing code)
        break;
    case 0x010:  /* VM ID register */
        val = s->vm_id;
        break;
    default:
        val = 0;
        break;
    }
    
    return val;
}
```

**New (enhanced):**
```c
static uint64_t vgpu_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    VGPUStubState *s = opaque;
    uint64_t val = 0;
    
    // Control registers (0x000-0x03F)
    if (addr < 0x040) {
        switch (addr) {
        case 0x000:  /* Doorbell (reads as 0) */
            val = 0;
            break;
        case 0x004:  /* Status */
            val = s->status_reg;
            break;
        case 0x008:  /* Pool ID */
            if (s->pool_id && strlen(s->pool_id) > 0) {
                val = (uint32_t)(s->pool_id[0]);
            } else {
                val = (uint32_t)'A';
            }
            break;
        case 0x00C:  /* Priority */
            if (s->priority) {
                if (strcmp(s->priority, "high") == 0) val = 2;
                else if (strcmp(s->priority, "medium") == 0) val = 1;
                else val = 0;
            } else {
                val = 1;
            }
            break;
        case 0x010:  /* VM ID */
            val = s->vm_id;
            break;
        case 0x014:  /* Error code */
            val = s->error_code;
            break;
        case 0x018:  /* Request length */
            val = s->request_len;
            break;
        case 0x01C:  /* Response length */
            val = s->response_len;
            break;
        case 0x020:  /* Protocol version */
            val = 0x00010000;  // v1.0
            break;
        case 0x024:  /* Capabilities */
            val = 0x00000001;  // Basic request/response
            break;
        case 0x028:  /* Interrupt control */
            val = s->interrupt_ctrl;
            break;
        case 0x02C:  /* Interrupt status */
            val = s->interrupt_status;
            break;
        case 0x030:  /* Request ID */
            val = s->request_id;
            break;
        case 0x034:  /* Timestamp low */
            val = (uint32_t)(s->completion_time & 0xFFFFFFFF);
            break;
        case 0x038:  /* Timestamp high */
            val = (uint32_t)(s->completion_time >> 32);
            break;
        case 0x03C:  /* Scratch */
            val = s->scratch;
            break;
        default:
            val = 0;
            break;
        }
    }
    // Request buffer (0x040-0x43F) - read back what guest wrote
    else if (addr >= 0x040 && addr < 0x440) {
        uint32_t offset = addr - 0x040;
        if (size == 4 && offset < 1024) {
            val = *(uint32_t*)(s->request_buffer + offset);
        }
    }
    // Response buffer (0x440-0x83F) - host writes, guest reads
    else if (addr >= 0x440 && addr < 0x840) {
        uint32_t offset = addr - 0x440;
        if (size == 4 && offset < 1024) {
            val = *(uint32_t*)(s->response_buffer + offset);
        }
    }
    // Reserved (0x840-0xFFF)
    else {
        val = 0;
    }
    
    return val;
}
```

**Why:** Support reading all new registers and response buffer

---

## Change 3: Extend vgpu_mmio_write() Handler

**Current (lines 208-222):**
```c
static void vgpu_mmio_write(void *opaque, hwaddr addr,
                            uint64_t val, unsigned size)
{
    VGPUStubState *s = opaque;
    
    switch (addr) {
    case 0x000:  /* Command register */
        s->command_reg = val;
        s->status_reg = 0x1;
        break;
    default:
        /* Other registers read-only */
        break;
    }
}
```

**New (enhanced):**
```c
static void vgpu_mmio_write(void *opaque, hwaddr addr,
                            uint64_t val, unsigned size)
{
    VGPUStubState *s = opaque;
    
    // Control registers
    if (addr < 0x040) {
        switch (addr) {
        case 0x000:  /* Doorbell - triggers request processing */
            if (val == 1) {
                vgpu_process_doorbell(s);  // NEW FUNCTION
            }
            break;
        case 0x018:  /* Request length */
            if (val <= 1024) {
                s->request_len = val;
            }
            break;
        case 0x028:  /* Interrupt control */
            s->interrupt_ctrl = val & 0x01;  // Only bit 0 valid
            break;
        case 0x02C:  /* Interrupt status (write 1 to clear) */
            s->interrupt_status &= ~(val & 0x01);
            break;
        case 0x030:  /* Request ID */
            s->request_id = val;
            break;
        case 0x03C:  /* Scratch */
            s->scratch = val;
            break;
        default:
            /* Other registers read-only */
            break;
        }
    }
    // Request buffer (0x040-0x43F) - guest writes request here
    else if (addr >= 0x040 && addr < 0x440) {
        uint32_t offset = addr - 0x040;
        if (size == 4 && offset < 1024) {
            *(uint32_t*)(s->request_buffer + offset) = val;
        }
    }
    // Response buffer is read-only for guest
    // Reserved regions are read-only
}
```

**Why:** Handle doorbell writes and request buffer writes

---

## Change 4: Add New Function - vgpu_process_doorbell()

**New function (add after vgpu_mmio_write):**
```c
/*
 * Process Doorbell Ring
 * Called when guest writes 1 to doorbell register
 * Forwards request to mediator daemon
 */
static void vgpu_process_doorbell(VGPUStubState *s)
{
    // Validate request length
    if (s->request_len == 0 || s->request_len > 1024) {
        s->status_reg = 0x03;  // ERROR
        s->error_code = 0x02;   // REQUEST_TOO_LARGE or INVALID
        return;
    }
    
    // Set status to BUSY
    s->status_reg = 0x01;
    s->error_code = 0x00;
    
    // Send to mediator (if connected)
    if (s->mediator_fd >= 0) {
        vgpu_send_to_mediator(s);  // NEW FUNCTION
    } else {
        // No mediator - set error
        s->status_reg = 0x03;  // ERROR
        s->error_code = 0x03;   // MEDIATOR_UNAVAILABLE
    }
}
```

---

## Change 5: Add New Function - vgpu_send_to_mediator()

**New function:**
```c
/*
 * Send Request to Mediator
 * Sends request over Unix socket
 */
static void vgpu_send_to_mediator(VGPUStubState *s)
{
    // Create request packet
    struct {
        uint32_t vm_id;
        uint32_t request_id;
        uint32_t length;
        char pool_id;
        uint8_t priority;
        uint8_t reserved[2];
        uint8_t data[1024];
    } packet;
    
    packet.vm_id = s->vm_id;
    packet.request_id = s->request_id;
    packet.length = s->request_len;
    packet.pool_id = s->pool_id ? s->pool_id[0] : 'A';
    
    if (s->priority) {
        if (strcmp(s->priority, "high") == 0) packet.priority = 2;
        else if (strcmp(s->priority, "medium") == 0) packet.priority = 1;
        else packet.priority = 0;
    } else {
        packet.priority = 1;
    }
    
    memcpy(packet.data, s->request_buffer, s->request_len);
    
    // Send via socket
    ssize_t sent = send(s->mediator_fd, &packet, 
                        sizeof(packet) - 1024 + s->request_len, 0);
    
    if (sent < 0) {
        s->status_reg = 0x03;  // ERROR
        s->error_code = 0x03;   // MEDIATOR_UNAVAILABLE
        return;
    }
    
    // For now, status stays BUSY until response arrives
    // Real implementation: register callback for async response
}
```

---

## Change 6: Add New Function - vgpu_connect_mediator()

**New function (called during realize):**
```c
/*
 * Connect to Mediator Daemon
 * Establishes Unix socket connection
 */
static int vgpu_connect_mediator(VGPUStubState *s)
{
    struct sockaddr_un addr;
    int fd;
    
    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }
    
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    snprintf(addr.sun_path, sizeof(addr.sun_path),
             "/tmp/vgpu-mediator-%u.sock", s->vm_id);
    
    if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(fd);
        return -1;
    }
    
    s->mediator_fd = fd;
    return 0;
}
```

---

## Change 7: Update vgpu_realize()

**Current (lines 244-287):**
```c
static void vgpu_realize(PCIDevice *pci_dev, Error **errp)
{
    VGPUStubState *s = VGPU_STUB(pci_dev);
    
    pci_dev->config[PCI_INTERRUPT_PIN] = 1;
    
    memory_region_init_io(&s->mmio, OBJECT(s), &vgpu_mmio_ops, s,
                          "vgpu-stub-mmio", 4096);
    pci_register_bar(pci_dev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);
    
    s->command_reg = 0;
    s->status_reg = 0;
    
    // ... property validation ...
}
```

**New (add initialization):**
```c
static void vgpu_realize(PCIDevice *pci_dev, Error **errp)
{
    VGPUStubState *s = VGPU_STUB(pci_dev);
    
    pci_dev->config[PCI_INTERRUPT_PIN] = 1;
    
    memory_region_init_io(&s->mmio, OBJECT(s), &vgpu_mmio_ops, s,
                          "vgpu-stub-mmio", 4096);
    pci_register_bar(pci_dev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);
    
    // Initialize all registers
    s->doorbell_reg = 0;
    s->status_reg = 0;      // IDLE
    s->error_code = 0;
    s->request_len = 0;
    s->response_len = 0;
    s->interrupt_ctrl = 0;
    s->interrupt_status = 0;
    s->request_id = 0;
    s->completion_time = 0;
    s->scratch = 0;
    
    // Clear buffers
    memset(s->request_buffer, 0, 1024);
    memset(s->response_buffer, 0, 1024);
    
    // ... existing property validation ...
    
    // Connect to mediator (optional - may not exist yet)
    s->mediator_fd = -1;
    vgpu_connect_mediator(s);  // Ignore error for now
}
```

---

## Change 8: Update vgpu_exit()

**Current (lines 292-297):**
```c
static void vgpu_exit(PCIDevice *pci_dev)
{
    VGPUStubState *s = VGPU_STUB(pci_dev);
    g_free(s->pool_id);
    g_free(s->priority);
}
```

**New (add socket cleanup):**
```c
static void vgpu_exit(PCIDevice *pci_dev)
{
    VGPUStubState *s = VGPU_STUB(pci_dev);
    
    // Close mediator socket
    if (s->mediator_fd >= 0) {
        close(s->mediator_fd);
        s->mediator_fd = -1;
    }
    
    g_free(s->pool_id);
    g_free(s->priority);
}
```

---

## Change 9: Update PCI Revision

**Current (line 329):**
```c
k->revision = 0x01;
```

**New:**
```c
k->revision = 0x02;  // Incremented to indicate MMIO comm support
```

---

## Change 10: Add Required Headers

**Add to includes (after line 136):**
```c
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
```

---

## Summary of Line Count Changes

| Metric | Current | New | Change |
|--------|---------|-----|--------|
| Lines of code | ~250 | ~450 | +200 lines |
| Functions | 6 | 10 | +4 functions |
| Registers | 5 | 16 | +11 registers |
| Buffers | 0 | 2 | +2 KB RAM |

---

## Build Impact

**Files to modify:**
1. `hw/misc/vgpu-stub.c` - main changes above
2. No changes to `hw/misc/Makefile.objs` - already includes vgpu-stub.o
3. No changes to `qemu.spec` - already has our patches

**Build time:** Same as before (~30-45 minutes)

**Compatibility:** 
- Old VMs continue to work (backward compatible)
- New features only active when mediator connected
- Graceful degradation if mediator unavailable

---

## Testing the Changes

### Test 1: Verify Compilation
```bash
cd ~/vgpu-build/rpmbuild
rpmbuild -bb SPECS/qemu.spec
# Should succeed without errors
```

### Test 2: Verify New Registers
```c
// In guest VM
volatile uint32_t *mmio = mmap(...);
printf("Protocol version: 0x%08x\n", mmio[0x020/4]);  // Should be 0x00010000
printf("Capabilities: 0x%08x\n", mmio[0x024/4]);       // Should be 0x00000001
```

### Test 3: Test Doorbell (without mediator)
```c
// In guest VM
mmio[0x018/4] = 16;  // Set request length
mmio[0x000/4] = 1;   // Ring doorbell
usleep(1000);
printf("Status: %u\n", mmio[0x004/4]);      // Should be 3 (ERROR)
printf("Error code: %u\n", mmio[0x014/4]);  // Should be 3 (MEDIATOR_UNAVAILABLE)
```

---

## Next Steps After This Change

1. ✅ Build and install modified QEMU
2. ✅ Test new registers accessible from guest
3. ⏳ Create mediator socket interface
4. ⏳ Test end-to-end request flow
5. ⏳ Create guest client library
6. ⏳ Integration testing

---

**Ready to implement these changes? Let me know and I'll create the complete enhanced vgpu-stub.c file!**
