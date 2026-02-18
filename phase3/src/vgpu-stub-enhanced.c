
#include "qemu/osdep.h"
#include "hw/pci/pci.h"
#include "hw/hw.h"
#include "hw/pci/msi.h"
#include "qemu/timer.h"
#include "qom/object.h"
#include "qemu/module.h"
#include "qemu/main-loop.h"     /* qemu_set_fd_handler */
#include "sysemu/kvm.h"
#include "hw/qdev-properties.h"
#include "qapi/error.h"

#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>

#include "vgpu_protocol.h"

/* ----------------------------------------------------------------
 * QEMU type plumbing
 * ---------------------------------------------------------------- */
#define TYPE_VGPU_STUB  "vgpu-stub"
#define VGPU_STUB(obj)  OBJECT_CHECK(VGPUStubState, (obj), TYPE_VGPU_STUB)

/* ----------------------------------------------------------------
 * Device state
 * ---------------------------------------------------------------- */
typedef struct VGPUStubState {
    PCIDevice parent_obj;

    /* BAR0 MMIO region (4 KB) */
    MemoryRegion mmio;

    /* --- Control registers ------------------------------------ */
    uint32_t status_reg;          /* 0x004 IDLE/BUSY/DONE/ERROR    */
    uint32_t error_code;          /* 0x014 last error code         */
    uint32_t request_len;         /* 0x018 set by guest            */
    uint32_t response_len;        /* 0x01C set by host             */
    uint32_t irq_ctrl;            /* 0x028 interrupt control       */
    uint32_t irq_status;          /* 0x02C interrupt status        */
    uint32_t request_id;          /* 0x030 tracking ID (guest set) */
    uint32_t timestamp_lo;        /* 0x034 completion timestamp    */
    uint32_t timestamp_hi;        /* 0x038 completion timestamp    */
    uint32_t scratch;             /* 0x03C scratch register        */

    /* --- Data buffers ----------------------------------------- */
    uint8_t  req_buf[VGPU_REQ_BUFFER_SIZE];   /* 0x040-0x43F */
    uint8_t  resp_buf[VGPU_RESP_BUFFER_SIZE]; /* 0x440-0x83F */

    /* --- Device properties (set at VM start via QEMU cmdline) - */
    char    *pool_id;             /* "A" or "B"                    */
    char    *priority;            /* "low", "medium", "high"       */
    uint32_t vm_id;               /* unique VM identifier          */

    /* --- Socket to mediator ----------------------------------- */
    int      mediator_fd;         /* -1 when not connected         */

    /* --- Receive buffer for socket (partial reads) ------------ */
    uint8_t  sock_rx_buf[VGPU_SOCKET_HDR_SIZE + VGPU_SOCKET_MAX_PAYLOAD];
    uint32_t sock_rx_len;         /* bytes accumulated so far      */
} VGPUStubState;

/* ================================================================
 * Forward declarations
 * ================================================================ */
static void vgpu_process_doorbell(VGPUStubState *s);
static void vgpu_try_connect_mediator(VGPUStubState *s);
static void vgpu_socket_read_handler(void *opaque);

/* ================================================================
 * Helper: convert priority string → integer
 * ================================================================ */
static uint32_t vgpu_priority_to_int(const char *p)
{
    if (p) {
        if (strcmp(p, "high") == 0)   return VGPU_PRIORITY_HIGH;
        if (strcmp(p, "medium") == 0) return VGPU_PRIORITY_MEDIUM;
    }
    return VGPU_PRIORITY_LOW;
}

/* ================================================================
 * MMIO read handler
 *
 * All accesses are 32-bit aligned.
 * ================================================================ */
static uint64_t vgpu_mmio_read(void *opaque, hwaddr addr, unsigned size)
{
    VGPUStubState *s = opaque;
    uint64_t val = 0;

    /* --- Control registers (0x000 – 0x03F) -------------------- */
    if (addr < VGPU_CTRL_REG_END) {
        switch (addr) {

        case VGPU_REG_DOORBELL:
            /* Doorbell reads as 0 - write-only semantics */
            val = 0;
            break;

        case VGPU_REG_STATUS:
            val = s->status_reg;
            break;

        case VGPU_REG_POOL_ID:
            val = (s->pool_id && s->pool_id[0]) ?
                  (uint32_t)(unsigned char)s->pool_id[0] : (uint32_t)'A';
            break;

        case VGPU_REG_PRIORITY:
            val = vgpu_priority_to_int(s->priority);
            break;

        case VGPU_REG_VM_ID:
            val = s->vm_id;
            break;

        case VGPU_REG_ERROR_CODE:
            val = s->error_code;
            break;

        case VGPU_REG_REQUEST_LEN:
            val = s->request_len;
            break;

        case VGPU_REG_RESPONSE_LEN:
            val = s->response_len;
            break;

        case VGPU_REG_PROTOCOL_VER:
            val = VGPU_PROTOCOL_VERSION;
            break;

        case VGPU_REG_CAPABILITIES:
            val = VGPU_CAP_BASIC_REQ;
            break;

        case VGPU_REG_IRQ_CTRL:
            val = s->irq_ctrl;
            break;

        case VGPU_REG_IRQ_STATUS:
            val = s->irq_status;
            break;

        case VGPU_REG_REQUEST_ID:
            val = s->request_id;
            break;

        case VGPU_REG_TIMESTAMP_LO:
            val = s->timestamp_lo;
            break;

        case VGPU_REG_TIMESTAMP_HI:
            val = s->timestamp_hi;
            break;

        case VGPU_REG_SCRATCH:
            val = s->scratch;
            break;

        default:
            val = 0;
            break;
        }
    }

    /* --- Request buffer (0x040-0x43F) - guest may read back --- */
    else if (addr >= VGPU_REQ_BUFFER_OFFSET &&
             addr < VGPU_REQ_BUFFER_OFFSET + VGPU_REQ_BUFFER_SIZE) {
        uint32_t off = addr - VGPU_REQ_BUFFER_OFFSET;
        if (off + 4 <= VGPU_REQ_BUFFER_SIZE) {
            memcpy(&val, &s->req_buf[off], 4);
        }
    }

    /* --- Response buffer (0x440-0x83F) - guest reads result --- */
    else if (addr >= VGPU_RESP_BUFFER_OFFSET &&
             addr < VGPU_RESP_BUFFER_OFFSET + VGPU_RESP_BUFFER_SIZE) {
        uint32_t off = addr - VGPU_RESP_BUFFER_OFFSET;
        if (off + 4 <= VGPU_RESP_BUFFER_SIZE) {
            memcpy(&val, &s->resp_buf[off], 4);
        }
    }

    /* --- Reserved / unmapped reads as 0 ----------------------- */

    return val;
}

/* ================================================================
 * MMIO write handler
 * ================================================================ */
static void vgpu_mmio_write(void *opaque, hwaddr addr,
                            uint64_t val, unsigned size)
{
    VGPUStubState *s = opaque;

    /* --- Control registers ------------------------------------ */
    if (addr < VGPU_CTRL_REG_END) {
        switch (addr) {

        case VGPU_REG_DOORBELL:
            if (val == 1) {
                vgpu_process_doorbell(s);
            }
            break;

        case VGPU_REG_REQUEST_LEN:
            s->request_len = (uint32_t)val;
            break;

        case VGPU_REG_IRQ_CTRL:
            s->irq_ctrl = (uint32_t)val & 0x01;
            break;

        case VGPU_REG_IRQ_STATUS:
            /* Write-1-to-clear */
            s->irq_status &= ~((uint32_t)val & 0x01);
            break;

        case VGPU_REG_REQUEST_ID:
            s->request_id = (uint32_t)val;
            break;

        case VGPU_REG_SCRATCH:
            s->scratch = (uint32_t)val;
            break;

        default:
            /* Other registers are read-only; silently ignore */
            break;
        }
    }

    /* --- Request buffer (0x040-0x43F) - guest writes request -- */
    else if (addr >= VGPU_REQ_BUFFER_OFFSET &&
             addr < VGPU_REQ_BUFFER_OFFSET + VGPU_REQ_BUFFER_SIZE) {
        uint32_t off = addr - VGPU_REQ_BUFFER_OFFSET;
        if (off + 4 <= VGPU_REQ_BUFFER_SIZE) {
            uint32_t v32 = (uint32_t)val;
            memcpy(&s->req_buf[off], &v32, 4);
        }
    }

    /* Response buffer and reserved region: writes ignored */
}

/* ================================================================
 * MMIO ops structure
 * ================================================================ */
static const MemoryRegionOps vgpu_mmio_ops = {
    .read  = vgpu_mmio_read,
    .write = vgpu_mmio_write,
    .endianness = DEVICE_LITTLE_ENDIAN,
    .impl = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

/* ================================================================
 * Doorbell handler
 *
 * Called when the guest writes 1 to register 0x000.
 * Validates the request and forwards it to the mediator daemon
 * over the Unix socket.
 * ================================================================ */
static void vgpu_process_doorbell(VGPUStubState *s)
{
    VGPUSocketHeader hdr;
    struct iovec iov[2];
    struct msghdr msg;
    ssize_t sent;

    /* Validate request length */
    if (s->request_len == 0) {
        s->status_reg = VGPU_STATUS_ERROR;
        s->error_code = VGPU_ERR_INVALID_LENGTH;
        return;
    }
    if (s->request_len > VGPU_REQ_BUFFER_SIZE) {
        s->status_reg = VGPU_STATUS_ERROR;
        s->error_code = VGPU_ERR_REQUEST_TOO_LARGE;
        return;
    }

    /* Mark device busy */
    s->status_reg = VGPU_STATUS_BUSY;
    s->error_code = VGPU_ERR_NONE;
    s->response_len = 0;

    /* If socket is not connected, try to connect now */
    if (s->mediator_fd < 0) {
        vgpu_try_connect_mediator(s);
    }

    /* Still not connected? → error */
    if (s->mediator_fd < 0) {
        s->status_reg = VGPU_STATUS_ERROR;
        s->error_code = VGPU_ERR_MEDIATOR_UNAVAIL;
        return;
    }

    /* Build socket message header */
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic       = VGPU_SOCKET_MAGIC;
    hdr.msg_type    = VGPU_MSG_REQUEST;
    hdr.vm_id       = s->vm_id;
    hdr.request_id  = s->request_id;
    hdr.pool_id     = (s->pool_id && s->pool_id[0]) ? s->pool_id[0] : 'A';
    hdr.priority    = (uint8_t)vgpu_priority_to_int(s->priority);
    hdr.payload_len = (uint16_t)s->request_len;

    /* Send header + payload in one sendmsg() call */
    iov[0].iov_base = &hdr;
    iov[0].iov_len  = VGPU_SOCKET_HDR_SIZE;
    iov[1].iov_base = s->req_buf;
    iov[1].iov_len  = s->request_len;

    memset(&msg, 0, sizeof(msg));
    msg.msg_iov    = iov;
    msg.msg_iovlen = 2;

    sent = sendmsg(s->mediator_fd, &msg, MSG_NOSIGNAL);
    if (sent < 0) {
        fprintf(stderr, "[vgpu-stub] sendmsg failed: %s\n", strerror(errno));
        /* Socket broken - close and mark error */
        qemu_set_fd_handler(s->mediator_fd, NULL, NULL, NULL);
        close(s->mediator_fd);
        s->mediator_fd = -1;
        s->status_reg  = VGPU_STATUS_ERROR;
        s->error_code  = VGPU_ERR_MEDIATOR_UNAVAIL;
        return;
    }

    /* Request sent.  STATUS stays BUSY until the mediator responds
     * on the socket (handled in vgpu_socket_read_handler). */
}

/* ================================================================
 * Socket: attempt connection to mediator
 * ================================================================ */
static void vgpu_try_connect_mediator(VGPUStubState *s)
{
    struct sockaddr_un addr;
    int fd;

    if (s->mediator_fd >= 0) {
        return;  /* already connected */
    }

    fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        fprintf(stderr, "[vgpu-stub] socket() failed: %s\n", strerror(errno));
        return;
    }

    /* Connect to filesystem Unix socket.
     * QEMU runs in a chroot (e.g. /var/xen/qemu/root-<domid>/),
     * so this path resolves to <chroot>/tmp/vgpu-mediator.sock on the host.
     * The mediator daemon creates the socket there. */
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, VGPU_SOCKET_PATH, sizeof(addr.sun_path) - 1);

    if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
        /* Not an error during startup - mediator might not be running yet */
        close(fd);
        return;
    }

    /* Make non-blocking so QEMU event loop can multiplex */
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK);

    s->mediator_fd  = fd;
    s->sock_rx_len  = 0;

    /* Register with QEMU main-loop so we get called when data arrives */
    qemu_set_fd_handler(fd, vgpu_socket_read_handler, NULL, s);

    fprintf(stderr, "[vgpu-stub] Connected to mediator at %s (fd=%d)\n",
            VGPU_SOCKET_PATH, fd);
}

/* ================================================================
 * Socket: read handler (called by QEMU event loop)
 *
 * The mediator sends back a VGPUSocketHeader + payload.
 * We accumulate bytes until we have a complete message, then
 * copy the payload into the MMIO response buffer and flip STATUS.
 * ================================================================ */
static void vgpu_socket_read_handler(void *opaque)
{
    VGPUStubState *s = opaque;
    ssize_t n;
    VGPUSocketHeader *hdr;
    uint32_t total_len;

    /* Read as much as available */
    n = read(s->mediator_fd,
             s->sock_rx_buf + s->sock_rx_len,
             sizeof(s->sock_rx_buf) - s->sock_rx_len);

    if (n <= 0) {
        if (n == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) {
            /* Connection closed or real error */
            fprintf(stderr, "[vgpu-stub] mediator socket closed (n=%zd)\n", n);
            qemu_set_fd_handler(s->mediator_fd, NULL, NULL, NULL);
            close(s->mediator_fd);
            s->mediator_fd = -1;

            /* If we were waiting for a response, signal error */
            if (s->status_reg == VGPU_STATUS_BUSY) {
                s->status_reg = VGPU_STATUS_ERROR;
                s->error_code = VGPU_ERR_MEDIATOR_UNAVAIL;
            }
        }
        return;
    }

    s->sock_rx_len += (uint32_t)n;

    /* Do we have at least a complete header? */
    if (s->sock_rx_len < VGPU_SOCKET_HDR_SIZE) {
        return;  /* need more data */
    }

    hdr = (VGPUSocketHeader *)s->sock_rx_buf;

    /* Sanity check */
    if (hdr->magic != VGPU_SOCKET_MAGIC) {
        fprintf(stderr, "[vgpu-stub] bad magic 0x%08x, dropping\n", hdr->magic);
        s->sock_rx_len = 0;
        return;
    }

    total_len = VGPU_SOCKET_HDR_SIZE + hdr->payload_len;

    /* Do we have the full message? */
    if (s->sock_rx_len < total_len) {
        return;  /* need more data */
    }

    /* ---- Process complete message ---- */

    if (hdr->msg_type == VGPU_MSG_RESPONSE) {
        uint32_t copy_len = hdr->payload_len;
        if (copy_len > VGPU_RESP_BUFFER_SIZE) {
            copy_len = VGPU_RESP_BUFFER_SIZE;
        }

        /* Copy payload into MMIO response buffer */
        memset(s->resp_buf, 0, VGPU_RESP_BUFFER_SIZE);
        memcpy(s->resp_buf,
               s->sock_rx_buf + VGPU_SOCKET_HDR_SIZE,
               copy_len);

        s->response_len = copy_len;

        /* Set completion timestamp (QEMU virtual clock, microseconds) */
        int64_t now_us = qemu_clock_get_us(QEMU_CLOCK_VIRTUAL);
        s->timestamp_lo = (uint32_t)(now_us & 0xFFFFFFFF);
        s->timestamp_hi = (uint32_t)((uint64_t)now_us >> 32);

        /* Check response status - map mediator error codes to MMIO codes */
        if (copy_len >= VGPU_RESPONSE_HEADER_SIZE) {
            VGPUResponse *resp = (VGPUResponse *)s->resp_buf;
            if (resp->status == 0) {
                s->status_reg = VGPU_STATUS_DONE;
                s->error_code = VGPU_ERR_NONE;
            } else if (resp->status == VGPU_ERR_RATE_LIMITED) {
                /* Phase 3: back-pressure - VM exceeded its rate limit */
                s->status_reg = VGPU_STATUS_ERROR;
                s->error_code = VGPU_ERR_RATE_LIMITED;
                fprintf(stderr,
                        "[vgpu-stub] vm%u req%u: rate-limited by mediator\n",
                        s->vm_id, s->request_id);
            } else if (resp->status == VGPU_ERR_VM_QUARANTINED) {
                /* Phase 3: VM quarantined due to excessive faults */
                s->status_reg = VGPU_STATUS_ERROR;
                s->error_code = VGPU_ERR_VM_QUARANTINED;
                fprintf(stderr,
                        "[vgpu-stub] vm%u req%u: VM quarantined\n",
                        s->vm_id, s->request_id);
            } else if (resp->status == VGPU_ERR_QUEUE_FULL) {
                /* Queue depth exceeded */
                s->status_reg = VGPU_STATUS_ERROR;
                s->error_code = VGPU_ERR_QUEUE_FULL;
                fprintf(stderr,
                        "[vgpu-stub] vm%u req%u: queue full\n",
                        s->vm_id, s->request_id);
            } else {
                /* Generic CUDA or other error */
                s->status_reg = VGPU_STATUS_ERROR;
                s->error_code = VGPU_ERR_CUDA_ERROR;
            }
        } else {
            s->status_reg = VGPU_STATUS_DONE;
            s->error_code = VGPU_ERR_NONE;
        }

        /* If interrupt enabled, raise it (future enhancement) */
        /* For now we just rely on guest polling STATUS. */
    }
    else if (hdr->msg_type == VGPU_MSG_BUSY) {
        /* Phase 3: mediator signals rate-limit rejection as a distinct
         * message type (no payload).  Map to MMIO error code. */
        s->status_reg = VGPU_STATUS_ERROR;
        s->error_code = VGPU_ERR_RATE_LIMITED;
        s->response_len = 0;
        fprintf(stderr,
                "[vgpu-stub] vm%u req%u: BUSY (rate-limited)\n",
                s->vm_id, s->request_id);
    }
    else if (hdr->msg_type == VGPU_MSG_QUARANTINED) {
        /* Phase 3: mediator signals VM quarantine as a distinct
         * message type (no payload).  Map to MMIO error code. */
        s->status_reg = VGPU_STATUS_ERROR;
        s->error_code = VGPU_ERR_VM_QUARANTINED;
        s->response_len = 0;
        fprintf(stderr,
                "[vgpu-stub] vm%u req%u: VM QUARANTINED\n",
                s->vm_id, s->request_id);
    }
    else if (hdr->msg_type == VGPU_MSG_PING) {
        /* Reply with PONG - keeps connection alive */
        VGPUSocketHeader pong;
        ssize_t sent;
        memset(&pong, 0, sizeof(pong));
        pong.magic    = VGPU_SOCKET_MAGIC;
        pong.msg_type = VGPU_MSG_PONG;
        pong.vm_id    = s->vm_id;
        sent = write(s->mediator_fd, &pong, VGPU_SOCKET_HDR_SIZE);
        if (sent < 0) {
            /* Connection may be broken, but don't error out on keepalive */
            /* The next real request will detect the failure */
        }
    }
    /* else: ignore unknown message types */

    /* Consume the processed message from rx buffer.
     * If there are trailing bytes from a next message, shift them. */
    if (s->sock_rx_len > total_len) {
        memmove(s->sock_rx_buf,
                s->sock_rx_buf + total_len,
                s->sock_rx_len - total_len);
        s->sock_rx_len -= total_len;
    } else {
        s->sock_rx_len = 0;
    }
}

/* ================================================================
 * Device realisation  (called when QEMU creates the device)
 * ================================================================ */
static void vgpu_realize(PCIDevice *pci_dev, Error **errp)
{
    VGPUStubState *s = VGPU_STUB(pci_dev);

    /* Interrupt pin A (for future MSI support) */
    pci_dev->config[PCI_INTERRUPT_PIN] = 1;

    /* Register BAR0 - 4 KB MMIO */
    memory_region_init_io(&s->mmio, OBJECT(s), &vgpu_mmio_ops, s,
                          "vgpu-stub-mmio", VGPU_BAR_SIZE);
    pci_register_bar(pci_dev, 0, PCI_BASE_ADDRESS_SPACE_MEMORY, &s->mmio);

    /* Initialise control registers */
    s->status_reg   = VGPU_STATUS_IDLE;
    s->error_code   = VGPU_ERR_NONE;
    s->request_len  = 0;
    s->response_len = 0;
    s->irq_ctrl     = 0;
    s->irq_status   = 0;
    s->request_id   = 0;
    s->timestamp_lo = 0;
    s->timestamp_hi = 0;
    s->scratch      = 0;

    /* Clear buffers */
    memset(s->req_buf,  0, VGPU_REQ_BUFFER_SIZE);
    memset(s->resp_buf, 0, VGPU_RESP_BUFFER_SIZE);

    /* Socket not yet connected */
    s->mediator_fd  = -1;
    s->sock_rx_len  = 0;

    /* Validate pool_id property */
    if (!s->pool_id || strlen(s->pool_id) == 0) {
        g_free(s->pool_id);
        s->pool_id = g_strdup("A");
    } else {
        if (strcmp(s->pool_id, "A") != 0 && strcmp(s->pool_id, "B") != 0) {
            error_setg(errp,
                       "vgpu-stub: pool_id must be 'A' or 'B', got '%s'",
                       s->pool_id);
            return;
        }
    }

    /* Validate priority property */
    if (!s->priority || strlen(s->priority) == 0) {
        g_free(s->priority);
        s->priority = g_strdup("medium");
    } else {
        if (strcmp(s->priority, "low") != 0 &&
            strcmp(s->priority, "medium") != 0 &&
            strcmp(s->priority, "high") != 0) {
            error_setg(errp,
                       "vgpu-stub: priority must be 'low', 'medium' "
                       "or 'high', got '%s'",
                       s->priority);
            return;
        }
    }

    fprintf(stderr,
            "[vgpu-stub] realised  vm_id=%u  pool=%s  priority=%s  rev=0x%02x\n",
            s->vm_id,
            s->pool_id  ? s->pool_id  : "A",
            s->priority ? s->priority : "medium",
            VGPU_REVISION);

    /* Try to connect to mediator (may not be running yet; that is fine) */
    vgpu_try_connect_mediator(s);
}

/* ================================================================
 * Device cleanup
 * ================================================================ */
static void vgpu_exit(PCIDevice *pci_dev)
{
    VGPUStubState *s = VGPU_STUB(pci_dev);

    if (s->mediator_fd >= 0) {
        qemu_set_fd_handler(s->mediator_fd, NULL, NULL, NULL);
        close(s->mediator_fd);
        s->mediator_fd = -1;
    }

    g_free(s->pool_id);
    g_free(s->priority);

    fprintf(stderr, "[vgpu-stub] destroyed  vm_id=%u\n", s->vm_id);
}

/* ================================================================
 * Properties exposed on the QEMU command line
 *
 * Example:
 *   -device vgpu-stub,pool_id=B,priority=high,vm_id=200
 * ================================================================ */
static Property vgpu_properties[] = {
    DEFINE_PROP_STRING("pool_id",  VGPUStubState, pool_id),
    DEFINE_PROP_STRING("priority", VGPUStubState, priority),
    DEFINE_PROP_UINT32("vm_id",    VGPUStubState, vm_id, 0),
    DEFINE_PROP_END_OF_LIST(),
};

/* ================================================================
 * PCI class initialisation
 * ================================================================ */
static void vgpu_class_init(ObjectClass *klass, void *data)
{
    DeviceClass    *dc = DEVICE_CLASS(klass);
    PCIDeviceClass *k  = PCI_DEVICE_CLASS(klass);

    k->realize   = vgpu_realize;
    k->exit      = vgpu_exit;
    k->vendor_id = VGPU_VENDOR_ID;
    k->device_id = VGPU_DEVICE_ID;
    k->revision  = VGPU_REVISION;
    k->class_id  = VGPU_CLASS_ID;   /* Processing Accelerator - NOT VGA */

    set_bit(DEVICE_CATEGORY_MISC, dc->categories);
    dc->desc  = "vGPU Stub v2 (MMIO + Phase 3 Isolation)";
    dc->props = vgpu_properties;
}

/* ================================================================
 * Type registration
 * ================================================================ */
static const TypeInfo vgpu_info = {
    .name          = TYPE_VGPU_STUB,
    .parent        = TYPE_PCI_DEVICE,
    .instance_size = sizeof(VGPUStubState),
    .class_init    = vgpu_class_init,
    .interfaces    = (InterfaceInfo[]) {
        { INTERFACE_CONVENTIONAL_PCI_DEVICE },
        { },
    },
};

static void vgpu_register_types(void)
{
    type_register_static(&vgpu_info);
}

type_init(vgpu_register_types)
