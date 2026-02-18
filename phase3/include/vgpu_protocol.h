#ifndef VGPU_PROTOCOL_H
#define VGPU_PROTOCOL_H

#include <stdint.h>

/* ================================================================
 * MMIO Register Offsets
 * ================================================================ */

#define VGPU_REG_DOORBELL        0x000  /* R/W  Write 1 to submit request       */
#define VGPU_REG_STATUS          0x004  /* RO   Device status                   */
#define VGPU_REG_POOL_ID         0x008  /* RO   Pool ID ('A'=0x41, 'B'=0x42)   */
#define VGPU_REG_PRIORITY        0x00C  /* RO   Priority (0=low,1=med,2=high)   */
#define VGPU_REG_VM_ID           0x010  /* RO   VM identifier                   */
#define VGPU_REG_ERROR_CODE      0x014  /* RO   Error code if STATUS==ERROR     */
#define VGPU_REG_REQUEST_LEN     0x018  /* R/W  Guest writes request length     */
#define VGPU_REG_RESPONSE_LEN    0x01C  /* RO   Host writes response length     */
#define VGPU_REG_PROTOCOL_VER    0x020  /* RO   Protocol version (0x00010000)   */
#define VGPU_REG_CAPABILITIES    0x024  /* RO   Feature bits                    */
#define VGPU_REG_IRQ_CTRL        0x028  /* R/W  Interrupt control               */
#define VGPU_REG_IRQ_STATUS      0x02C  /* RW1C Interrupt status                */
#define VGPU_REG_REQUEST_ID      0x030  /* R/W  Request tracking ID             */
#define VGPU_REG_TIMESTAMP_LO    0x034  /* RO   Completion timestamp low 32     */
#define VGPU_REG_TIMESTAMP_HI    0x038  /* RO   Completion timestamp high 32    */
#define VGPU_REG_SCRATCH         0x03C  /* R/W  Scratch register for testing    */

/* Buffer regions */
#define VGPU_REQ_BUFFER_OFFSET   0x040  /* Request buffer start                 */
#define VGPU_REQ_BUFFER_SIZE     1024   /* Request buffer size in bytes         */
#define VGPU_RESP_BUFFER_OFFSET  0x440  /* Response buffer start                */
#define VGPU_RESP_BUFFER_SIZE    1024   /* Response buffer size in bytes        */
#define VGPU_RESERVED_OFFSET     0x840  /* Reserved region start                */
#define VGPU_BAR_SIZE            4096   /* Total BAR0 size                      */

/* Control register block end */
#define VGPU_CTRL_REG_END        0x040

/* ================================================================
 * STATUS Register Values (offset 0x004)
 * ================================================================ */

#define VGPU_STATUS_IDLE         0x00   /* Ready for new request                */
#define VGPU_STATUS_BUSY         0x01   /* Processing request                   */
#define VGPU_STATUS_DONE         0x02   /* Request completed, response ready    */
#define VGPU_STATUS_ERROR        0x03   /* Request failed, see ERROR_CODE       */

/* ================================================================
 * ERROR_CODE Register Values (offset 0x014)
 * ================================================================ */

#define VGPU_ERR_NONE                 0x00  /* No error                         */
#define VGPU_ERR_INVALID_REQUEST      0x01  /* Request format invalid           */
#define VGPU_ERR_REQUEST_TOO_LARGE    0x02  /* Request exceeds 1024 bytes       */
#define VGPU_ERR_MEDIATOR_UNAVAIL     0x03  /* Cannot connect to mediator       */
#define VGPU_ERR_TIMEOUT              0x04  /* Request processing timeout       */
#define VGPU_ERR_CUDA_ERROR           0x05  /* CUDA execution failed            */
#define VGPU_ERR_INVALID_POOL         0x06  /* Pool ID not available            */
#define VGPU_ERR_QUEUE_FULL           0x07  /* Request queue full               */
#define VGPU_ERR_UNSUPPORTED_OP       0x08  /* Operation not supported          */
#define VGPU_ERR_INVALID_LENGTH       0x09  /* Request length is 0              */
/* Phase 3: Isolation error codes */
#define VGPU_ERR_RATE_LIMITED         0x0A  /* VM hit rate limit (back-pressure)*/
#define VGPU_ERR_VM_QUARANTINED       0x0B  /* VM quarantined (too many faults) */

/* ================================================================
 * CAPABILITIES Register Bits (offset 0x024)
 * ================================================================ */

#define VGPU_CAP_BASIC_REQ       (1 << 0)   /* Basic request/response           */
#define VGPU_CAP_INTERRUPT       (1 << 1)   /* Interrupt notification (future)  */
#define VGPU_CAP_DMA             (1 << 2)   /* DMA transfers (future)           */
#define VGPU_CAP_MULTI_REQ       (1 << 3)   /* Multiple outstanding (future)    */

/* ================================================================
 * PROTOCOL_VER Value (offset 0x020)
 * ================================================================ */

#define VGPU_PROTOCOL_VERSION    0x00010000  /* v1.0 = major<<16 | minor         */

/* ================================================================
 * Request/Response Operation Codes
 * ================================================================ */

#define VGPU_OP_NOP              0x0000  /* No operation (test)                  */
#define VGPU_OP_CUDA_KERNEL      0x0001  /* Execute CUDA kernel (vector add)    */
#define VGPU_OP_GET_DEVICE_INFO  0x0005  /* Query device information            */

/* ================================================================
 * Request Structure (written to MMIO request buffer at 0x040)
 *
 * Guest writes this structure, then sets REQUEST_LEN, then
 * writes 1 to DOORBELL.
 * ================================================================ */

typedef struct __attribute__((packed)) VGPURequest {
    uint32_t version;        /* Protocol version (VGPU_PROTOCOL_VERSION)        */
    uint32_t opcode;         /* Operation code (VGPU_OP_*)                      */
    uint32_t flags;          /* Request flags (0 for now)                       */
    uint32_t param_count;    /* Number of uint32_t parameters following         */
    uint32_t data_offset;    /* Byte offset to variable data (from struct start)*/
    uint32_t data_length;    /* Length of variable data in bytes                */
    uint32_t reserved[2];    /* Must be 0                                       */
    /* params[param_count] follows, then data[data_length] */
} VGPURequest;

#define VGPU_REQUEST_HEADER_SIZE  sizeof(VGPURequest)  /* 32 bytes */

/* Maximum number of uint32_t parameters that fit in request buffer */
#define VGPU_MAX_PARAMS  ((VGPU_REQ_BUFFER_SIZE - VGPU_REQUEST_HEADER_SIZE) / sizeof(uint32_t))

/* ================================================================
 * Response Structure (read from MMIO response buffer at 0x440)
 *
 * Host writes this structure after CUDA completes. Guest reads
 * after STATUS == DONE.
 * ================================================================ */

typedef struct __attribute__((packed)) VGPUResponse {
    uint32_t version;        /* Protocol version (VGPU_PROTOCOL_VERSION)        */
    uint32_t status;         /* 0=success, non-zero=error                       */
    uint32_t result_count;   /* Number of uint32_t results following            */
    uint32_t data_offset;    /* Byte offset to variable data (from struct start)*/
    uint32_t data_length;    /* Length of variable data in bytes                */
    uint32_t exec_time_us;   /* CUDA execution time in microseconds            */
    uint32_t reserved[2];    /* Reserved                                        */
    /* results[result_count] follows, then data[data_length] */
} VGPUResponse;

#define VGPU_RESPONSE_HEADER_SIZE  sizeof(VGPUResponse)  /* 32 bytes */

/* Maximum number of uint32_t results that fit in response buffer */
#define VGPU_MAX_RESULTS  ((VGPU_RESP_BUFFER_SIZE - VGPU_RESPONSE_HEADER_SIZE) / sizeof(uint32_t))

/* ================================================================
 * Socket IPC Protocol (between vgpu-stub and mediator daemon)
 *
 * vgpu-stub connects to mediator via Unix domain socket.
 * Socket: Filesystem Unix socket (inside QEMU chroot directory)
 * The mediator auto-discovers the QEMU chroot and creates the socket there.
 * QEMU (chrooted) sees it at /tmp/vgpu-mediator.sock.
 *
 * Message format (both directions):
 *   [VGPUSocketHeader][payload]
 * ================================================================ */

/* Filesystem Unix socket path (as seen from inside the QEMU chroot) */
#define VGPU_SOCKET_PATH  "/tmp/vgpu-mediator.sock"

/* Message types for socket IPC */
#define VGPU_MSG_REQUEST    0x01   /* vgpu-stub → mediator: new request       */
#define VGPU_MSG_RESPONSE   0x02   /* mediator → vgpu-stub: result ready      */
#define VGPU_MSG_PING       0x03   /* Either direction: keepalive             */
#define VGPU_MSG_PONG       0x04   /* Reply to PING                           */
/* Phase 3: Back-pressure and quarantine notifications */
#define VGPU_MSG_BUSY       0x05   /* mediator → vgpu-stub: rate-limited      */
#define VGPU_MSG_QUARANTINED 0x06  /* mediator → vgpu-stub: VM quarantined    */

typedef struct __attribute__((packed)) VGPUSocketHeader {
    uint32_t magic;          /* 0x56475055 = "VGPU"                            */
    uint32_t msg_type;       /* VGPU_MSG_* type                                */
    uint32_t vm_id;          /* VM identifier                                  */
    uint32_t request_id;     /* Request tracking ID                            */
    char     pool_id;        /* 'A' or 'B'                                     */
    uint8_t  priority;       /* 0=low, 1=medium, 2=high                        */
    uint16_t payload_len;    /* Length of payload following this header         */
} VGPUSocketHeader;

#define VGPU_SOCKET_MAGIC    0x56475055  /* "VGPU" in little-endian            */
#define VGPU_SOCKET_HDR_SIZE sizeof(VGPUSocketHeader)  /* 20 bytes             */

/* Maximum payload size */
#define VGPU_SOCKET_MAX_PAYLOAD  VGPU_REQ_BUFFER_SIZE  /* 1024 bytes          */

/* ================================================================
 * Phase 3: Admin Socket (for vgpu-admin CLI → mediator)
 * ================================================================ */

/* Admin socket path (separate from QEMU chroot sockets) */
#define VGPU_ADMIN_SOCKET_PATH  "/var/vgpu/admin.sock"

/* Admin command codes (sent over admin socket) */
#define VGPU_ADMIN_SHOW_METRICS      0x10
#define VGPU_ADMIN_SHOW_HEALTH       0x11
#define VGPU_ADMIN_RELOAD_CONFIG     0x12
#define VGPU_ADMIN_QUARANTINE_VM     0x13
#define VGPU_ADMIN_UNQUARANTINE_VM   0x14

/* Admin request header */
typedef struct __attribute__((packed)) VGPUAdminRequest {
    uint32_t magic;          /* VGPU_SOCKET_MAGIC                              */
    uint32_t command;        /* VGPU_ADMIN_* code                              */
    uint32_t param1;         /* Command-specific parameter                     */
    uint32_t param2;         /* Command-specific parameter                     */
} VGPUAdminRequest;

/* Admin response: variable-length text follows this header */
typedef struct __attribute__((packed)) VGPUAdminResponse {
    uint32_t magic;          /* VGPU_SOCKET_MAGIC                              */
    uint32_t status;         /* 0 = success, non-zero = error                  */
    uint32_t data_len;       /* Length of text payload that follows             */
} VGPUAdminResponse;

/* ================================================================
 * PCI Device Identification
 * ================================================================ */

#define VGPU_VENDOR_ID       0x1AF4  /* Red Hat, Inc.                          */
#define VGPU_DEVICE_ID       0x1111  /* Custom vGPU stub                       */
#define VGPU_CLASS_ID        0x1200  /* Processing Accelerator                 */
#define VGPU_REVISION        0x02    /* Rev 2: MMIO communication support      */

/* ================================================================
 * Priority Values (shared with mediator)
 * ================================================================ */

#define VGPU_PRIORITY_LOW    0
#define VGPU_PRIORITY_MEDIUM 1
#define VGPU_PRIORITY_HIGH   2

#endif /* VGPU_PROTOCOL_H */
