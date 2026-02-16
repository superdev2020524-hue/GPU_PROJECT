# vGPU PCI/MMIO Communication — Implementation Report

**Author:** Bren  
**Date:** February 13, 2026  
**Status:** Phase 1 Complete — Operational

---

## 1. Background

Following the feedback we received, the direction was clear: retire the file-based
NFS transport between the VM and the host-side mediator, and replace it with a proper
PCI/MMIO communication channel through the vGPU stub device. The request made a lot
of sense — the vGPU stub was already sitting in the guest as a real PCI device with
MMIO registers, but the actual data path was still going through NFS-mounted files,
which was always meant to be temporary scaffolding. The ask was to close that gap and
make the PCI device the real communication endpoint.

We took the three-day window we had and focused entirely on getting this working end
to end. The result is a fully functional PCI/MMIO-based request path that eliminates
NFS from the guest entirely. This report covers what was built, how it works, what
we validated, and where we plan to take it next.

---

## 2. What Was Built

### 2.1 Extended vGPU Stub Device (`vgpu-stub-enhanced.c`)

The vGPU stub was already a PCI device in QEMU, but it only exposed a few read-only
identification registers (pool ID, priority, VM ID). We extended the BAR0 layout to
support bidirectional communication:

**Register map (4KB BAR0):**

| Offset | Register | Access | Purpose |
|--------|----------|--------|---------|
| 0x000 | DOORBELL | W | Guest writes 1 to submit a request |
| 0x004 | STATUS | R | IDLE / BUSY / DONE / ERROR |
| 0x008 | POOL_ID | R | Pool assignment ('A' or 'B') |
| 0x00C | PRIORITY | R | Scheduling priority (0–2) |
| 0x010 | VM_ID | R | VM identifier (set by hypervisor) |
| 0x014 | ERROR_CODE | R | Detailed error when STATUS = ERROR |
| 0x018 | REQUEST_LEN | R/W | Length of request payload in bytes |
| 0x01C | RESPONSE_LEN | R | Length of response payload |
| 0x020 | PROTOCOL_VER | R | Protocol version (v1.0) |
| 0x024 | CAPABILITIES | R | Feature flags |
| 0x028 | IRQ_CTRL | R/W | Interrupt enable (reserved for future) |
| 0x02C | IRQ_STATUS | R/W | Interrupt status (reserved for future) |
| 0x030 | REQUEST_ID | R/W | Request tracking ID |
| 0x040–0x43F | Request buffer | R/W | 1KB — guest writes request payload here |
| 0x440–0x83F | Response buffer | R | 1KB — host writes response payload here |

The key addition is the doorbell/status handshake. The guest writes its request into
the buffer region, sets the request length, and writes 1 to the doorbell register.
The MMIO write handler on the host side picks up the request immediately — no polling,
no filesystem, no network stack. When the result comes back, the host writes it into
the response buffer and sets STATUS to DONE. The guest polls STATUS and reads the
result.

On the host side, the vgpu-stub's MMIO handler forwards the request to the mediator
daemon over a Unix domain socket. This is internal host IPC — the guest never sees it
and doesn't need to know about it. Once the mediator returns a result, the stub writes
it back into the BAR and flips the status register.

### 2.2 Updated VM Client (`vm_client_enhanced.c`)

The guest-side client was rewritten to talk directly to the PCI device instead of NFS.
At startup, it scans sysfs for the vGPU stub by vendor/device ID, mmaps the BAR0
resource, and communicates entirely through register reads and writes. From the guest
OS perspective, this looks like any other PCI device driver interaction — open the
resource file, mmap it, read and write at known offsets.

The NFS mount, file creation, and file polling are all gone. The guest doesn't need
an NFS client, doesn't need network connectivity to the host, and doesn't need to
know anything about the mediator. It just sees a PCI device.

### 2.3 Updated Mediator Daemon (`mediator_enhanced.c`)

The mediator's scheduling logic and CUDA execution path are unchanged — same priority
queue (high > medium > low, FIFO within each level), same async CUDA dispatch, same
result handling. What changed is the input/output side: instead of polling a directory
for request files, the mediator now listens on Unix domain sockets for connections
from vgpu-stub instances running inside QEMU.

Since each QEMU process on XCP-ng runs inside its own chroot jail, the mediator
auto-discovers all running QEMU instances with vgpu-stub devices, creates a listening
socket inside each chroot directory, and multiplexes across all of them using
`select()`. From the mediator's perspective, it's still one process with one queue —
it just has multiple entry points, one per VM.

### 2.4 Shared Protocol (`vgpu_protocol.h`)

All register offsets, status codes, error codes, request/response structures, and
socket message formats are defined in a single shared header. Both the QEMU device
and the mediator include this header, so the protocol stays consistent. The header
also defines capability bits and a protocol version field for forward compatibility.

---

## 3. What Was Validated

We tested with two VMs running concurrently on XCP-ng, each with its own vgpu-stub
device:

- **VM1** (pool A, priority high, VM ID 1): `vm_client_enhanced 100 200` → result 300 ✓
- **VM2** (pool A, priority medium, VM ID 2): `vm_client_enhanced 400 500` → result 900 ✓

Both VMs discovered their PCI devices automatically, communicated exclusively through
MMIO, and received correct results from the GPU. No NFS mounts were present in either
guest. The mediator processed both requests through its priority queue and dispatched
them to CUDA.

---

## 4. Current Architecture

```
  VM 1                              VM 2
  ┌──────────────────┐              ┌──────────────────┐
  │ vm_client        │              │ vm_client        │
  │   │              │              │   │              │
  │   └─ mmap BAR0   │              │   └─ mmap BAR0   │
  │      write regs  │              │      write regs  │
  │      ring bell   │              │      ring bell   │
  │      poll status │              │      poll status │
  │      read result │              │      read result │
  └───────┬──────────┘              └───────┬──────────┘
          │ MMIO                            │ MMIO
          ▼                                 ▼
  ┌──────────────┐                  ┌──────────────┐
  │ QEMU         │                  │ QEMU         │
  │ vgpu-stub    │                  │ vgpu-stub    │
  │ (chroot-73)  │                  │ (chroot-74)  │
  └──────┬───────┘                  └──────┬───────┘
         │ socket                          │ socket
         └────────────┐  ┌────────────────-┘
                      ▼  ▼
               ┌──────────────┐
               │  MEDIATOR    │
               │  (1 process) │
               │  (1 queue)   │
               └──────┬───────┘
                      │ CUDA API
                      ▼
               ┌──────────────┐
               │  NVIDIA GPU  │
               └──────────────┘
```

---

## 5. What Changed from the Previous System

| Before (NFS) | After (PCI/MMIO) |
|---|---|
| Guest mounts NFS share from host | Guest sees a PCI device on its bus |
| Guest writes request files to `/mnt/vgpu/` | Guest writes to PCI BAR registers |
| Mediator polls directory for new files | Mediator receives socket events immediately |
| Response delivered via file | Response delivered via MMIO status + buffer |
| Guest needs NFS client, network, mount | Guest needs nothing — PCI device is just there |
| VM identity encoded in filenames | VM identity encoded in PCI registers (set at boot) |
| Anyone on the NFS share can see other VMs' data | Each VM can only access its own PCI device |

---

## 6. What's Next

The three-day sprint was focused on getting the PCI/MMIO path working and validated.
There are a few areas where we can improve the implementation going forward:

### 6.1 Shared Memory + eventfd (Data Plane Optimization)

Right now, when the guest writes request data into the BAR, the vgpu-stub copies it
into a socket message and sends it to the mediator, which reads it on the other end.
That's three copies of the data. The standard approach for high-throughput virtual
devices (used by vhost-net, SPDK, etc.) is to back the BAR with shared memory
(`memfd`) that both QEMU and the mediator can access directly, and use `eventfd` for
the doorbell notification. This gives zero-copy data transfer and takes QEMU out of
the data path entirely after initial setup.

This is the natural next step and would make the PCI channel truly direct — the
guest's BAR writes would land in memory that the mediator reads without any
intermediate copy or relay.

### 6.2 Interrupt-Driven Completion

The guest currently polls the STATUS register in a loop to detect completion. The
register map already includes IRQ_CTRL and IRQ_STATUS fields (reserved in v1.0).
Adding MSI/MSI-X interrupt support would let the host signal the guest when a result
is ready, eliminating the polling loop and reducing latency.

### 6.3 Multiple Outstanding Requests

The current protocol supports one request at a time per VM (submit, wait, read
result, repeat). For workloads that benefit from pipelining, we could add a small
submission queue in the BAR — similar to how NVMe uses submission and completion
queues — so the guest can have several requests in flight simultaneously.

### 6.4 Larger Payloads via DMA

The 1KB request/response buffers are fine for the current vector-addition workload,
but larger GPU workloads (matrix operations, inference) will need more space. Rather
than growing the BAR, the proper approach is scatter-gather DMA, where the guest
provides physical addresses of larger buffers and the host reads them directly. The
CAPABILITIES register includes a DMA bit for this purpose.

---

## 7. File Inventory

| File | Location | Purpose |
|------|----------|---------|
| `vgpu_protocol.h` | Shared (QEMU + host) | Protocol definitions, register map, data structures |
| `vgpu-stub-enhanced.c` | QEMU source tree | PCI device implementation with MMIO + socket IPC |
| `mediator_enhanced.c` | Host daemon | Socket server, priority queue, CUDA dispatch |
| `cuda_vector_add.c/h` | Host daemon | CUDA kernel for vector addition |
| `vm_client_enhanced.c` | Guest userspace | PCI device discovery and MMIO communication |
| `build_enhanced_qemu.sh` | Build tooling | Integrates vgpu-stub into QEMU RPM for XCP-ng |
| `Makefile` | Build tooling | Builds mediator and guest client |

---

## 8. Summary

The core ask was to retire the NFS-based transport and make the vGPU stub the real
communication endpoint. That's done. The guest now talks to a PCI device through
standard MMIO registers, the mediator receives requests through host-internal IPC,
and all the scheduling and CUDA execution works exactly as before. NFS is no longer
in the picture.

Given the three-day window, the priority was getting a working, validated
implementation rather than optimizing every layer. The guest-side interface is clean
and follows standard device-driver conventions. The host-side IPC (Unix sockets
between QEMU and the mediator) works but can be upgraded to shared memory for
zero-copy performance — that's the clear next step, and the architecture is set up
to support it without changing the guest-side protocol at all.

 Bren
