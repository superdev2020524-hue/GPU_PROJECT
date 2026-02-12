# MMIO vs NFS Communication - Detailed Explanation

## Overview

This document explains how the new MMIO-based system works compared to the old NFS-based system, with a focus on how each VM's requests and responses are handled.

---

## NFS System (Old)

### How it worked:

```
┌─────────────────────────────────────────────────────────────┐
│                    NFS File System                          │
│  /var/vgpu/                                                  │
│  ├── vm1/                                                    │
│  │   ├── request.txt   ← VM1 writes here                    │
│  │   └── response.txt  ← Mediator writes here              │
│  ├── vm2/                                                    │
│  │   ├── request.txt   ← VM2 writes here                    │
│  │   └── response.txt  ← Mediator writes here              │
│  └── vm3/                                                    │
│      ├── request.txt   ← VM3 writes here                    │
│      └── response.txt  ← Mediator writes here              │
└─────────────────────────────────────────────────────────────┘
```

**Key Points:**
- Each VM had its **own directory** (`vm1/`, `vm2/`, etc.)
- Each VM had its **own files** (`request.txt`, `response.txt`)
- Mediator **polled all directories** every second
- Mediator **read from** `vmX/request.txt` and **wrote to** `vmX/response.txt`
- **Separation**: Files in different directories = different VMs

---

## MMIO System (New)

### How it works:

```
┌─────────────────────────────────────────────────────────────┐
│                    VM1 (Guest)                               │
│  ┌──────────────────────────────────────┐                   │
│  │  vGPU Stub Device (PCI)               │                   │
│  │  ┌────────────────────────────────┐  │                   │
│  │  │  MMIO BAR0 (4KB)                │  │                   │
│  │  │  ┌──────────────────────────┐  │  │                   │
│  │  │  │ Request Buffer (0x040)    │  │  │                   │
│  │  │  │ Response Buffer (0x440)   │  │  │                   │
│  │  │  └──────────────────────────┘  │  │                   │
│  │  └────────────────────────────────┘  │                   │
│  │         │                              │                   │
│  │         │ Unix Socket                  │                   │
│  │         ▼                              │                   │
│  └─────────┼──────────────────────────────┘                   │
│            │                                                   │
┌────────────┼─────────────────────────────────────────────────┐
│            │                                                   │
│  ┌─────────▼──────────────────────────────────────────────┐   │
│  │  VM2 (Guest)                                           │   │
│  │  ┌──────────────────────────────────────┐             │   │
│  │  │  vGPU Stub Device (PCI)               │             │   │
│  │  │  ┌────────────────────────────────┐  │             │   │
│  │  │  │  MMIO BAR0 (4KB)                │  │             │   │
│  │  │  │  ┌──────────────────────────┐  │  │             │   │
│  │  │  │  │ Request Buffer (0x040)    │  │  │             │   │
│  │  │  │  │ Response Buffer (0x440)   │  │  │             │   │
│  │  │  │  └──────────────────────────┘  │  │             │   │
│  │  │  └────────────────────────────────┘  │             │   │
│  │  │         │                              │             │   │
│  │  │         │ Unix Socket                  │             │   │
│  │  │         ▼                              │             │   │
│  │  └─────────┼──────────────────────────────┘             │   │
│  │            │                                             │   │
│  └────────────┼─────────────────────────────────────────────┘   │
│               │                                                   │
│               │  All connect to                                │
│               │  SAME socket                                     │
│               ▼                                                   │
│  ┌──────────────────────────────────────────────────────────┐    │
│  │  Mediator Daemon (Host/Dom0)                              │    │
│  │  ┌────────────────────────────────────────────────────┐  │    │
│  │  │  Unix Socket Server                                │  │    │
│  │  │  /tmp/vgpu-mediator.sock                           │  │    │
│  │  │                                                      │  │    │
│  │  │  Receives: VGPUSocketHeader (contains vm_id)       │  │    │
│  │  │  Sends: Response with same vm_id                   │  │    │
│  │  └────────────────────────────────────────────────────┘  │    │
│  └──────────────────────────────────────────────────────────┘    │
└───────────────────────────────────────────────────────────────────┘
```

**Key Points:**
- Each VM has its **own vGPU stub PCI device**
- Each vGPU stub device has its **own MMIO memory** (BAR0, 4KB)
- Each vGPU stub has its **own request buffer** (offset 0x040) and **response buffer** (offset 0x440)
- **BUT** - all vGPU stub devices connect to the **SAME mediator socket** (`/tmp/vgpu-mediator.sock`)
- **Separation**: Each message includes `vm_id` in the header, so the mediator knows which VM it's from

---

## Detailed Comparison

### Request Flow

#### NFS System:
1. **VM1** writes to `/mnt/vgpu/vm1/request.txt`: `"A:2:1:10:20"`
2. **VM2** writes to `/mnt/vgpu/vm2/request.txt`: `"B:1:2:30:40"`
3. Mediator polls `/var/vgpu/`, finds both files
4. Mediator reads `vm1/request.txt` → knows it's from VM1 (from directory name)
5. Mediator reads `vm2/request.txt` → knows it's from VM2 (from directory name)

#### MMIO System:
1. **VM1** writes request to its **own MMIO request buffer** (0x040-0x43F)
2. **VM1** rings doorbell → vGPU stub sends to `/tmp/vgpu-mediator.sock`
   - Message includes: `vm_id=1` in the header
3. **VM2** writes request to its **own MMIO request buffer** (0x040-0x43F)
4. **VM2** rings doorbell → vGPU stub sends to `/tmp/vgpu-mediator.sock`
   - Message includes: `vm_id=2` in the header
5. Mediator receives both messages on the **same socket**
6. Mediator looks at `vm_id` in the header to know which VM sent it

### Response Flow

#### NFS System:
1. Mediator writes result to `/var/vgpu/vm1/response.txt`: `"30"`
2. Mediator writes result to `/var/vgpu/vm2/response.txt`: `"70"`
3. **VM1** polls `/mnt/vgpu/vm1/response.txt` → reads `"30"`
4. **VM2** polls `/mnt/vgpu/vm2/response.txt` → reads `"70"`

#### MMIO System:
1. Mediator sends response to socket with `vm_id=1` in header
2. **VM1's vGPU stub** receives it (because it's still connected)
3. vGPU stub writes result to **VM1's MMIO response buffer** (0x440-0x83F)
4. **VM1** reads from its **own MMIO response buffer**
5. Same process for VM2 - mediator sends with `vm_id=2`, VM2's stub receives it

---

## Key Differences

| Aspect | NFS System | MMIO System |
|--------|------------|-------------|
| **Request Storage** | Each VM has separate file | Each VM has separate MMIO buffer |
| **Response Storage** | Each VM has separate file | Each VM has separate MMIO buffer |
| **Connection** | No connection (files) | All VMs connect to same socket |
| **VM Identification** | Directory name (`vm1/`, `vm2/`) | `vm_id` field in message header |
| **Mediator Access** | Polls all directories | Receives on single socket |
| **Separation** | Physical (different files) | Logical (different buffers + vm_id) |

---

## How VM Separation Works in MMIO

### Each VM Has:
1. **Its own PCI device** - Each VM gets a separate vGPU stub device
2. **Its own MMIO memory** - Each device has its own 4KB BAR0
3. **Its own buffers** - Each device has its own request/response buffers
4. **Its own connection** - Each vGPU stub opens its own socket connection

### But They Share:
- **The same mediator socket** - All connections go to `/tmp/vgpu-mediator.sock`

### How Mediator Knows Which VM:
- Every message includes `vm_id` in the `VGPUSocketHeader`
- When mediator sends response, it includes the same `vm_id`
- Each vGPU stub only processes messages with its own `vm_id` (or all messages if it's the only one connected)

Actually, wait - let me check the code... The vGPU stub doesn't filter by vm_id on receive. It just receives whatever the mediator sends. But since each vGPU stub has its own connection, the mediator sends the response to the correct connection (the one that sent the request).

---

## Example: Two VMs Making Requests

### Scenario:
- **VM1** (vm_id=1): Wants to compute 10+20
- **VM2** (vm_id=2): Wants to compute 30+40

### Step-by-Step:

1. **VM1** writes to its MMIO request buffer:
   - Address: `BAR0 + 0x040` (VM1's own buffer)
   - Data: Request structure with num1=10, num2=20

2. **VM1** rings doorbell → **VM1's vGPU stub** sends:
   ```
   Socket message to /tmp/vgpu-mediator.sock:
   - Header: vm_id=1, pool_id='A', priority=2
   - Payload: Request with 10+20
   ```

3. **VM2** writes to its MMIO request buffer:
   - Address: `BAR0 + 0x040` (VM2's own buffer - DIFFERENT from VM1)
   - Data: Request structure with num1=30, num2=40

4. **VM2** rings doorbell → **VM2's vGPU stub** sends:
   ```
   Socket message to /tmp/vgpu-mediator.sock:
   - Header: vm_id=2, pool_id='B', priority=1
   - Payload: Request with 30+40
   ```

5. **Mediator** receives both messages (on same socket, but different connections):
   - Connection 1: vm_id=1, request=10+20
   - Connection 2: vm_id=2, request=30+40

6. **Mediator** processes requests, sends responses:
   - Sends to Connection 1: vm_id=1, result=30
   - Sends to Connection 2: vm_id=2, result=70

7. **VM1's vGPU stub** receives response → writes to **VM1's MMIO response buffer** (0x440)
8. **VM2's vGPU stub** receives response → writes to **VM2's MMIO response buffer** (0x440)

9. **VM1** reads from its own buffer → gets 30
10. **VM2** reads from its own buffer → gets 70

---

## Summary

**Yes, each VM has its own independent request and response addresses!**

- Each VM has its **own PCI device** with **own MMIO memory**
- Each device has its **own request buffer** (0x040) and **own response buffer** (0x440)
- These are **physically separate** memory regions (different PCI devices)
- The mediator distinguishes VMs by the `vm_id` in the message header
- Each vGPU stub maintains its **own socket connection** to the mediator

The key insight: **Separation is achieved through separate PCI devices and MMIO memory, not through separate files or sockets.**
