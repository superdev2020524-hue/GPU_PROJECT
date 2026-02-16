# Architecture Review: PCI/MMIO Communication Channel

## Customer Requirements

> Move away from file-based (NFS) communication and make the vGPU stub itself the
> communication channel. The VM talks to a PCI device using registers/memory, the
> mediator sits behind that device and schedules work on the physical GPU, and
> pool/priority/VM ID all travel through that PCI channel instead of through files.

### Specific Plan Outlined

1. **Extend the vGPU stub's BAR layout** — request/response area + control registers
   (doorbell, status, error code).
2. **Change the VM-side client** — write requests into MMIO area, ring doorbell,
   poll status, read result from MMIO (instead of `/mnt/vgpu/...`).
3. **Update the host-side vGPU stub** — MMIO write handler pushes requests into
   the mediator queue instead of the filesystem.
4. **Remove the NFS dependency** from this path entirely.

---

## Current Implementation Status

### ✅ FULLY SATISFIED

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| vGPU stub is a proper PCI device | ✅ Done | `vgpu-stub-enhanced.c` — PCI vendor/device `0x1234/0x0003`, revision `0x02` |
| BAR layout with request/response area | ✅ Done | 4KB BAR0: registers (0x00–0x2F), request buffer (0x100–0x1FF), response buffer (0x200–0x2FF) |
| Control registers (doorbell, status, error) | ✅ Done | `VGPU_REG_DOORBELL` (0x20), `VGPU_REG_STATUS` (0x24), `VGPU_REG_ERROR` (0x28) |
| Pool ID, priority, VM ID via MMIO | ✅ Done | `VGPU_REG_POOL_ID` (0x08), `VGPU_REG_PRIORITY` (0x0C), `VGPU_REG_VM_ID` (0x10) |
| VM client writes to MMIO (not NFS) | ✅ Done | `vm_client_enhanced.c` — mmaps PCI BAR, writes request, rings doorbell, polls status |
| VM client reads result from MMIO | ✅ Done | Polls `VGPU_REG_STATUS`, reads `VGPU_RESP_RESULT` from response buffer |
| Host vGPU stub pushes to mediator queue | ✅ Done | MMIO doorbell handler sends request via socket to mediator |
| Mediator scheduling + CUDA unchanged | ✅ Done | Same priority queue (high/medium/low + FIFO), same async CUDA execution |
| NFS dependency removed | ✅ Done | No NFS mounts, no file polling, no `/mnt/vgpu/` paths anywhere |

### ⚠️ PARTIALLY SATISFIED

| Aspect | Status | Current State | Ideal State |
|--------|--------|---------------|-------------|
| "Direct" PCI channel to mediator | ⚠️ 70% | MMIO → vgpu-stub → **Unix socket** → mediator | MMIO → **shared memory** → mediator (zero-copy) |
| Reliability of host-side IPC | ⚠️ 60% | Socket file inside QEMU chroot (fragile path discovery) | eventfd/memfd (FD-based, no filesystem paths) |
| Multi-VM support | ⚠️ 80% | One socket per chroot (works but complex) | Shared memory per VM (simpler, no chroot issues) |

---

## Data Flow: Current Implementation

```
┌─────────────────────────────────────────────────────────────────────┐
│  VM GUEST                                                           │
│                                                                     │
│  vm_client_enhanced                                                 │
│       │                                                             │
│       ├── mmap("/sys/bus/pci/.../resource0")   ← PCI BAR0          │
│       ├── Read pool_id, priority, vm_id        ← MMIO registers    │
│       ├── Write num1, num2 to request buffer   ← MMIO buffer       │
│       ├── Write to DOORBELL register           ← Triggers handler  │
│       ├── Poll STATUS register                 ← Wait for result   │
│       └── Read result from response buffer     ← MMIO buffer       │
│                                                                     │
│  No NFS. No files. No network. Pure PCI/MMIO.                       │
└─────────────────────┬───────────────────────────────────────────────┘
                      │  MMIO trap (hardware-assisted)
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  HOST: QEMU (vgpu-stub-enhanced.c)                                  │
│                                                                     │
│  vgpu_mmio_write() handler:                                         │
│       ├── Doorbell write detected                                   │
│       ├── Read request from BAR buffer                              │
│       ├── Pack into VGPUSocketHeader + VGPURequest                  │
│       └── Send to mediator via Unix socket                          │
│                                                                     │
│  vgpu_socket_read_handler():                                        │
│       ├── Receive response from mediator                            │
│       ├── Write result into BAR response buffer                     │
│       └── Set STATUS = COMPLETE                                     │
└─────────────────────┬───────────────────────────────────────────────┘
                      │  Unix domain socket (host-internal IPC)
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│  HOST: mediator_enhanced.c                                          │
│                                                                     │
│       ├── Receive request from socket                               │
│       ├── Enqueue into priority queue (high > medium > low > FIFO)  │
│       ├── Dequeue highest priority request                          │
│       ├── Execute on GPU via CUDA (async)                           │
│       ├── Collect result                                            │
│       └── Send response back via socket                             │
└─────────────────────────────────────────────────────────────────────┘
```

## What Was Removed (NFS)

The old system required:
- ❌ NFS server on the host
- ❌ NFS client + mount in each VM
- ❌ Guest writing request files to `/mnt/vgpu/requests/`
- ❌ Mediator polling the filesystem for new files
- ❌ Response files written to `/mnt/vgpu/results/`
- ❌ Guest polling for response files

**All of this is eliminated.** The VM now interacts with a PCI device — no filesystem,
no network, no NFS infrastructure.

---

## Files Implementing This Architecture

| File | Role |
|------|------|
| `vgpu_protocol.h` | Shared protocol: register offsets, message formats, constants |
| `vgpu-stub-enhanced.c` | QEMU PCI device: MMIO handlers, socket communication |
| `mediator_enhanced.c` | Host daemon: socket server, priority queue, CUDA dispatch |
| `cuda_vector_add.c/.h` | CUDA kernel: actual GPU computation |
| `vm_client_enhanced.c` | Guest client: PCI discovery, MMIO read/write |
| `build_enhanced_qemu.sh` | Build script: integrates vgpu-stub into QEMU RPM |

---

## Conclusion

The current implementation **satisfies the core requirement**: the vGPU stub IS the
communication endpoint, NFS is retired, and pool/priority/VM ID travel through the
PCI channel. The guest-side architecture is exactly as described — a proper
device-driver style MMIO interaction.

The remaining gap is on the **host-internal IPC** (vgpu-stub → mediator), which uses
Unix sockets instead of shared memory. This is invisible to the guest and does not
affect the guest-side architecture, but could be improved for performance and
reliability in a future iteration.
