# Phase 2 Design: ioeventfd + Shared Memory Architecture

**Author:** Bren  
**Date:** February 14, 2026  
**Status:** Design — Not Yet Implemented  
**Depends On:** Phase 1 (PCI/MMIO Communication, complete)

---

## 1. Motivation

Phase 1 replaced the NFS transport with PCI/MMIO communication. From the guest's
perspective, that job is done — the VM talks to a PCI device, writes registers, rings
a doorbell, and reads results back. No NFS, no files, no network. That part stays.

The issue is on the host side. Right now, when the guest writes request data into the
BAR and rings the doorbell, the following happens inside QEMU:

1. QEMU's MMIO trap fires `vgpu_mmio_write()`.
2. The doorbell handler copies the request out of the BAR buffer.
3. It packs the data into a `VGPUSocketHeader` + payload.
4. It calls `sendmsg()` on a Unix domain socket to the mediator.
5. The mediator calls `read()` on the other end to receive the data.
6. The mediator processes the request, builds a response.
7. The mediator calls `write()` to send the response back.
8. QEMU's `vgpu_socket_read_handler()` receives it and copies it into the BAR.

That's four data copies (BAR → stub buffer → kernel socket buffer → mediator buffer,
and the reverse for responses), two context switches for the socket I/O, and QEMU
sitting in the middle of every single request. On top of that, the socket setup
depends on discovering QEMU's chroot directory by scanning `/proc`, which works but
is fragile — it requires the mediator to be started after the VMs, and if a VM
restarts, the mediator needs to be restarted too.

The shared memory + ioeventfd approach solves both problems: data moves zero-copy
between the guest and the mediator, and notification happens through kernel file
descriptors that don't depend on filesystem paths or chroot environments.

---

## 2. How It Works — Overview

The idea comes from how production virtual devices work in QEMU (vhost-net, vhost-user,
SPDK). Instead of QEMU owning the device's data buffers and relaying traffic through
sockets, the data buffers live in shared memory that both QEMU and the mediator can
map. Notifications (doorbell, completion) travel through `eventfd` descriptors that
the kernel delivers directly, bypassing the socket stack entirely.

```
  VM Guest
  ┌─────────────────────────────────────────────────┐
  │  vm_client                                       │
  │    │                                             │
  │    ├─ mmap BAR0 resource                         │
  │    ├─ write request to buffer at 0x040           │
  │    ├─ write 1 to DOORBELL at 0x000               │
  │    ├─ poll STATUS at 0x004                       │
  │    └─ read result from buffer at 0x440           │
  │                                                  │
  │  (unchanged from Phase 1)                        │
  └──────────────────────┬───────────────────────────┘
                         │  MMIO trap (KVM → QEMU)
                         ▼
  ┌──────────────────────────────────────────────────┐
  │  QEMU  (vgpu-stub)                               │
  │                                                  │
  │  Phase 1 (current):                              │
  │    doorbell write → copy data → sendmsg(socket)  │
  │    socket read    → copy data → write to BAR     │
  │                                                  │
  │  Phase 2 (proposed):                             │
  │    doorbell write → write(ioeventfd)  (8 bytes)  │
  │    ← mediator writes result to shared memory →   │
  │    ← mediator signals completion eventfd →        │
  │    completion eventfd → set STATUS = DONE         │
  │                                                  │
  │  BAR0 backed by memfd (shared with mediator)     │
  └──────────────────────┬───────────────────────────┘
                         │  memfd (shared memory)
                         │  eventfd (doorbell + completion)
                         ▼
  ┌──────────────────────────────────────────────────┐
  │  MEDIATOR                                        │
  │                                                  │
  │  mmap(memfd) → direct access to BAR contents     │
  │  poll(doorbell_efd) → guest rang the doorbell    │
  │  read request directly from shared memory        │
  │  execute on GPU (CUDA)                           │
  │  write result directly into shared memory        │
  │  write(completion_efd) → QEMU sets STATUS=DONE   │
  └──────────────────────────────────────────────────┘
```

The guest-side protocol is identical. The guest still writes to the same register
offsets, rings the same doorbell, polls the same status register. Nothing changes
inside the VM. The difference is entirely on the host — how the data gets from QEMU
to the mediator and back.

---

## 3. Core Mechanisms

### 3.1 memfd — Anonymous Shared Memory

`memfd_create()` creates an anonymous file-backed memory region that exists only in
RAM. It returns a file descriptor, not a filesystem path. Any process with the fd
can `mmap()` it and see the same physical pages.

```c
// In QEMU (vgpu-stub realize function):
int memfd = memfd_create("vgpu-bar0", MFD_CLOEXEC);
ftruncate(memfd, VGPU_BAR_SIZE);   // 4096 bytes

// Map it as the BAR0 backing store
void *bar_mem = mmap(NULL, VGPU_BAR_SIZE, PROT_READ | PROT_WRITE,
                     MAP_SHARED, memfd, 0);
```

QEMU registers this memory as the BAR0 region using
`memory_region_init_ram_from_fd()`. When the guest writes to BAR0 at offset 0x040,
those bytes land directly in the memfd-backed pages. The mediator maps the same fd
and reads from offset 0x040 — zero copies.

The critical property of memfd is that it has no filesystem path. It doesn't care
about chroot, mount namespaces, or network namespaces. The fd is passed between
processes (via `SCM_RIGHTS` on a setup socket, or via a command-line argument if
inherited from a parent process), and that's it.

### 3.2 eventfd — Lightweight Notification

`eventfd()` creates a kernel object that acts as a counter. Writing 8 bytes
(a `uint64_t`) increments it. Reading 8 bytes returns the current value and resets
it to zero. Both ends can be polled with `poll()`, `select()`, or `epoll()`.

We use two eventfds per VM:

| eventfd | Direction | Purpose |
|---------|-----------|---------|
| `doorbell_efd` | QEMU → Mediator | Guest wrote to DOORBELL register |
| `completion_efd` | Mediator → QEMU | Result is ready in shared memory |

```c
// Created during setup:
int doorbell_efd   = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
int completion_efd = eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC);
```

When the guest rings the doorbell:

```c
// In QEMU's vgpu_mmio_write(), case VGPU_REG_DOORBELL:
uint64_t val = 1;
write(doorbell_efd, &val, sizeof(val));
// That's it. No data copy. The mediator wakes up from poll().
```

When the mediator finishes:

```c
// In mediator, after writing result into shared memory:
uint64_t val = 1;
write(completion_efd, &val, sizeof(val));
// QEMU's event loop wakes up, sets STATUS = DONE.
```

### 3.3 ioeventfd — KVM Shortcut (Optional Optimization)

KVM supports `ioeventfd`, which is one step beyond regular eventfd. With ioeventfd,
the doorbell write from the guest doesn't even trap to QEMU's MMIO handler — KVM
intercepts it directly in the kernel and writes to the eventfd without a VM exit to
userspace. This skips the QEMU overhead entirely for the notification path.

```c
// In QEMU, during device setup:
memory_region_add_eventfd(&s->mmio,
                          VGPU_REG_DOORBELL,       // offset
                          4,                        // size
                          false,                    // match_data (any write)
                          0,                        // data (unused)
                          &s->doorbell_eventfd);    // eventfd
```

With this in place, a guest write to `BAR0 + 0x000` goes:

```
Guest write → KVM kernel module → eventfd increment → mediator wakes up
```

No QEMU involvement at all on the request path. QEMU is only involved on the
completion path (to update the STATUS register that the guest polls).

This is the same mechanism that virtio uses for submission queues and is well-tested
in production QEMU.

---

## 4. FD Passing — Setup Protocol

The challenge is getting the memfd and eventfd descriptors from QEMU to the mediator.
There are two approaches:

### Option A: SCM_RIGHTS Over a Setup Socket

This is the standard approach used by vhost-user. On device realize, the vgpu-stub
connects to the mediator over a single Unix socket (the same one we use now, or a
dedicated setup socket) and sends a one-time setup message containing:

```c
struct VGPUSetupMessage {
    uint32_t magic;            // VGPU_SOCKET_MAGIC
    uint32_t msg_type;         // VGPU_MSG_SETUP (new type)
    uint32_t vm_id;
    uint32_t bar_size;         // 4096
    // Ancillary data (SCM_RIGHTS): 3 file descriptors
    //   fd[0] = memfd      (shared memory backing BAR0)
    //   fd[1] = doorbell_efd
    //   fd[2] = completion_efd
};
```

The mediator receives the message, extracts the three fds from the ancillary data,
mmaps the memfd, and starts polling the doorbell_efd. After this handshake, the
socket can be closed — all further communication happens through shared memory and
eventfds.

This still needs the chroot socket for the initial handshake. But it's a one-time
event at VM startup, not per-request, so the fragility is much reduced.

### Option B: Inherit FDs via QEMU Command Line

QEMU supports passing pre-created file descriptors via the `-add-fd` command-line
option. The orchestration layer (Xen toolstack or a wrapper script) would:

1. Create the memfd, doorbell_efd, and completion_efd before starting QEMU.
2. Pass them to QEMU as inherited fds: `-add-fd fd=N,set=1,opaque=memfd` etc.
3. Pass the same fds to the mediator (or let the mediator create them and hand
   them to QEMU).

This approach eliminates the setup socket entirely — no filesystem paths, no chroot
issues, no namespace issues. The tradeoff is that it requires the orchestration layer
to manage the fd lifecycle, which couples it more tightly to the deployment.

**Recommendation:** Start with Option A (SCM_RIGHTS) because it keeps the vgpu-stub
self-contained and doesn't require changes to the VM launch scripts. Move to Option B
later if we want to eliminate the setup socket entirely.

---

## 5. Detailed Data Flow

### 5.1 Request Path (Guest → GPU)

```
1. Guest: writes request payload to BAR0 + 0x040   (lands in memfd pages)
2. Guest: writes request length to BAR0 + 0x018     (QEMU MMIO trap)
3. Guest: writes 1 to BAR0 + 0x000 (DOORBELL)
   → KVM ioeventfd fires → doorbell_efd incremented
4. Mediator: poll() returns on doorbell_efd
5. Mediator: reads request directly from its mmap of the memfd at offset 0x040
6. Mediator: enqueues request in priority queue
7. Mediator: dequeues, dispatches to CUDA
```

Data copies: **zero** for the payload (it's already in shared pages).  
Context switches: **one** (KVM eventfd delivery to mediator's poll).

### 5.2 Response Path (GPU → Guest)

```
1. CUDA completes, mediator has result
2. Mediator: writes response to its mmap at offset 0x440   (shared memory)
3. Mediator: writes response length to its mmap at offset 0x01C
4. Mediator: write(completion_efd, 1)
5. QEMU: event loop fires completion handler
6. QEMU: sets STATUS register = DONE (internal state, not in memfd)
7. Guest: polls STATUS, sees DONE
8. Guest: reads result from BAR0 + 0x440   (already in memfd pages)
```

Data copies: **zero** for the payload.  
The STATUS register is the one thing QEMU still manages, because it needs to be
visible to the guest via MMIO reads. The response buffer itself is zero-copy.

### 5.3 Comparison with Current System

| Metric | Phase 1 (sockets) | Phase 2 (shared memory) |
|--------|-------------------|------------------------|
| Data copies per request | 4 (BAR→stub→kernel→mediator, ×2) | 0 |
| Context switches per request | 4+ (sendmsg/recvmsg, ×2) | 2 (eventfd, ×2) |
| QEMU in data path | Yes (copies + relays every byte) | No (only status register) |
| Chroot dependency | Yes (socket file in each chroot) | No (fd-based, no paths) |
| Namespace dependency | No (filesystem socket) | No (fd-based) |
| Setup complexity | Auto-discover chroots, create sockets | One-time fd handshake |
| Hot-add VMs | Mediator must be restarted | Mediator accepts new handshake |

---

## 6. Changes Required

### 6.1 vgpu-stub-enhanced.c

```
Current:
  - Owns req_buf[] and resp_buf[] as local arrays in VGPUStubState
  - BAR0 backed by memory_region_init_io() (MMIO trap on every access)
  - Doorbell handler copies data and sends via socket
  - Socket read handler copies response into resp_buf

Proposed:
  - Create memfd at realize time, ftruncate to 4KB
  - Map BAR0 buffer regions (0x040–0x83F) to the memfd
  - Control registers (0x000–0x03F) remain as MMIO-trapped (QEMU manages them)
  - Create doorbell_efd and completion_efd
  - Register doorbell_efd as ioeventfd on VGPU_REG_DOORBELL
  - Register completion_efd in QEMU event loop (like current socket handler)
  - On realize: send setup message with 3 fds to mediator via SCM_RIGHTS
  - Doorbell handler: do nothing (ioeventfd handles it)
  - Completion handler: read completion_efd → set STATUS = DONE
```

The key insight is that BAR0 becomes a split region:
- **Control registers** (0x000–0x03F): traditional MMIO, trapped by QEMU, because
  the guest reads STATUS here and QEMU needs to control what it sees.
- **Data buffers** (0x040–0x83F): backed by memfd, directly visible to both QEMU
  and the mediator. No MMIO trap on buffer reads/writes — they go straight to RAM.

QEMU's `memory_region` API supports this split via sub-regions or aliasing.

### 6.2 mediator_enhanced.c

```
Current:
  - Scans /proc for QEMU chroots
  - Creates filesystem sockets in each chroot
  - select() on server sockets, accept(), read messages
  - Sends response via socket write()

Proposed:
  - Listen on a single setup socket (or keep chroot sockets for initial handshake)
  - On new connection: receive VGPUSetupMessage via recvmsg() with SCM_RIGHTS
  - Extract memfd, doorbell_efd, completion_efd from ancillary data
  - mmap(memfd) to get direct access to BAR0 contents
  - Add doorbell_efd to epoll set (replaces select on server sockets)
  - On doorbell event:
      read request from mmap'd memory at offset 0x040
      enqueue in priority queue (same as now)
  - On CUDA completion:
      write response into mmap'd memory at offset 0x440
      write(completion_efd, 1) to notify QEMU
  - Close the setup socket (no longer needed after handshake)
```

The chroot discovery logic, multiple server sockets, and `select()` loop all go away.
Replaced by a simpler model: one epoll loop waiting on doorbell eventfds, one per VM.

### 6.3 vgpu_protocol.h

```
New additions:
  - VGPU_MSG_SETUP message type for the handshake
  - VGPUSetupMessage structure definition
  - Document that VGPU_MSG_SETUP carries 3 fds via SCM_RIGHTS
```

### 6.4 vm_client_enhanced.c

**No changes.** The guest-side protocol is identical. The guest still mmaps BAR0 from
sysfs, writes to the same offsets, and polls the same STATUS register. Whether the
host side uses sockets or shared memory is invisible to the guest.

---

## 7. Implementation Order

This can be done incrementally without breaking the existing system:

### Step 1: memfd-backed BAR buffers

Change the vgpu-stub to back the data buffer regions with a memfd instead of local
arrays. Keep the socket IPC for now. This validates that the guest can still read and
write the buffers correctly when they're backed by shared memory instead of QEMU's
heap.

**Risk:** Low. If the memfd mapping doesn't work with QEMU's memory region API,
we find out early with no other changes.

### Step 2: eventfd notifications

Create the doorbell and completion eventfds. Wire the doorbell as an ioeventfd on
the DOORBELL register offset. Wire the completion eventfd into QEMU's event loop.
Keep the socket IPC — the doorbell eventfd fires alongside the existing socket send,
and we verify both paths agree.

**Risk:** Low. ioeventfd is a well-understood QEMU mechanism.

### Step 3: FD passing via SCM_RIGHTS

Add the setup handshake. On vgpu-stub realize, connect to the mediator's setup socket
and send the three fds. The mediator receives them, mmaps the memfd, and starts
polling the doorbell_efd. At this point both paths (socket and shared memory) are
active, and we can compare results.

**Risk:** Medium. SCM_RIGHTS is fiddly to get right, especially across chroot
boundaries. But it only needs to work once per VM startup.

### Step 4: Remove socket data path

Once the shared memory path is validated, remove the socket-based data relay from the
doorbell handler. The vgpu-stub no longer copies data or sends socket messages on
every request. The mediator no longer reads data from the socket — it reads directly
from the mmap'd memfd. Socket remains only for the initial setup handshake.

**Risk:** Low, if Step 3 is validated.

### Step 5: Clean up

Remove the chroot discovery logic, the multi-socket select loop, and the per-request
socket I/O code. What remains is a much simpler mediator: one epoll loop, N doorbell
eventfds, N mmap'd regions, one priority queue, one CUDA backend.

---

## 8. Compatibility and Rollback

The Phase 1 socket-based path will remain in the codebase behind a runtime flag
(e.g. `--legacy-socket`) until Phase 2 is fully validated. If anything goes wrong
with the shared memory path in a particular deployment, we can fall back to sockets
without rebuilding QEMU.

The guest-side code and protocol are unchanged between Phase 1 and Phase 2. A Phase 2
host is fully compatible with unmodified guest clients.

---

## 9. Known Constraints

- **QEMU memory region API**: The split-BAR approach (MMIO-trapped registers +
  RAM-backed buffers) requires careful use of QEMU's sub-region priority system.
  The control register region must have higher priority than the memfd-backed region
  so that register accesses trap to QEMU while buffer accesses go to RAM. This is
  well-documented in QEMU's memory model but needs care to get right.

- **XCP-ng / Xen QEMU fork**: XCP-ng runs a patched QEMU. We need to verify that
  `memory_region_init_ram_from_fd()` and `memory_region_add_eventfd()` are available
  in their build. Both are upstream QEMU features, but XCP-ng's fork may be older.

- **Concurrency**: With shared memory, the guest and the mediator can read and write
  the buffer region simultaneously. The current protocol (write request → ring bell →
  wait for STATUS → read response) is inherently sequential per-VM, so there's no
  race in practice. But if we later add multi-request queues, we'll need proper
  memory barriers and possibly a lock-free ring buffer.

- **Security**: The memfd gives the mediator full read/write access to the BAR
  contents, including the control register region if we're not careful. The memfd
  should be mapped with a size/offset that only covers the data buffers (0x040–0x83F),
  not the control registers. QEMU retains sole ownership of the register state.

---

## 10. Summary

Phase 1 got the communication path right from the guest's perspective. Phase 2 gets
it right from the host's perspective. The shared memory + eventfd approach is the
standard way high-performance virtual devices work in QEMU — it's what vhost-net and
vhost-user have been doing for years. We're applying the same pattern to the vGPU
stub.

The guest doesn't change. The mediator's scheduling and CUDA logic don't change.
What changes is the plumbing between QEMU and the mediator: from socket-based copy
to memory-mapped zero-copy. The result is lower latency, fewer copies, simpler
setup (no chroot dance), and a cleaner architecture overall.

 Bren
