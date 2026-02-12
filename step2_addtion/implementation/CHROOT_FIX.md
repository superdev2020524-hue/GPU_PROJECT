# Chroot Fix: Abstract Unix Socket

## Problem

QEMU in XCP-ng runs in a **chroot jail** (`-chroot /var/xen/qemu/root-<domid>`), which means it cannot access files outside the chroot directory. The original socket path `/tmp/vgpu-mediator.sock` was not accessible from within the chroot, causing connection failures.

## Solution

Changed from a **filesystem Unix socket** to an **abstract Unix socket** (Linux-specific feature). Abstract sockets:
- Don't exist in the filesystem
- Work even in chroot environments
- Are identified by a name (not a file path)
- Are created by setting the first byte of `sun_path` to `\0`

## Changes Made

### 1. `vgpu_protocol.h`
- Changed `VGPU_SOCKET_PATH` to `VGPU_SOCKET_NAME`
- Socket name: `"vgpu-mediator"` (abstract socket)

### 2. `mediator_enhanced.c`
- Updated `setup_socket_server()` to create an abstract socket
- Removed `unlink()` and `chmod()` calls (not needed for abstract sockets)
- Added `stddef.h` for `offsetof()`
- Updated log messages to show `@vgpu-mediator`

### 3. `vgpu-stub-enhanced.c`
- Updated `vgpu_try_connect_mediator()` to connect to abstract socket
- Uses correct address length calculation for abstract sockets

## How Abstract Sockets Work

```c
struct sockaddr_un addr;
memset(&addr, 0, sizeof(addr));
addr.sun_family = AF_UNIX;
addr.sun_path[0] = '\0';  // Abstract socket marker
strncpy(addr.sun_path + 1, "vgpu-mediator", sizeof(addr.sun_path) - 2);

// Address length: offsetof(sun_path) + 1 (for \0) + strlen(name)
socklen_t addr_len = offsetof(struct sockaddr_un, sun_path) + 1 + strlen("vgpu-mediator");
```

## Next Steps

1. **Rebuild the mediator:**
   ```bash
   cd /root/step_review
   make clean
   make mediator_enhanced
   ```

2. **Rebuild QEMU with the updated vGPU stub:**
   - The vGPU stub code needs to be recompiled into QEMU
   - Follow the QEMU build instructions from `BUILD_INSTRUCTIONS.txt`

3. **Test the connection:**
   - Start the mediator: `sudo ./mediator_enhanced`
   - Run the VM client: `sudo ./vm_client_enhanced 100 200`
   - The connection should now work even though QEMU is in a chroot!

## Verification

To verify the abstract socket is being used:
- Check mediator logs: should show `Listening on abstract socket @vgpu-mediator`
- Check QEMU logs (if available): should show `Connected to mediator at abstract socket @vgpu-mediator`
- The socket will NOT appear in `/tmp` or any filesystem location

## Notes

- Abstract sockets are Linux-specific (won't work on BSD/macOS)
- The socket name must be unique system-wide
- No filesystem permissions needed (abstract sockets are accessible based on process permissions)
