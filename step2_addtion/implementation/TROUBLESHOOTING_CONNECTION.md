# Troubleshooting: vGPU Stub Cannot Connect to Mediator

## Problem
VM client shows: `[ERROR] Device error: code=3 (MEDIATOR_UNAVAILABLE)`

This means the vGPU stub device in QEMU cannot connect to the mediator socket.

## Diagnostic Steps

### 1. Verify Mediator is Running

On host (Dom0):
```bash
# Check if mediator process is running
ps aux | grep mediator_enhanced

# Check if socket exists
ls -la /tmp/vgpu-mediator.sock
# Should show: srw-rw-rw- ... /tmp/vgpu-mediator.sock
```

### 2. Check Socket Permissions

The socket should be readable/writable by all:
```bash
ls -la /tmp/vgpu-mediator.sock
# Should show permissions: srw-rw-rw- or srwxrwxrwx
```

If permissions are wrong:
```bash
sudo chmod 666 /tmp/vgpu-mediator.sock
```

### 3. Verify Enhanced vGPU Stub is Installed

The VM must be using the **enhanced vGPU stub v2**, not the old v1.

**Check QEMU version:**
```bash
# On host
/usr/lib64/xen/bin/qemu-system-i386 -device help 2>/dev/null | grep vgpu-stub
# Should show: vgpu-stub
```

**Check device revision in VM:**
```bash
# Inside VM
lspci -v | grep -A 10 "Processing accelerators"
# Should show: Revision: 02 (for v2 with MMIO support)
```

If revision is `01`, the old stub is still installed. Rebuild and install QEMU with enhanced stub.

### 4. Check QEMU Logs

On host, check QEMU log for vGPU stub messages:
```bash
# Find VM UUID
xe vm-list name-label="your-vm-name" --minimal

# Check QEMU log
tail -f /var/log/xen/qemu-dm-<VM_UUID>.log | grep vgpu-stub
```

**Expected messages:**
```
[vgpu-stub] realised  vm_id=1  pool=A  priority=high  rev=0x02
[vgpu-stub] Connected to mediator at /tmp/vgpu-mediator.sock (fd=X)
```

**If you see:**
```
[vgpu-stub] sendmsg failed: Connection refused
```
→ Mediator not running or socket not accessible

**If you see:**
```
[vgpu-stub] socket() failed: Permission denied
```
→ QEMU doesn't have permission to create sockets

**If you see nothing:**
→ vGPU stub might not be the enhanced version, or logs are elsewhere

### 5. Test Socket Accessibility

Test if QEMU can access the socket:
```bash
# On host, as the user running QEMU (usually root)
sudo -u qemu test -S /tmp/vgpu-mediator.sock && echo "Accessible" || echo "Not accessible"
```

### 6. Check QEMU Process Context

QEMU might be running in a restricted environment:
```bash
# Check QEMU process
ps aux | grep qemu | grep your-vm-name

# Check if it's in a container/chroot
cat /proc/$(pgrep -f "qemu.*your-vm")/cwd
```

### 7. Verify Socket Path

The vGPU stub connects to `/tmp/vgpu-mediator.sock`. Make sure:
- Mediator creates socket at this exact path
- QEMU can access `/tmp` directory
- No SELinux/AppArmor blocking socket access

## Common Solutions

### Solution 1: Fix Socket Permissions
```bash
# Make socket world-readable/writable
sudo chmod 666 /tmp/vgpu-mediator.sock

# Or change ownership to qemu user
sudo chown qemu:qemu /tmp/vgpu-mediator.sock
```

### Solution 2: Use Different Socket Path

If `/tmp` is not accessible, modify the socket path:

**In vgpu_protocol.h:**
```c
#define VGPU_SOCKET_PATH  "/var/run/vgpu-mediator.sock"
```

**In mediator_enhanced.c:**
- Change `VGPU_SOCKET_PATH` to match

**Then rebuild both.**

### Solution 3: Rebuild QEMU with Enhanced Stub

If the VM is using old vGPU stub v1:
```bash
# On host
cd ~/vgpu-build
# Follow build_enhanced_qemu.sh steps
# Reinstall QEMU RPM
# Restart VM
```

### Solution 4: Check SELinux/AppArmor

If SELinux is enabled:
```bash
# Check SELinux status
getenforce

# If enforcing, check for denials
sudo ausearch -m avc -ts recent | grep vgpu

# Temporarily allow (for testing)
sudo setenforce 0
```

## Quick Test

**On host:**
```bash
# 1. Ensure mediator is running
sudo ./mediator_enhanced &
MEDIATOR_PID=$!

# 2. Check socket
ls -la /tmp/vgpu-mediator.sock

# 3. Test connection manually
echo "test" | nc -U /tmp/vgpu-mediator.sock
# (This will fail, but confirms socket is accessible)

# 4. Check QEMU log when VM client runs
tail -f /var/log/xen/qemu-dm-*.log | grep -i vgpu
```

## Expected Working State

**Mediator terminal:**
```
[MEDIATOR] Accepting connections on /tmp/vgpu-mediator.sock
[ENQUEUE] Pool A: vm=1, req_id=..., prio=2, 100+200
```

**QEMU log:**
```
[vgpu-stub] Connected to mediator at /tmp/vgpu-mediator.sock (fd=5)
```

**VM client:**
```
[RESPONSE] Received: 300
Result: 100 + 200 = 300
```

## Still Not Working?

1. **Check if enhanced QEMU is actually installed:**
   ```bash
   rpm -qa | grep qemu
   # Verify version includes your custom build
   ```

2. **Verify VM is using enhanced stub:**
   ```bash
   # In VM
   lspci -v | grep -A 5 vgpu
   # Check revision number
   ```

3. **Try connecting from QEMU context:**
   ```bash
   # Find QEMU process
   QEMU_PID=$(pgrep -f "qemu.*your-vm")
   
   # Check its environment
   sudo cat /proc/$QEMU_PID/environ | tr '\0' '\n' | grep -i path
   ```

4. **Check for firewall/network restrictions:**
   - Unix sockets shouldn't be affected, but verify
