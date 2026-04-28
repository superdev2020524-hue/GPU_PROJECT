# Host-Side Operations Guide for Beginners

## Overview

This guide will help you verify and set up the host-side components needed for CUDA remoting. The host is the physical machine running XCP-ng/Dom0 where QEMU and the mediator daemon run.

## Prerequisites

Before starting, make sure you have:
- Access to the host machine (root or sudo privileges)
- Basic command-line knowledge (cd, ls, grep, etc.)
- The `phase3` directory on the host

## Step-by-Step Guide

### Step 1: Check Current Status

First, let's see what's already running on the host.

#### 1.1 Check if Mediator is Running

```bash
# Check if mediator process is running
ps aux | grep mediator

# Or check systemd service (if it exists)
systemctl status mediator.service
```

**What to look for:**
- If you see a process like `mediator_phase3`, it's running
- If you see "Unit mediator.service could not be found", the service isn't set up yet

#### 1.2 Check if QEMU Has VGPU-STUB Device

```bash
# Find QEMU binary (usually in /usr/lib64/xen/bin/)
QEMU_BIN="/usr/lib64/xen/bin/qemu-system-i386"
if [ -x "$QEMU_BIN" ]; then
    $QEMU_BIN -device help 2>/dev/null | grep -i vgpu
else
    echo "QEMU binary not found at expected location"
fi
```

**What to look for:**
- ✅ **Good**: `name "vgpu-cuda", bus PCI, desc "Virtual GPU (MMIO + BAR1 + CUDA Remoting)"` - QEMU has the device
- ✅ **Good**: `name "vgpu-stub"` - Alternative name for the device
- ❌ **Bad**: No output or "QEMU binary not found" - QEMU needs to be rebuilt or path is wrong

**Example of good output:**
```
name "vgpu-cuda", bus PCI, desc "Virtual GPU (MMIO + BAR1 + CUDA Remoting)"
name "vgpu", bus System
```

#### 1.3 Check Mediator Socket

```bash
# Check if mediator socket exists (common locations)
ls -l /tmp/vgpu-mediator.sock
ls -l /var/vgpu/mediator.sock
# Or check in QEMU chroot directories:
ls -l /var/xen/qemu/*/tmp/vgpu-mediator.sock
```

**What to look for:**
- ✅ **Good**: Socket file exists with permissions like `srw-rw-rw-` (the 's' means it's a socket)
- ✅ **Good**: Example: `srw-rw-rw- 1 root root 0 Feb 19 04:46 /tmp/vgpu-mediator.sock`
- ❌ **Bad**: "No such file or directory" - Mediator isn't running or using a different path

**Note:** The mediator may create sockets in multiple locations:
- `/tmp/vgpu-mediator.sock` - Fallback location
- `/var/vgpu/mediator.sock` - Standard location  
- `/var/xen/qemu/root-XXX/tmp/vgpu-mediator.sock` - Per-VM location (where XXX is the VM ID)

---

### Step 2: Rebuild Components (If Needed)

#### 2.1 Rebuild Mediator with CUDA Support

If the mediator isn't running or doesn't have CUDA support:

```bash
# Navigate to phase3 directory
cd /path/to/phase3

# Build the mediator
make host

# This creates: mediator_phase3
```

**What happens:**
- Compiles the mediator daemon with CUDA executor support
- Creates the `mediator_phase3` binary

**Troubleshooting:**
- If you get "CUDA not found" errors, install CUDA Toolkit:
  ```bash
  # Check if CUDA is installed
  which nvcc
  ls -l /usr/local/cuda
  ```
- If you get "sqlite3 not found", install SQLite development package:
  ```bash
  # On CentOS/RHEL:
  yum install sqlite-devel
  # On Debian/Ubuntu:
  apt-get install libsqlite3-dev
  ```

#### 2.2 Rebuild QEMU with VGPU-STUB (Advanced)

**⚠️ WARNING: This is complex and may require rebuilding the entire QEMU package.**

If QEMU doesn't have the vgpu-stub device, you need to:

1. **Copy the VGPU-STUB source into QEMU source tree:**
   ```bash
   # Find QEMU source (usually in /usr/src or downloaded separately)
   # Copy phase3/src/vgpu-stub-enhanced.c to QEMU's hw/misc/ directory
   # Copy phase3/include/vgpu_protocol.h to QEMU's include/ directory
   ```

2. **Rebuild QEMU:**
   ```bash
   # This is platform-specific and complex
   # Usually involves: ./configure, make, make install
   # Or building an RPM package
   ```

**Note:** If you're using XCP-ng, QEMU is typically provided as an RPM package. Rebuilding requires:
- QEMU source code
- XCP-ng build environment
- Rebuilding the entire QEMU RPM

**Alternative:** If QEMU already has vgpu-stub but it's an older version, you may need to update it. Check with your system administrator.

---

### Step 3: Start/Restart Mediator

#### 3.1 Start Mediator Manually

```bash
# Navigate to phase3 directory
cd /path/to/phase3

# Stop any existing mediator
pkill mediator_phase3
sleep 2

# Start mediator (runs in foreground - use Ctrl+C to stop)
./mediator_phase3

# Or start in background with logging:
nohup ./mediator_phase3 > /tmp/mediator.log 2>&1 &

# Check if it started
sleep 2
ps aux | grep mediator_phase3 | grep -v grep
```

**What to look for in output:**
- ✅ `CUDA initialized` - CUDA is working
- ✅ `Found X QEMU VMs with vgpu-cuda devices` - VMs are detected
- ✅ `Listening on socket: /tmp/vgpu-mediator.sock` or similar - Ready to receive connections
- ✅ `[HEARTBEAT] alive` - Mediator is running (appears periodically)
- ✅ `[SOCKET] New connection` - VGPU-STUB connected successfully
- ✅ `Total processed: X` - CUDA calls are being processed (number should increase)

**Check mediator is running:**
```bash
# See if process exists
ps aux | grep mediator_phase3 | grep -v grep

# Check recent logs
tail -50 /tmp/mediator.log
```

#### 3.2 Create Systemd Service (Optional but Recommended)

Create `/etc/systemd/system/mediator.service`:

```ini
[Unit]
Description=VGPU Mediator Daemon
After=network.target

[Service]
Type=simple
ExecStart=/path/to/phase3/mediator_phase3
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Then:
```bash
# Reload systemd
systemctl daemon-reload

# Enable service (starts on boot)
systemctl enable mediator.service

# Start service
systemctl start mediator.service

# Check status
systemctl status mediator.service
```

---

### Step 4: Verify Everything is Working

#### 4.1 Check Mediator Logs

```bash
# If using systemd:
journalctl -u mediator.service --since '5 minutes ago' | tail -50

# If using manual start with log file (common location):
tail -50 /tmp/mediator.log

# Or check other possible locations:
tail -50 /var/log/mediator_phase3.log

# Look for key indicators:
tail -100 /tmp/mediator.log | grep -E 'CUDA initialized|Found.*VMs|Listening|SOCKET|CONNECTION|processed'
```

**What to look for:**
- ✅ `CUDA initialized` - CUDA is working
- ✅ `Found X QEMU VMs with vgpu-cuda devices` - VMs are detected
- ✅ `Listening on socket` - Ready to receive connections
- ✅ `[SOCKET] New connection` - VGPU-STUB connected
- ✅ `[CONNECTION] New connection` - Connection established
- ✅ `Total processed: X` - CUDA calls processed (should increase when guest sends requests)
- ✅ `Pool A processed: X` - Requests from your VM's pool

#### 4.2 Check QEMU Logs for VGPU Activity

```bash
# Find QEMU process for your VM
ps aux | grep qemu | grep your-vm-name

# Check QEMU logs (location depends on how QEMU was started)
# Common locations:
journalctl -u qemu* --since '5 minutes ago' | grep -i vgpu
# Or check QEMU's stderr if redirected to a file
```

**What to look for:**
- `[vgpu] vm_id=X: CUDA DOORBELL RING` - VGPU-STUB received doorbell
- `[vgpu] vm_id=X: PROCESSING CUDA DOORBELL` - Processing the call
- `[vgpu] vm_id=X: SENDING CUDA CALL to mediator` - Forwarding to mediator
- `[vgpu] vm_id=X: CUDA CALL SENT to mediator` - Successfully sent

#### 4.3 Test End-to-End

From the guest VM, trigger a CUDA operation (e.g., run Ollama). Then check:

**On Host - Mediator Logs:**
```bash
# If mediator is running in foreground or with nohup:
tail -f /tmp/mediator.log | grep -E 'CUDA|cuMemAlloc|call_id|processed'

# Or if using systemd:
journalctl -u mediator.service --since '1 minute ago' | grep -E 'CUDA|cuMemAlloc|call_id'
```

**What to Look For:**
- `[SOCKET] New connection` - VGPU-STUB connected to mediator
- `[CONNECTION] New connection` - Connection established
- `Total processed: X` - Number of CUDA calls processed (should increase)
- `Pool A processed: X` - Requests from Pool A (your VM's pool)

**On Host - QEMU Logs:**
```bash
# QEMU logs are usually in journalctl or redirected to a file
journalctl -u qemu* --since '1 minute ago' | grep -E '\[vgpu\].*CUDA|\[vgpu\].*DOORBELL'

# Or check QEMU's stderr if redirected:
# (Location depends on how QEMU was started)
```

**Expected Flow:**
1. Guest sends CUDA call → VGPU-STUB receives doorbell
2. VGPU-STUB processes → Sends to mediator via socket
3. Mediator receives → Forwards to CUDA executor
4. CUDA executor → Calls real CUDA API on physical GPU
5. Result flows back: CUDA executor → Mediator → VGPU-STUB → Guest

**Success Indicators:**
- ✅ Mediator shows "Total processed: X" increasing
- ✅ Mediator shows "Pool A processed: X" increasing  
- ✅ Connection established: `[SOCKET] New connection`
- ✅ No errors in mediator logs

---

### Step 5: Troubleshooting

#### Problem: Mediator Not Starting

**Check:**
```bash
# Check if CUDA is available
nvidia-smi
nvcc --version

# Check if socket path is writable
ls -ld /tmp
# Or
ls -ld /var/vgpu
```

**Solutions:**
- Install CUDA Toolkit if missing
- Check file permissions on socket directory
- Check if another process is using the socket

#### Problem: QEMU Not Receiving Doorbells

**Check:**
- Is VGPU-STUB device configured in VM? (Check QEMU command line)
- Are the new VGPU-STUB logs appearing? (If not, QEMU may need rebuild)
- Is the VM actually sending doorbells? (Check guest-side logs)

#### Problem: Mediator Not Receiving Messages

**Check:**
```bash
# Verify socket exists
ls -l /tmp/vgpu-mediator.sock

# Check if mediator is listening
netstat -lx | grep vgpu
# Or
ss -lx | grep vgpu

# Check mediator process
ps aux | grep mediator
```

**Solutions:**
- Restart mediator
- Check socket permissions
- Verify VGPU-STUB is connecting (check QEMU logs for connection messages)

#### Problem: CUDA Executor Not Processing Calls

**Check Mediator Logs:**
```bash
journalctl -u mediator.service | grep -E '\[cuda-executor\]|cuMemAlloc|CUDA_CALL'
```

**What to look for:**
- `[cuda-executor] cuMemAlloc: allocating X bytes` - Executor is working
- `[cuda-executor] cuMemAlloc SUCCESS` - Operation succeeded
- No logs = Executor may not be receiving calls

**Solutions:**
- Check if physical GPU is accessible: `nvidia-smi`
- Check CUDA driver version matches requirements
- Verify mediator was built with CUDA support

---

### Step 6: Update VGPU-STUB Code (After Making Changes)

If you've modified `phase3/src/vgpu-stub-enhanced.c` (like we did to add logging):

1. **Copy updated file to QEMU source:**
   ```bash
   # Find QEMU source location
   # Copy: phase3/src/vgpu-stub-enhanced.c → QEMU/hw/misc/vgpu-stub-enhanced.c
   ```

2. **Rebuild QEMU** (see Step 2.2)

3. **Restart VMs** to load new QEMU:
   ```bash
   # Stop VM
   xe vm-shutdown uuid=<vm-uuid>
   
   # Install new QEMU RPM (if built as RPM)
   rpm -Uvh /path/to/new-qemu.rpm
   
   # Start VM
   xe vm-start uuid=<vm-uuid>
   ```

4. **Check new logs appear:**
   ```bash
   # Trigger CUDA operation from guest, then check:
   journalctl -u qemu* --since '1 minute ago' | grep '\[vgpu\].*CUDA DOORBELL'
   ```

---

## Quick Reference Commands

```bash
# Check mediator status
ps aux | grep mediator
systemctl status mediator.service

# Check mediator logs
journalctl -u mediator.service --since '10 minutes ago' | tail -100

# Check QEMU has vgpu-stub
/usr/lib64/xen/bin/qemu-system-i386 -device help 2>/dev/null | grep vgpu

# Check QEMU logs
journalctl -u qemu* --since '10 minutes ago' | grep -i vgpu

# Check socket
ls -l /tmp/vgpu-mediator.sock
netstat -lx | grep vgpu

# Restart mediator
pkill mediator_phase3
cd /path/to/phase3
nohup ./mediator_phase3 > /var/log/mediator_phase3.log 2>&1 &

# Or with systemd
systemctl restart mediator.service
```

---

## Summary Checklist

- [ ] Mediator daemon is running
- [ ] Mediator socket exists and is accessible
- [ ] QEMU has vgpu-stub/vgpu-cuda device
- [ ] VGPU-STUB logs appear when guest sends CUDA calls
- [ ] Mediator receives CUDA calls from VGPU-STUB
- [ ] CUDA executor processes calls on physical GPU
- [ ] End-to-end flow works: Guest → VGPU-STUB → Mediator → Physical GPU

---

## Getting Help

If something doesn't work:

1. **Check the logs first** - Most issues show up in logs
2. **Verify each step** - Make sure prerequisites are met
3. **Check file permissions** - Socket directories need proper permissions
4. **Verify CUDA is working** - Run `nvidia-smi` to confirm GPU is accessible

## Next Steps

After verifying host-side is working:
- Check guest-side logs to confirm end-to-end communication
- Test with actual CUDA operations (like Ollama)
- Monitor performance and adjust as needed
