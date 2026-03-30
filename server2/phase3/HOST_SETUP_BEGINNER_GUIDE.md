# Host Setup Beginner's Guide

## Overview

This guide walks you through setting up the host-side components for CUDA remoting. The host is the physical XCP-ng/Dom0 machine where QEMU and the mediator daemon run.

**Important:** This guide assumes you're working from a local development machine and need to copy files to the host.

## Prerequisites

Before starting, make sure you have:
- Access to the host machine via SSH (root@host-ip)
- The `phase3` directory on your **local machine** (where you're reading this)
- Basic command-line knowledge (cd, ls, scp, ssh, etc.)

## Step 1: Copy Files to the Host

**This is the first step!** You need to copy the necessary files from your local machine to the host before you can build anything.

### 1.1 Identify Your Host

First, identify your host machine's IP address and username. Common examples:
- Host IP: `10.25.33.10`
- Username: `root`
- Full address: `root@10.25.33.10`

### 1.2 Copy the Entire phase3 Directory

The easiest way is to copy the entire `phase3` directory to the host:

```bash
# From your local machine, navigate to the gpu directory
cd /home/david/Downloads/gpu

# Copy the entire phase3 directory to the host
scp -r phase3 root@10.25.33.10:/root/
```

**What this does:**
- Copies all source files, headers, Makefile, and scripts to `/root/phase3/` on the host
- This includes everything needed to build the mediator and QEMU

**Alternative: Copy Individual Files (if you only updated specific files)**

If you only changed a few files, you can copy them individually:

```bash
# Copy source files
scp phase3/src/mediator_phase3.c root@10.25.33.10:/root/phase3/src/
scp phase3/src/vgpu-stub-enhanced.c root@10.25.33.10:/root/phase3/src/
scp phase3/src/cuda_executor.c root@10.25.33.10:/root/phase3/src/

# Copy header files
scp phase3/include/vgpu_protocol.h root@10.25.33.10:/root/phase3/include/
scp phase3/include/cuda_protocol.h root@10.25.33.10:/root/phase3/include/
scp phase3/include/cuda_executor.h root@10.25.33.10:/root/phase3/include/

# Copy Makefile
scp phase3/Makefile root@10.25.33.10:/root/phase3/
```

### 1.3 Verify Files Were Copied

After copying, verify the files are on the host:

```bash
# SSH into the host
ssh root@10.25.33.10

# Check that phase3 directory exists
ls -la /root/phase3/

# Check key files exist
ls -la /root/phase3/src/mediator_phase3.c
ls -la /root/phase3/src/vgpu-stub-enhanced.c
ls -la /root/phase3/Makefile
```

**Expected output:**
- You should see the `phase3` directory with subdirectories: `src/`, `include/`, `guest-shim/`, etc.
- Key files should show their file sizes (not "No such file or directory")

## Step 2: Build the Mediator on the Host

Now that the files are on the host, you can build the mediator.

### 2.1 SSH into the Host

```bash
ssh root@10.25.33.10
```

### 2.2 Navigate to phase3 Directory

```bash
cd /root/phase3
```

### 2.3 Build the Mediator

```bash
# Build the mediator daemon
make host
```

**What this does:**
- Compiles `mediator_phase3` with CUDA executor support
- Creates the `mediator_phase3` binary in the current directory

**Expected output:**
```
[CC]   src/scheduler_wfq.c
[CC]   src/rate_limiter.c
[CC]   src/metrics.c
...
[LINK] mediator_phase3
[LINK] vgpu-admin

Host binaries built:
  mediator_phase3  — mediator daemon
  vgpu-admin        — admin CLI
```

**Troubleshooting:**

If you get "CUDA not found" errors:
```bash
# Check if CUDA is installed
ls -l /usr/local/cuda
which nvcc

# If CUDA is at a different location, override:
CUDA_PATH=/opt/cuda make host
```

If you get "sqlite3 not found":
```bash
# Install SQLite development package
yum install sqlite-devel
# Or on Debian/Ubuntu:
apt-get install libsqlite3-dev
```

### 2.4 Verify the Build

```bash
# Check that mediator_phase3 was created
ls -lh mediator_phase3

# Should show something like:
# -rwxr-xr-x 1 root root 2.5M Feb 27 10:30 mediator_phase3
```

## Step 3: Start the Mediator

### 3.1 Stop Any Existing Mediator

```bash
# Check if mediator is already running
ps aux | grep mediator_phase3

# If it's running, stop it
pkill mediator_phase3
sleep 2
```

### 3.2 Start the Mediator

```bash
# Start mediator in background with logging
nohup ./mediator_phase3 > /tmp/mediator.log 2>&1 &

# Wait a moment for it to start
sleep 2

# Verify it's running
ps aux | grep mediator_phase3 | grep -v grep
```

**Expected output:**
```
root     12345  0.1  0.2  ...  ./mediator_phase3
```

### 3.3 Check Mediator Logs

```bash
# Check recent logs
tail -50 /tmp/mediator.log
```

**What to look for:**
- ✅ `CUDA initialized` - CUDA is working
- ✅ `Found X QEMU VM(s) with vgpu-cuda device` - VMs are detected
- ✅ `Listening on socket` - Ready to receive connections
- ✅ `[SOCKET] New connection` - VGPU-STUB connected
- ✅ `Total processed: X` - CUDA calls processed (should increase)

## Step 4: (Optional) Rebuild QEMU with VGPU-STUB

**⚠️ WARNING: This step takes 30-45 minutes and is only needed if QEMU doesn't have the vgpu-cuda device.**

### 4.1 Check if QEMU Has the Device

```bash
# Check if QEMU has vgpu-cuda device
/usr/lib64/xen/bin/qemu-system-i386 -device help 2>/dev/null | grep -i vgpu
```

**If you see:**
```
name "vgpu-cuda", bus PCI, desc "Virtual GPU (MMIO + BAR1 + CUDA Remoting)"
```
**Then QEMU already has the device - skip to Step 5!**

**If you see nothing or "QEMU binary not found":**
- You need to rebuild QEMU (continue with Step 4.2)

### 4.2 Prepare QEMU Build Environment

```bash
# Create RPM build directory structure
mkdir -p ~/vgpu-build/rpmbuild/{SOURCES,SPECS,BUILD,RPMS,SRPMS}

# Copy QEMU spec file (you may need to get this from your XCP-ng installation)
# This step depends on your XCP-ng setup
```

### 4.3 Build QEMU

```bash
# From the phase3 directory on the host
cd /root/phase3

# Prepare files for RPM build
make qemu-prepare

# Build QEMU RPM (this takes 30-45 minutes!)
make qemu-build
```

**What this does:**
- Copies vgpu-stub source files to RPM build directory
- Integrates vgpu-stub into QEMU spec file
- Builds QEMU RPM with vgpu-cuda device included

### 4.4 Install QEMU RPM

```bash
# Install the built RPM
make qemu-install

# This will prompt you to confirm installation
# Type 'y' to proceed
```

**After installation:**
- Restart any VMs that use the vgpu device
- The new QEMU binary will be used automatically

## Step 5: Verify Everything is Working

### 5.1 Check Mediator is Running

```bash
# On the host
ps aux | grep mediator_phase3 | grep -v grep
```

### 5.2 Check Mediator Socket

```bash
# Check socket exists
ls -l /var/xen/qemu/*/tmp/vgpu-mediator.sock

# Or fallback location
ls -l /tmp/vgpu-mediator.sock
```

### 5.3 Check Mediator Logs

```bash
# Check for successful connections
tail -100 /tmp/mediator.log | grep -E 'SOCKET|CONNECTION|processed'
```

**Expected output:**
```
[SOCKET] New connection on /var/xen/qemu/root-176/tmp/vgpu-mediator.sock
[CONNECTION] New connection from (fd=44)
Total processed: 3
```

## Quick Reference: Common Commands

### Copy Files to Host
```bash
# Copy entire phase3 directory
scp -r phase3 root@10.25.33.10:/root/

# Copy single file
scp phase3/src/mediator_phase3.c root@10.25.33.10:/root/phase3/src/
```

### Build on Host
```bash
ssh root@10.25.33.10 'cd /root/phase3 && make host'
```

### Restart Mediator
```bash
ssh root@10.25.33.10 'pkill mediator_phase3; sleep 2; cd /root/phase3 && nohup ./mediator_phase3 > /tmp/mediator.log 2>&1 &'
```

### Check Mediator Status
```bash
ssh root@10.25.33.10 'ps aux | grep mediator_phase3 | grep -v grep'
ssh root@10.25.33.10 'tail -50 /tmp/mediator.log'
```

## Troubleshooting

### Problem: "Permission denied" when copying files

**Solution:**
- Make sure you're using the correct username (usually `root`)
- Check that SSH key authentication is set up, or be ready to enter password

### Problem: "CUDA not found" during build

**Solution:**
```bash
# Find where CUDA is installed
find /usr -name "cuda.h" 2>/dev/null | head -1

# Override CUDA path in Makefile
CUDA_PATH=/found/path make host
```

### Problem: Mediator won't start

**Solution:**
```bash
# Check for errors in logs
tail -100 /tmp/mediator.log

# Check if socket directory exists
ls -la /var/xen/qemu/*/tmp/

# Check if mediator binary exists
ls -lh /root/phase3/mediator_phase3
```

### Problem: "No QEMU VMs found"

**Solution:**
- This is normal if no VMs are running with vgpu-cuda device
- Mediator will use fallback socket location: `/tmp/vgpu-mediator.sock`
- Start a VM with vgpu-cuda device configured, and mediator will detect it

## Next Steps

After completing this guide:
1. ✅ Files are copied to host
2. ✅ Mediator is built and running
3. ✅ QEMU has vgpu-cuda device (if rebuilt)
4. ✅ Mediator is listening for connections

**Next:** Set up the guest VM with shim libraries (see guest-side documentation).

## Summary Checklist

- [ ] Copied `phase3` directory to host (`/root/phase3/`)
- [ ] Verified files exist on host
- [ ] Built mediator (`make host`)
- [ ] Started mediator daemon
- [ ] Verified mediator is running (`ps aux | grep mediator`)
- [ ] Checked mediator logs for successful startup
- [ ] (Optional) Rebuilt QEMU if needed
- [ ] Verified QEMU has vgpu-cuda device
- [ ] Confirmed mediator socket exists

---

**Note:** This guide focuses on the host-side setup. For guest VM setup, refer to the guest-side documentation.
