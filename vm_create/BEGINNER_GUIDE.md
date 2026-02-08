# Complete Beginner's Guide: Creating and Managing VMs in XCP-ng

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Step 1: Setting Up Xen Orchestra](#step-1-setting-up-xen-orchestra)
4. [Step 2: Creating ISO Storage Repository](#step-2-creating-iso-storage-repository)
5. [Step 3: Creating a New VM](#step-3-creating-a-new-vm)
6. [Step 4: Installing Ubuntu](#step-4-installing-ubuntu)
7. [Step 5: Post-Installation Configuration](#step-5-post-installation-configuration)
8. [Step 6: Connecting to Your VM](#step-6-connecting-to-your-vm)
9. [Troubleshooting](#troubleshooting)
10. [Quick Reference](#quick-reference)

---

## Overview

This guide will walk you through creating a new Ubuntu VM in XCP-ng from start to finish. You'll learn:
- How to set up Xen Orchestra (web-based management interface)
- How to create an ISO storage repository
- How to create and configure a VM
- How to install Ubuntu
- How to configure the VM to boot from the installed system (not the CD)

**Important Concepts:**
- **dom0**: The control domain where you run management commands
- **SR (Storage Repository)**: Where virtual disks and ISO files are stored
- **VBD (Virtual Block Device)**: Connects a disk or CD to a VM
- **VNC**: Virtual Network Computing - used to see the VM's screen

---

## Prerequisites

Before starting, make sure you have:
1. **SSH access to dom0** (the XCP-ng host)
   - Host: `10.25.33.10`
   - Username: `root`
   - Password: (your password)
2. **Ubuntu workstation** with:
   - Docker installed
   - Web browser
   - SSH client
3. **Ubuntu ISO file** (e.g., `ubuntu-22.04.5-desktop-amd64.iso`)
   - Should be in a VGS volume accessible from dom0

---

## Step 1: Setting Up Xen Orchestra

Xen Orchestra (XO) is a web-based interface for managing XCP-ng. We'll install it using Docker on your Ubuntu workstation.

### 1.1 Install Docker (if not already installed)

```bash
# On Ubuntu workstation
sudo apt update
sudo apt install -y docker.io docker-compose
sudo systemctl start docker
sudo systemctl enable docker
```

### 1.2 Install Xen Orchestra

```bash
# On Ubuntu workstation
docker run -d \
  --name=xo \
  --restart=always \
  -p 80:80 \
  -p 443:443 \
  -v /var/lib/xo-server:/var/lib/xo-server \
  -v /var/lib/xo-server/xo-data:/var/lib/xo-server/xo-data \
  ronivay/xen-orchestra:latest
```

### 1.3 Access Xen Orchestra

1. Open your web browser
2. Go to: `http://localhost`
3. Complete the registration:
   - Create an admin account
   - Set a password
   - Click "Register"

### 1.4 Connect XCP-ng Host to Xen Orchestra

1. In Xen Orchestra, click the **"+ Connect pool"** button (top right)
2. Fill in the connection details:
   - **Host**: `10.25.33.10`
   - **Username**: `root`
   - **Password**: (your XCP-ng root password)
   - **Name**: `xcp-ng-syovfxoz` (or any name you prefer)
3. Click **"Connect"**
4. Wait for connection to establish (you'll see your host appear in the left sidebar)

**Troubleshooting:**
- If connection fails, verify SSH access: `ssh root@10.25.33.10`
- Check firewall rules on XCP-ng host
- Ensure XCP-ng API is running: `systemctl status xapi`

---

## Step 2: Creating ISO Storage Repository

An ISO Storage Repository (SR) is where ISO files are stored so VMs can boot from them.

### 2.1 Prepare ISO Storage Location on dom0

```bash
# SSH to dom0
ssh root@10.25.33.10

# Mount VGS volume (if not already mounted)
VG_NAME=$(lvs --noheadings -o vg_name | grep -i iso | head -1)
lvchange -ay "$VG_NAME/iso_download"
mkdir -p /mnt/iso-storage
mount "/dev/$VG_NAME/iso_download" /mnt/iso-storage

# Verify ISO file exists
ls -lh /mnt/iso-storage/*.iso
```

### 2.2 Create ISO SR via Xen Orchestra

1. In Xen Orchestra, click on your host in the left sidebar
2. Click the **"New"** button (top right)
3. Select **"ISO library"**
4. Choose **"File system"** (not SMB or NFS)
5. Fill in the form:
   - **Name**: `VGS ISO Storage`
   - **Path**: `/mnt/iso-storage`
6. Click **"Create"**
7. Wait for the SR to be created (you'll see it in the Storage section)

### 2.3 Fix Mount Point (if needed)

Sometimes the mount point symlink needs to be created manually:

```bash
# On dom0, find the SR UUID
SR_UUID=$(xe sr-list name-label="VGS ISO Storage" params=uuid --minimal)

# Create symlink if it doesn't exist
if [ ! -e "/var/run/sr-mount/$SR_UUID" ]; then
    mkdir -p /var/run/sr-mount
    ln -s /mnt/iso-storage "/var/run/sr-mount/$SR_UUID"
fi
```

### 2.4 Register ISO File

The ISO file should be automatically detected. If not:

```bash
# On dom0
SR_UUID=$(xe sr-list name-label="VGS ISO Storage" params=uuid --minimal)
xe sr-scan uuid="$SR_UUID"

# Verify ISO is registered
xe vdi-list sr-uuid="$SR_UUID" params=name-label
```

---

## Step 3: Creating a New VM

We'll use a script that automates the entire VM creation process.

### 3.1 Download the Script

The script is located at: `/home/david/Downloads/gpu/vm_create/create_vm.sh`

If you need to copy it to dom0:

```bash
# From Ubuntu workstation
scp /home/david/Downloads/gpu/vm_create/create_vm.sh root@10.25.33.10:/root/
```

### 3.2 Create the VM

```bash
# On dom0
cd /root
chmod +x create_vm.sh

# Create Test-3 VM (or Test-4, Test-5, etc.)
bash create_vm.sh Test-3
```

**What the script does:**
- Finds and mounts the VGS volume with ISO
- Verifies ISO SR is accessible
- Registers ISO if needed
- Creates VM with:
  - 4GB RAM
  - 4 CPUs
  - 40GB disk
  - Network interface
  - CD drive with ISO inserted
- Configures UEFI boot and disables VIRIDIAN
- Sets boot order to CD first (for installation)
- Starts the VM

**Auto-generated settings:**
- **IP Address**: `10.25.33.13` (for Test-3)
  - Test-4 = `10.25.33.14`
  - Test-5 = `10.25.33.15`
  - etc.
- **Gateway**: `10.25.33.254`
- **DNS**: `8.8.8.8`
- **VNC Port**: `5901` (for Test-3)
  - Test-4 = `5902`
  - Test-5 = `5903`
  - etc.

**Custom options:**
```bash
# Delete existing VM if it exists
bash create_vm.sh Test-3 --delete-existing

# Custom IP address
bash create_vm.sh Test-3 --ip 10.25.33.20

# Custom memory and disk
bash create_vm.sh Test-3 --memory 8GiB --disk 80GiB

# Custom CPU count
bash create_vm.sh Test-3 --cpus 8
```

### 3.3 Verify VM Creation

```bash
# Check VM status
xe vm-list name-label=Test-3 params=uuid,power-state

# Check VNC socket
DOMID=$(xe vm-param-get uuid=<VM_UUID> param-name=dom-id)
ls -l /var/run/xen/vnc-$DOMID
```

---

## Step 4: Installing Ubuntu

### 4.1 Connect via VNC

**Option 1: Using the helper script**

```bash
# On dom0
bash connect_vnc.sh Test-3
```

**Option 2: Manual connection**

```bash
# On dom0 - Start VNC bridge
DOMID=$(xe vm-param-get uuid=<VM_UUID> param-name=dom-id)
socat TCP-LISTEN:5901,fork,reuseaddr UNIX-CONNECT:/var/run/xen/vnc-$DOMID &

# From Ubuntu workstation - Create SSH tunnel
ssh -N -L 5901:127.0.0.1:5901 root@10.25.33.10 &

# Connect VNC client to: 127.0.0.1:5901
```

### 4.2 Install Ubuntu

1. You should see the Ubuntu installer screen
2. Follow the installation wizard:
   - Select language, keyboard layout
   - Choose installation type (use entire disk)
   - Set up user account
   - **Important**: When asked about network configuration:
     - **IP**: `10.25.33.13/24` (or your VM's IP)
     - **Gateway**: `10.25.33.254`
     - **DNS**: `8.8.8.8`
3. Complete the installation
4. When prompted, click **"Restart Now"**

---

## Step 5: Post-Installation Configuration

After installation, the VM needs to be configured to boot from the installed disk instead of the CD.

### 5.1 Run Post-Installation Script

```bash
# On dom0
bash post_install_vm.sh Test-3 --shutdown
```

**What the script does:**
- Shuts down the VM (required for boot order changes)
- Removes all CD drives completely
- Sets boot order to disk only (`order=c`)
- Verifies disk is bootable
- Ensures UEFI and VIRIDIAN settings are correct

**Important:** The `--shutdown` flag is required because boot order changes only take effect when the VM is halted.

### 5.2 Start the VM

```bash
# On dom0
VM_UUID=$(xe vm-list name-label=Test-3 params=uuid --minimal)
xe vm-start uuid=$VM_UUID
```

### 5.3 Verify Boot Configuration

```bash
# Check boot order (should show order=c)
xe vm-param-get uuid=$VM_UUID param-name=HVM-boot-params

# Check that no CD drives exist
xe vbd-list vm-uuid=$VM_UUID type=CD

# Check disk is bootable
DISK_VBD=$(xe vbd-list vm-uuid=$VM_UUID type=Disk params=uuid --minimal)
xe vbd-param-get uuid=$DISK_VBD param-name=bootable
```

The VM should now boot directly into the installed Ubuntu system.

---

## Step 6: Connecting to Your VM

### 6.1 Via VNC (Graphical Access)

```bash
# On dom0
bash connect_vnc.sh Test-3

# From Ubuntu workstation (in another terminal)
ssh -N -L 5901:127.0.0.1:5901 root@10.25.33.10

# Connect VNC client to: 127.0.0.1:5901
```

**Note:** The domain ID may change after VM reboot. Re-run `connect_vnc.sh` to get the updated port.

### 6.2 Via SSH (Command Line)

Once Ubuntu is installed and running:

```bash
# From Ubuntu workstation
ssh username@10.25.33.13
```

Replace `username` with the user account you created during installation.

---

## Troubleshooting

### Problem: VM boots into UEFI shell instead of installer

**Solution:**
```bash
# Check if ISO is inserted
CD_VBD=$(xe vbd-list vm-uuid=<VM_UUID> type=CD params=uuid --minimal)
xe vbd-param-get uuid=$CD_VBD param-name=vdi-uuid

# If empty, insert ISO
ISO_VDI_UUID=$(xe vdi-list name-label="ubuntu-22.04.5-desktop-amd64.iso" params=uuid --minimal)
xe vbd-insert uuid=$CD_VBD vdi-uuid=$ISO_VDI_UUID

# Check boot order (should be dc for installation)
xe vm-param-get uuid=<VM_UUID> param-name=HVM-boot-params
```

### Problem: VM keeps booting from CD after installation

**Solution:**
```bash
# Run the fix script
bash fix_test3_disk_boot.sh

# Or manually:
# 1. Remove CD drive
CD_VBD=$(xe vbd-list vm-uuid=<VM_UUID> type=CD params=uuid --minimal)
xe vbd-destroy uuid=$CD_VBD

# 2. Set boot order to disk only (c=disk, d=CD)
xe vm-param-set uuid=<VM_UUID> HVM-boot-params:order=c

# 3. Verify
xe vm-param-get uuid=<VM_UUID> param-name=HVM-boot-params
```

### Problem: VM shuts down immediately after starting

**Causes:**
- Boot order is incorrect (trying to boot from non-existent CD)
- Disk is not bootable
- Boot method mismatch (BIOS vs UEFI)

**Solution:**
```bash
# Check boot order
xe vm-param-get uuid=<VM_UUID> param-name=HVM-boot-params

# Check disk bootability
DISK_VBD=$(xe vbd-list vm-uuid=<VM_UUID> type=Disk params=uuid --minimal)
xe vbd-param-get uuid=$DISK_VBD param-name=bootable

# Fix boot method
bash fix_boot_method.sh Test-3
```

### Problem: VNC connection fails

**Solution:**
```bash
# Check if VM is running
xe vm-list name-label=Test-3 params=power-state

# Check domain ID
DOMID=$(xe vm-param-get uuid=<VM_UUID> param-name=dom-id)

# Check VNC socket exists
ls -l /var/run/xen/vnc-$DOMID

# Kill old socat processes
pkill -f "socat.*5901"

# Restart VNC bridge
bash connect_vnc.sh Test-3
```

### Problem: ISO SR not accessible

**Solution:**
```bash
# Check SR mount point
SR_UUID=$(xe sr-list name-label="VGS ISO Storage" params=uuid --minimal)
ls -l /var/run/sr-mount/$SR_UUID

# If symlink is broken, recreate it
rm /var/run/sr-mount/$SR_UUID
ln -s /mnt/iso-storage /var/run/sr-mount/$SR_UUID

# Verify PBD is attached
PBD_UUID=$(xe pbd-list sr-uuid=$SR_UUID params=uuid --minimal)
xe pbd-param-get uuid=$PBD_UUID param-name=currently-attached
```

### Problem: Insufficient storage space

**Solution:**
```bash
# Check available space
xe sr-list params=name-label,physical-size,physical-utilisation

# Delete unused VMs or VDIs
bash delete_test3_vm_complete.sh

# Or use smaller disk size
bash create_vm.sh Test-3 --disk 20GiB
```

---

## Quick Reference

### Boot Order Codes
- `c` = disk (hard drive)
- `d` = CD/DVD
- `n` = network
- `dc` = CD first, then disk (for installation)
- `c` = disk only (after installation)

### Common Commands

```bash
# List all VMs
xe vm-list params=name-label,power-state

# Get VM UUID
VM_UUID=$(xe vm-list name-label=Test-3 params=uuid --minimal)

# Start VM
xe vm-start uuid=$VM_UUID

# Shutdown VM
xe vm-shutdown uuid=$VM_UUID

# Force stop VM
xe vm-destroy uuid=$VM_UUID

# Check boot order
xe vm-param-get uuid=$VM_UUID param-name=HVM-boot-params

# Set boot order
xe vm-param-set uuid=$VM_UUID HVM-boot-params:order=c

# List CD drives
xe vbd-list vm-uuid=$VM_UUID type=CD

# Remove CD drive
xe vbd-destroy uuid=$CD_VBD_UUID

# Check disk bootability
DISK_VBD=$(xe vbd-list vm-uuid=$VM_UUID type=Disk params=uuid --minimal)
xe vbd-param-get uuid=$DISK_VBD param-name=bootable
```

### Script Locations

All scripts are in: `/home/david/Downloads/gpu/vm_create/`

- `create_vm.sh` - Create new VM
- `post_install_vm.sh` - Configure VM after installation
- `connect_vnc.sh` - Connect to VM via VNC
- `fix_test3_disk_boot.sh` - Fix boot from disk issues
- `fix_boot_method.sh` - Fix boot method (UEFI/VIRIDIAN)
- `check_storage.sh` - Check storage space
- `delete_test3_vm_complete.sh` - Delete VM completely

### IP Address Assignment

VMs are assigned IPs based on their number:
- Test-3 → `10.25.33.13`
- Test-4 → `10.25.33.14`
- Test-5 → `10.25.33.15`
- etc.

Formula: `10.25.33.(10 + VM_NUMBER)`

### VNC Port Assignment

VNC ports are assigned based on VM number:
- Test-3 → `5901`
- Test-4 → `5902`
- Test-5 → `5903`
- etc.

Formula: `5900 + VM_NUMBER`

---

## Summary

You've learned how to:
1. ✅ Set up Xen Orchestra for web-based management
2. ✅ Create an ISO Storage Repository
3. ✅ Create a new VM with proper configuration
4. ✅ Install Ubuntu via VNC
5. ✅ Configure the VM to boot from the installed system
6. ✅ Connect to your VM via VNC or SSH

**Key Takeaways:**
- Always verify ISO SR is accessible before creating VMs
- Use `order=dc` during installation (CD first)
- Use `order=c` after installation (disk only)
- Boot order changes require VM to be halted
- Remove CD drives completely after installation

**Next Steps:**
- Create additional VMs using the same process
- Customize VM resources (memory, CPU, disk) as needed
- Set up networking for your specific requirements
- Explore advanced features like snapshots and backups

---

## Additional Resources

- **XCP-ng Documentation**: https://xcp-ng.org/docs/
- **Xen Orchestra Documentation**: https://xen-orchestra.com/docs/
- **Session Log**: `SESSION_LOG.txt` - Detailed technical notes
- **Solution Summary**: `SOLUTION_SUMMARY.md` - Technical solution details

---

*Last updated: 2026-02-04*
*Guide created for: Beginner-friendly VM creation workflow*
