# Ubuntu Installation Verification Guide

## Problem
VM shows "No bootable device" even though:
- Boot order is set to "d" (disk only) ✓
- Disk is bootable ✓
- No CD drives exist ✓
- Storage is accessible ✓

## Root Cause
**Ubuntu installation likely did NOT complete properly:**
- Disk was never partitioned
- GRUB bootloader was never installed
- Installation process was interrupted or failed silently

## How to Verify Installation Completed

### During Installation (Before Removing CD)

**Check for these indicators:**

1. **Installation Progress:**
   - Did you see "Installing system" progress?
   - Did it reach 100%?
   - Did you see "Installation complete" message?

2. **Disk Partitioning:**
   - During installation, did you see disk partitioning options?
   - Did you select "Erase disk and install Ubuntu" or manually partition?
   - Did partitioning complete successfully?

3. **GRUB Installation:**
   - Did the installer ask where to install GRUB?
   - Did it show "Installing GRUB bootloader"?
   - Any errors during GRUB installation?

4. **Final Steps:**
   - Did you see "Installation complete"?
   - Did you see "Please remove the installation medium, then press ENTER"?
   - Any error messages before this screen?

### After Installation (If VM Fails to Boot)

**Check VM logs:**
```bash
tail -100 /var/log/xensource.log | grep -i "Test-3\|error\|fail"
```

**Check if disk has partitions:**
```bash
# This requires VM to be running, but if it won't boot, we can't check easily
# Alternative: Check if installation actually completed
```

## Solution: Ensure Installation Completes Properly

### Step-by-Step Installation Process

1. **Start Installation:**
   - VM is created with CD containing Ubuntu ISO
   - Boot order is "CD first, then disk" (order=dc)
   - VM starts and boots from CD

2. **During Installation:**
   - **DO NOT** remove CD or change boot order yet
   - Follow Ubuntu installer steps:
     - Select language, keyboard layout
     - Choose "Erase disk and install Ubuntu" (or manual partitioning)
     - Set up user account
     - **Wait for installation to complete 100%**
     - **Wait for "Installation complete" message**

3. **Critical: Verify Installation Completed:**
   - Look for "Installation complete" or "Installation finished" message
   - Check that no errors appeared during installation
   - If you see errors, installation may be incomplete

4. **After Installation Completes:**
   - You'll see: "Please remove the installation medium, then press ENTER"
   - **DO NOT press ENTER yet!**
   - First, remove CD and fix boot order (see below)

5. **Remove CD and Fix Boot Order:**
   ```bash
   # On dom0, while installer is waiting for ENTER
   bash post_install_vm.sh Test-3 --shutdown
   ```
   This will:
   - Eject ISO from CD
   - Remove CD drive completely
   - Set boot order to disk only
   - Shut down VM

6. **Then Press ENTER:**
   - After the script completes, press ENTER in the installer
   - VM will reboot and should boot from disk

## If Installation Didn't Complete

**Symptoms:**
- No "Installation complete" message
- Installation progress didn't reach 100%
- Errors during installation
- VM fails to boot even after fixing boot order

**Solution: Reinstall**

1. **Delete the VM:**
   ```bash
   VM_UUID="8c57e7bd-14ff-6ff3-2bfe-a23b57b1d0be"
   xe vm-shutdown uuid="$VM_UUID" force=true
   xe vm-destroy uuid="$VM_UUID"
   xe vm-uninstall uuid="$VM_UUID" force=true
   ```

2. **Recreate VM:**
   ```bash
   bash create_vm.sh Test-3 --ip 10.25.33.11
   ```

3. **Complete Installation Properly:**
   - Let installation finish completely
   - Wait for "Installation complete" message
   - Then run post-install script
   - Then press ENTER

## Common Installation Issues

1. **Installation Interrupted:**
   - VNC connection lost during installation
   - VM shut down during installation
   - Network issues during package download

2. **Disk Partitioning Failed:**
   - Disk was too small
   - Disk had errors
   - Partitioning step was skipped

3. **GRUB Installation Failed:**
   - Bootloader installation step failed
   - Disk wasn't selected for GRUB installation

## Verification Checklist

Before removing CD, verify:
- [ ] Installation progress reached 100%
- [ ] "Installation complete" message appeared
- [ ] No error messages during installation
- [ ] Disk partitioning completed successfully
- [ ] GRUB installation completed (if shown)

If any item is unchecked, installation likely didn't complete and you should reinstall.
