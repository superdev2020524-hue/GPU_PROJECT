# Solution Summary: ISO Boot Failure Resolution

## Problem Statement
VMs failed to start with error: "VDI could not be found on the storage substrate" or "Cannot start here [VM requires access to SR: ... (SMB ISO library)]"

## Root Cause
- SMB ISO library SR was unreachable (network issue: `mount error(113)`)
- No other accessible ISO SRs were available
- XCP-ng CLI cannot create file-based ISO SRs directly

## Solution Path (Option A - Xen Orchestra)

### Step 1: Install Xen Orchestra
- Installed XO via Docker on Ubuntu machine
- Accessed web interface at `http://localhost`
- Connected to XCP-ng host: `10.25.33.10` (root / Calvin@123)

### Step 2: Create VGS ISO Storage SR
- Created "VGS ISO Storage" SR via XO web interface
- Type: ISO library / File system (ISO library)
- Path: `/mnt/iso-storage`
- XOA automatically scanned and registered existing ISO files

### Step 3: Fix Mount Point Issue
- Problem: PBD was attached but mount point didn't exist
- Solution: Created symlink: `/var/run/sr-mount/<SR_UUID>` → `/mnt/iso-storage`
- Script: `fix_vgs_sr_mount.sh`

### Step 4: Update VM Creation Script
- Modified `verify_sr_accessible()` to handle symlinks
- File-based ISO SRs use symlinks, not actual mounts
- Verification checks: symlink validity, PBD attachment, file readability

### Step 5: Create VM
- Script found "VGS ISO Storage" SR
- Used registered ISO VDI
- Created and started Test-3 VM successfully

## Key Technical Details

### Storage Repository (SR)
- **Name**: "VGS ISO Storage"
- **UUID**: `fc99ca21-ebc8-89f5-dbd3-b7e9a0b2ae49`
- **Type**: `iso` (file-based ISO library)
- **Path**: `/mnt/iso-storage` (VGS logical volume mount)
- **Mount Point**: `/var/run/sr-mount/fc99ca21-ebc8-89f5-dbd3-b7e9a0b2ae49` (symlink)
- **Access Mode**: Local
- **PBD Status**: Attached (`currently-attached: true`)

### ISO File
- **Location**: `/mnt/iso-storage/ubuntu-22.04.5-desktop-amd64.iso`
- **VDI UUID**: `7d1daaac-9cc7-465a-89ba-567518f18298`
- **Registration**: Auto-registered by XOA when SR was created/scanned

### VM Configuration (Test-3)
- **VM UUID**: `fbf30cd3-b5d6-6902-e7ce-cdf2841022e0`
- **Memory**: 4GB
- **CPUs**: 4
- **Disk**: 40GB
- **Network**: Static IP `10.25.33.13/24`, Gateway `10.25.33.254`, DNS `8.8.8.8`
- **Boot Order**: CD first, then disk

## Important Lessons Learned

1. **File-based ISO SRs use symlinks, not mounts**
   - Don't require mount table entry
   - Mount point is a symlink to actual directory
   - Verification must check symlink validity, not mount table

2. **XOA auto-registers ISOs**
   - When SR is created/scanned, XOA automatically registers ISO files
   - No manual `xe vdi-create` needed if ISO is already in directory

3. **PBD attachment ≠ mount point existence**
   - PBD can be attached but mount point may not exist
   - Must create symlink manually for file-based SRs

4. **Robust verification is critical**
   - Check: directory/symlink existence, PBD attachment, file readability
   - Don't rely on mount table for file-based ISO SRs

## Files Created/Modified

- `XOA_DOCKER_METHOD.md` - XO installation guide
- `CONNECT_XCP_HOST_IN_XO.md` - Connecting to XCP-ng host
- `CREATE_VGS_ISO_SR_VIA_XO.md` - Creating ISO SR via XO
- `fix_vgs_sr_mount.sh` - Fix mount point symlink
- `create_test3_vm.sh` - Updated with symlink support
- `delete_test3_vm.sh` - Helper to clean up VMs

## Next Steps

1. Post-installation configuration (remove CD, change boot order)
2. VNC connection after reboot (domain ID changes)
3. Create universal VM creation script for Test-X VMs
