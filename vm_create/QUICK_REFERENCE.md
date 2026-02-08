# Quick Reference Guide: Creating Test-X VMs

## Universal VM Creation Script

### Basic Usage
```bash
# Create Test-3 VM (auto-generates IP: 10.25.33.13)
bash create_vm.sh Test-3

# Create Test-4 VM (auto-generates IP: 10.25.33.14)
bash create_vm.sh Test-4

# Create Test-5 VM with custom IP
bash create_vm.sh Test-5 --ip 10.25.33.20

# Create Test-6 VM, delete existing if present
bash create_vm.sh Test-6 --delete-existing

# Create Test-7 VM with custom resources
bash create_vm.sh Test-7 --memory 8GiB --disk 80GiB --cpus 8
```

### IP Address Auto-Generation
- Test-3 → 10.25.33.13
- Test-4 → 10.25.33.14
- Test-5 → 10.25.33.15
- Test-10 → 10.25.33.20
- Formula: `10.25.33.(10 + VM_NUMBER)`

### Options
- `--delete-existing` - Delete existing VM if it exists
- `--ip X.X.X.X` - Override auto-generated IP
- `--memory SIZE` - Override default memory (default: 4GiB)
- `--disk SIZE` - Override default disk size (default: 40GiB)
- `--cpus N` - Override default CPU count (default: 4)

## Post-Installation Steps

### After Ubuntu Installation Completes

1. **Remove ISO and change boot order:**
   ```bash
   bash post_install_vm.sh Test-3
   ```

2. **Get VNC connection info (domain ID may change after reboot):**
   ```bash
   # The script will show the new domain ID and VNC socket
   bash post_install_vm.sh Test-3
   ```

3. **Connect via VNC:**
   ```bash
   # On dom0 (uses helper script - automatically handles port conflicts)
   bash connect_vnc.sh Test-3
   
   # From Ubuntu (SSH tunnel - use port shown by helper script)
   ssh -N -L 5901:127.0.0.1:5901 root@10.25.33.10
   
   # Connect VNC client to: 127.0.0.1:5901
   ```
   
   **Note**: VNC ports are auto-assigned:
   - Test-3 → port 5901
   - Test-4 → port 5902
   - Test-5 → port 5903
   - Formula: `5900 + VM_NUMBER`

## Important Notes

### Domain ID Changes After Reboot
- When VM reboots, domain ID changes
- VNC socket path changes: `/var/run/xen/vnc-<NEW_DOMID>`
- Re-run `post_install_vm.sh` to get updated VNC info

### Boot Order
- **During installation**: CD first, then disk (`order=dc`)
- **After installation**: Disk only (`order=d`)
- Post-install script automatically changes this

### Network Configuration
- All VMs use static IP: `10.25.33.X/24`
- Gateway: `10.25.33.254`
- DNS: `8.8.8.8`
- Configure during Ubuntu installation

## Troubleshooting

### VM Creation Fails
- Check if VGS ISO Storage SR is accessible: `bash fix_vgs_sr_mount.sh`
- Verify ISO file exists: `ls -lh /mnt/iso-storage/`

### VNC Connection Fails
- Use helper script: `bash connect_vnc.sh Test-3` (automatically handles conflicts)
- Check domain ID: `xe vm-param-get uuid=<VM_UUID> param-name=dom-id`
- Verify VM is running: `xe vm-list name-label="Test-3" params=power-state`
- Check VNC socket exists: `ls -l /var/run/xen/vnc-<DOMID>`
- If port is in use: The helper script will kill old processes automatically

### VM Won't Boot After Installation
- Verify boot order: `xe vm-param-get uuid=<VM_UUID> param-name=HVM-boot-params`
- Should be `order=d` (disk only)
- Check if CD still has ISO: `xe vbd-list vm-uuid=<VM_UUID> type=CD`

## File Locations

- **VM Creation Script**: `create_vm.sh`
- **Post-Install Script**: `post_install_vm.sh`
- **VNC Connection Helper**: `connect_vnc.sh` (handles port conflicts automatically)
- **SR Mount Fix**: `fix_vgs_sr_mount.sh`
- **Solution Summary**: `SOLUTION_SUMMARY.md`
- **Post-Install Guide**: `POST_INSTALLATION_STEPS.md`
