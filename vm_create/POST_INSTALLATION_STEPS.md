# Post-Installation Steps for Ubuntu VMs

## Overview
After Ubuntu installation completes and the system reboots, you need to:
1. Remove the ISO from CD drive
2. Change boot order to boot from hard disk (not CD)
3. Update VNC connection (domain ID changes after reboot)

## Step 1: Remove ISO from CD Drive

After installation completes and VM reboots, the ISO is still in the CD drive. Remove it:

```bash
# On dom0
VM_NAME="Test-3"
VM_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal)

# Find CD VBD
CD_VBD=$(xe vbd-list vm-uuid="$VM_UUID" type=CD params=uuid --minimal | head -1)

# Eject ISO
xe vbd-eject uuid="$CD_VBD"
echo "✓ ISO removed from CD drive"
```

## Step 2: Change Boot Order

Set boot order to hard disk first (remove CD from boot sequence):

```bash
# Set boot order to disk only (no CD)
xe vm-param-set uuid="$VM_UUID" HVM-boot-params:order=d

# Or use BIOS order with disk first
xe vm-param-set uuid="$VM_UUID" HVM-boot-policy="BIOS order"
xe vm-param-set uuid="$VM_UUID" HVM-boot-params:order=d

echo "✓ Boot order set to disk only"
```

## Step 3: Get New VNC Connection (After Reboot)

After the VM reboots, the domain ID changes. Get the new VNC socket:

```bash
# Get current domain ID
DOMID=$(xe vm-param-get uuid="$VM_UUID" param-name=dom-id 2>/dev/null || echo "-1")

if [ "$DOMID" != "-1" ]; then
    VNC_SOCKET="/var/run/xen/vnc-$DOMID"
    echo "VNC socket: $VNC_SOCKET"
    echo "Domain ID: $DOMID"
    
    # Bridge to TCP (if needed)
    echo "To bridge VNC to TCP:"
    echo "  socat TCP-LISTEN:5901,fork,reuseaddr UNIX-CONNECT:$VNC_SOCKET &"
else
    echo "VM is not running"
fi
```

## Step 4: Connect via VNC

### On dom0:
```bash
# Bridge VNC socket to TCP (use new domain ID)
DOMID=$(xe vm-param-get uuid="$VM_UUID" param-name=dom-id)
socat TCP-LISTEN:5901,fork,reuseaddr UNIX-CONNECT:/var/run/xen/vnc-$DOMID &
```

### From Ubuntu (SSH tunnel):
```bash
# Create SSH tunnel
ssh -N -L 5901:127.0.0.1:5901 root@10.25.33.10
```

### Connect VNC client:
- Host: `127.0.0.1`
- Port: `5901`

## Complete Post-Installation Script

```bash
#!/bin/bash
# Post-installation configuration for Ubuntu VM

set -euo pipefail

VM_NAME="${1:-Test-3}"

echo "========================================================================"
echo "  Post-Installation Configuration: $VM_NAME"
echo "========================================================================"
echo ""

# Get VM UUID
VM_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)
if [ -z "$VM_UUID" ]; then
    echo "ERROR: VM '$VM_NAME' not found"
    exit 1
fi

echo "VM UUID: $VM_UUID"
echo ""

# Step 1: Remove ISO from CD
echo "STEP 1: Removing ISO from CD drive..."
echo "--------------------------------------"
CD_VBD=$(xe vbd-list vm-uuid="$VM_UUID" type=CD params=uuid --minimal | head -1)
if [ -n "$CD_VBD" ]; then
    xe vbd-eject uuid="$CD_VBD" 2>/dev/null || true
    echo "✓ ISO removed from CD drive"
else
    echo "⚠ No CD drive found"
fi
echo ""

# Step 2: Change boot order
echo "STEP 2: Setting boot order to disk only..."
echo "--------------------------------------------"
xe vm-param-set uuid="$VM_UUID" HVM-boot-params:order=d
echo "✓ Boot order set to disk only"
echo ""

# Step 3: Get VNC info
echo "STEP 3: VNC Connection Info..."
echo "-------------------------------"
DOMID=$(xe vm-param-get uuid="$VM_UUID" param-name=dom-id 2>/dev/null || echo "-1")
POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")

if [ "$POWER_STATE" = "running" ] && [ "$DOMID" != "-1" ]; then
    VNC_SOCKET="/var/run/xen/vnc-$DOMID"
    echo "Domain ID: $DOMID"
    echo "VNC socket: $VNC_SOCKET"
    echo ""
    echo "To connect via VNC:"
    echo "  1. On dom0: socat TCP-LISTEN:5901,fork,reuseaddr UNIX-CONNECT:$VNC_SOCKET &"
    echo "  2. From Ubuntu: ssh -N -L 5901:127.0.0.1:5901 root@10.25.33.10"
    echo "  3. Connect VNC client to: 127.0.0.1:5901"
else
    echo "VM is not running (power state: $POWER_STATE)"
    echo "Start the VM first: xe vm-start uuid=$VM_UUID"
fi
echo ""

echo "========================================================================"
echo "  ✓ Post-installation configuration complete"
echo "========================================================================"
```

## Usage

```bash
# Run post-installation steps for Test-3
bash post_install_vm.sh Test-3

# Or for any other VM
bash post_install_vm.sh Test-4
```

## Notes

- **Domain ID changes**: After reboot, the domain ID changes, so VNC socket path changes
- **Boot order**: Must be changed to disk only, otherwise VM will try to boot from CD again
- **CD removal**: Not strictly necessary, but prevents accidental boot from CD
- **VNC persistence**: The `socat` process runs in background, so it persists until you kill it
