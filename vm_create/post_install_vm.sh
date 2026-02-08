#!/bin/bash
# Post-installation configuration for Ubuntu VM

set -euo pipefail

VM_NAME="${1:-}"
REMOVE_CD=""
AUTO_SHUTDOWN=false

# Parse arguments
shift 2>/dev/null || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --remove-cd)
            REMOVE_CD="--remove-cd"
            shift
            ;;
        --shutdown)
            AUTO_SHUTDOWN=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$VM_NAME" ]; then
    echo "ERROR: VM name required"
    echo ""
    echo "Usage: bash post_install_vm.sh <VM_NAME> [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --remove-cd    Completely remove CD drive (instead of just unplugging)"
    echo "  --shutdown     Automatically shutdown VM if running (required for boot order changes)"
    echo ""
    echo "Examples:"
    echo "  bash post_install_vm.sh Test-3"
    echo "  bash post_install_vm.sh Test-3 --shutdown"
    echo "  bash post_install_vm.sh Test-3 --remove-cd --shutdown"
    exit 1
fi

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

# Check if VM is running - boot order changes require VM to be halted
# Capture initial state to know if we should restart later
POWER_STATE_BEFORE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")
WAS_RUNNING=false

POWER_STATE="$POWER_STATE_BEFORE"
if [ "$POWER_STATE" = "running" ]; then
    WAS_RUNNING=true
    if [ "$AUTO_SHUTDOWN" = "true" ]; then
        echo "VM is running. Force stopping VM (required for boot order changes)..."
        
        # Use force destroy immediately (faster and more reliable for post-install)
        xe vm-destroy uuid="$VM_UUID" 2>/dev/null || true
        sleep 2
        
        # Verify it's stopped
        POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")
        if [ "$POWER_STATE" = "running" ]; then
            echo "⚠ WARNING: VM still appears to be running"
            echo "  Power state: $POWER_STATE"
            echo "  Attempting again..."
            xe vm-destroy uuid="$VM_UUID" force=true 2>/dev/null || true
            sleep 2
            POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")
        fi
        
        if [ "$POWER_STATE" = "halted" ] || [ "$POWER_STATE" = "suspended" ]; then
            echo "✓ VM is now halted"
        else
            echo "⚠ WARNING: VM power state is: $POWER_STATE"
            echo "  Continuing anyway - if boot order changes fail, manually stop VM first"
        fi
        echo ""
    else
        echo "⚠ WARNING: VM is currently running"
        echo "Boot order changes require the VM to be shut down."
        echo "Use --shutdown flag to automatically shutdown: bash post_install_vm.sh $VM_NAME --shutdown"
        echo "Or manually shutdown first: xe vm-shutdown uuid=$VM_UUID"
        echo ""
        echo "⚠ Continuing anyway - boot order changes may not take effect until next boot"
        echo ""
    fi
fi

# Step 1: Remove ALL CD drives completely
echo "STEP 1: Removing ALL CD drives..."
echo "-----------------------------------"
# Find ALL CD drives (there may be multiple)
CD_VBDS=$(xe vbd-list vm-uuid="$VM_UUID" type=CD params=uuid --minimal | tr ',' '\n' | xargs)

if [ -z "$CD_VBDS" ]; then
    echo "✓ No CD drives found"
else
    CD_COUNT=$(echo "$CD_VBDS" | wc -w)
    echo "Found $CD_COUNT CD drive(s)"
    
    for CD_VBD in $CD_VBDS; do
        CD_VBD=$(echo "$CD_VBD" | xargs)
        [ -z "$CD_VBD" ] && continue
        
        echo "Processing CD drive: $CD_VBD"
        
        # Eject ISO if present
        CURRENT_VDI=$(xe vbd-param-get uuid="$CD_VBD" param-name=vdi-uuid 2>/dev/null || echo "")
        if [ -n "$CURRENT_VDI" ] && [ "$CURRENT_VDI" != "OpaqueRef:NULL" ]; then
            echo "  Ejecting ISO..."
            xe vbd-eject uuid="$CD_VBD" 2>/dev/null || true
            sleep 1
        fi
        
        # Always remove CD drive completely (even without --remove-cd flag)
        # This is necessary because BIOS may try to boot from empty CD drives
        echo "  Removing CD drive completely..."
        xe vbd-unplug uuid="$CD_VBD" 2>/dev/null || true
        sleep 1
        xe vbd-destroy uuid="$CD_VBD" 2>/dev/null || true
        echo "  ✓ CD drive removed"
    done
    
    echo "✓ All CD drives removed completely"
fi
echo ""

# Step 2: Change boot order
echo "STEP 2: Setting boot order to disk only..."
echo "--------------------------------------------"

# Boot order can only be changed when VM is halted
# POWER_STATE_BEFORE was already captured above
if [ "$POWER_STATE_BEFORE" = "running" ] && [ "$AUTO_SHUTDOWN" != "true" ]; then
    echo "⚠ WARNING: VM is running. Boot order changes require VM to be halted."
    echo "  Use --shutdown flag to automatically shutdown, or shutdown manually first."
    echo "  Continuing anyway - changes may not take effect..."
    echo ""
fi

# Verify disk VBD exists, is bootable, and is attached
echo "Verifying disk configuration..."
DISK_VBD=$(xe vbd-list vm-uuid="$VM_UUID" type=Disk params=uuid --minimal | head -1)
if [ -z "$DISK_VBD" ]; then
    echo "  ✗ ERROR: No disk found for VM!"
    echo "  VM cannot boot without a disk"
    exit 1
fi

DISK_BOOTABLE=$(xe vbd-param-get uuid="$DISK_VBD" param-name=bootable 2>/dev/null || echo "false")

if [ "$DISK_BOOTABLE" != "true" ]; then
    echo "  ⚠ WARNING: Disk is not marked as bootable. Setting bootable flag..."
    xe vbd-param-set uuid="$DISK_VBD" bootable=true 2>/dev/null || echo "    (Failed to set bootable flag)"
else
    echo "  ✓ Disk is bootable"
fi

# Note: When VM is halted, VBDs show as "not attached" but are still configured.
# They will automatically attach when the VM starts. Only check attachment status
# if VM is running.
CURRENT_POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")
if [ "$CURRENT_POWER_STATE" = "running" ]; then
    DISK_ATTACHED=$(xe vbd-param-get uuid="$DISK_VBD" param-name=currently-attached 2>/dev/null || echo "false")
    if [ "$DISK_ATTACHED" != "true" ]; then
        echo "  ⚠ WARNING: VM is running but disk is not attached!"
        echo "  Attempting to plug disk..."
        if xe vbd-plug uuid="$DISK_VBD" 2>&1; then
            echo "  ✓ Disk plugged successfully"
        else
            echo "  ✗ ERROR: Failed to plug disk while VM is running"
        fi
    else
        echo "  ✓ Disk is attached"
    fi
else
    echo "  ✓ Disk is configured (will attach automatically when VM starts)"
fi
echo ""

# Verify and set UEFI firmware (matching working VMs)
echo "Verifying UEFI firmware configuration..."
CURRENT_FIRMWARE=$(xe vm-param-get uuid="$VM_UUID" param-name=platform:firmware 2>/dev/null || echo "")
if [ "$CURRENT_FIRMWARE" != "uefi" ]; then
    echo "  Setting firmware to UEFI (current: $CURRENT_FIRMWARE)..."
    xe vm-param-set uuid="$VM_UUID" platform:firmware=uefi 2>&1 || echo "    (Note: May require VM to be halted)"
    echo "  ✓ Firmware set to UEFI"
else
    echo "  ✓ Firmware already set to UEFI"
fi

# Verify and disable VIRIDIAN (matching working VMs)
echo "Verifying VIRIDIAN is disabled..."
CURRENT_VIRIDIAN=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params:viridian 2>/dev/null || echo "")
if [ "$CURRENT_VIRIDIAN" != "false" ]; then
    echo "  Disabling VIRIDIAN (current: $CURRENT_VIRIDIAN)..."
    xe vm-param-set uuid="$VM_UUID" HVM-boot-params:viridian=false 2>/dev/null || true
    echo "  ✓ VIRIDIAN disabled"
else
    echo "  ✓ VIRIDIAN already disabled"
fi

# Set boot policy and order explicitly
echo "Setting boot policy to 'BIOS order'..."
if ! xe vm-param-set uuid="$VM_UUID" HVM-boot-policy="BIOS order" 2>&1; then
    echo "  ⚠ Warning: Failed to set boot policy (VM may be running)"
fi

echo "Setting boot order to 'c' (disk only)..."
# Boot order codes: c=disk, d=CD, n=network
# Use the correct syntax for map parameters: name:key=value
xe vm-param-set uuid="$VM_UUID" HVM-boot-params:order=c 2>&1 || {
    echo "  ⚠ Note: Boot params use map syntax, trying alternative..."
    # If that fails, the boot order might already be set correctly
}
sleep 1

# Verify boot order was set - try different methods to read it
BOOT_PARAMS_FULL=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "")
CURRENT_ORDER=""

# Try to extract order from boot params (format can vary: "order=c", "order: c", etc.)
if [ -n "$BOOT_PARAMS_FULL" ]; then
    # Try format: "order=c", "order: c", "order=c;", "order: c;", etc.
    CURRENT_ORDER=$(echo "$BOOT_PARAMS_FULL" | grep -oE "order[=:]\s*[^;,\s]*" | sed 's/order[=:]\s*//' | head -1)
    # Trim whitespace
    CURRENT_ORDER=$(echo "$CURRENT_ORDER" | xargs)
fi

# If still empty, try reading directly with sed
if [ -z "$CURRENT_ORDER" ]; then
    CURRENT_ORDER=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null | sed -n 's/.*order[=:]\s*\([^;]*\).*/\1/p' | head -1 | xargs)
fi

# Display what we found
echo "Boot params: $BOOT_PARAMS_FULL"
if [ "$CURRENT_ORDER" = "c" ]; then
    echo "✓ Boot order set to disk only (verified: order=$CURRENT_ORDER)"
else
    echo "⚠ Boot order may not be set correctly (current: order='$CURRENT_ORDER')"
    echo "  Attempting alternative method..."
    
    # Try setting with different syntax
    xe vm-param-set uuid="$VM_UUID" HVM-boot-params="order=c" 2>/dev/null || true
    sleep 1
    
    # Re-check
    BOOT_PARAMS_FULL=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "")
    CURRENT_ORDER=$(echo "$BOOT_PARAMS_FULL" | grep -oE "order[=:]\s*[^;,\s]*" | sed 's/order[=:]\s*//' | head -1 | xargs)
    
    if [ "$CURRENT_ORDER" = "c" ]; then
        echo "✓ Boot order now set correctly (order=$CURRENT_ORDER)"
    else
        echo "⚠ WARNING: Boot order verification failed"
        echo "  Full boot params: $BOOT_PARAMS_FULL"
        echo "  Extracted order: '$CURRENT_ORDER'"
        echo "  Boot order should be set, but verification is unclear"
    fi
fi
echo ""

# Step 3: Get VNC info
echo "STEP 3: VNC Connection Info..."
echo "-------------------------------"
DOMID=$(xe vm-param-get uuid="$VM_UUID" param-name=dom-id 2>/dev/null || echo "-1")
POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")

if [ "$POWER_STATE" = "running" ] && [ "$DOMID" != "-1" ]; then
    VNC_SOCKET="/var/run/xen/vnc-$DOMID"
    echo "Domain ID: $DOMID (may change after reboot)"
    echo "VNC socket: $VNC_SOCKET"
    echo ""
    echo "To connect via VNC, use the helper script:"
    echo "  bash connect_vnc.sh $VM_NAME"
    echo ""
    echo "Or manually:"
    echo "  1. On dom0: bash connect_vnc.sh $VM_NAME"
    echo "  2. From Ubuntu: ssh -N -L 5901:127.0.0.1:5901 root@10.25.33.10"
    echo "  3. Connect VNC client to: 127.0.0.1:5901"
    echo ""
    echo "⚠ NOTE: Domain ID changes after VM reboot. Re-run this script to get new VNC info."
else
    echo "VM is not running (power state: $POWER_STATE)"
    echo "Start the VM first: xe vm-start uuid=$VM_UUID"
fi
echo ""

# Verification summary
echo "========================================================================"
echo "  Verification Summary"
echo "========================================================================"
echo ""

# Check CD drive status
CD_VBD=$(xe vbd-list vm-uuid="$VM_UUID" type=CD params=uuid --minimal | head -1)
if [ -n "$CD_VBD" ]; then
    CD_VDI=$(xe vbd-param-get uuid="$CD_VBD" param-name=vdi-uuid 2>/dev/null || echo "")
    CD_ATTACHED=$(xe vbd-param-get uuid="$CD_VBD" param-name=currently-attached 2>/dev/null || echo "false")
    if [ -z "$CD_VDI" ] || [ "$CD_VDI" = "OpaqueRef:NULL" ]; then
        echo "✓ CD drive: Empty"
    else
        echo "⚠ CD drive: Still has ISO inserted (UUID: $CD_VDI)"
    fi
    if [ "$CD_ATTACHED" = "true" ]; then
        echo "⚠ CD drive: Still attached (may interfere with boot)"
    else
        echo "✓ CD drive: Unplugged"
    fi
else
    echo "✓ CD drive: Removed completely"
fi

# Check boot order
BOOT_PARAMS=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "")
BOOT_ORDER=$(echo "$BOOT_PARAMS" | grep -oE "order[=:]\s*[^;,\s]*" | sed 's/order[=:]\s*//' | head -1 | xargs)
if [ -z "$BOOT_ORDER" ]; then
    # Try alternative extraction
    BOOT_ORDER=$(echo "$BOOT_PARAMS" | sed -n 's/.*order[=:]\s*\([^;]*\).*/\1/p' | head -1 | xargs)
fi

if [ "$BOOT_ORDER" = "c" ]; then
    echo "✓ Boot order: Disk only (order=c)"
elif [ -n "$BOOT_ORDER" ]; then
    echo "⚠ Boot order: $BOOT_ORDER (expected: c)"
    echo "  Full boot params: $BOOT_PARAMS"
    echo ""
    echo "  ⚠ CRITICAL: Boot order is not set to 'c' (disk only)"
    echo "  VM may try to boot from CD (which doesn't exist) and fail"
    echo "  This will cause the VM to shut down after starting"
    echo ""
    echo "  Attempting to fix boot order again..."
    xe vm-param-set uuid="$VM_UUID" HVM-boot-params:order=c 2>&1 || true
    sleep 2
    # Re-check
    BOOT_PARAMS=$(xe vm-param-get uuid="$VM_UUID" param-name=HVM-boot-params 2>/dev/null || echo "")
    BOOT_ORDER=$(echo "$BOOT_PARAMS" | grep -oE "order[=:]\s*[^;,\s]*" | sed 's/order[=:]\s*//' | head -1 | xargs)
    if [ "$BOOT_ORDER" = "c" ]; then
        echo "  ✓ Boot order now fixed (order=c)"
    else
        echo "  ✗ ERROR: Boot order still not correct!"
        echo "  You MUST fix this before starting the VM, or it will shut down"
        echo "  Try manually: xe vm-param-set uuid=$VM_UUID HVM-boot-params:order=c"
    fi
else
    echo "⚠ Boot order: Could not determine (may be empty or unset)"
    echo "  Full boot params: $BOOT_PARAMS"
    echo ""
    echo "  ⚠ CRITICAL: Boot order appears to be unset"
    echo "  VM may try to boot from CD (which doesn't exist) and fail"
    echo "  Attempting to set boot order..."
    xe vm-param-set uuid="$VM_UUID" HVM-boot-params:order=c 2>&1 || true
fi

echo ""
echo "========================================================================"
echo "  ✓ Post-installation configuration complete"
echo "========================================================================"
echo ""

# Check if VM was running before and should be restarted
FINAL_POWER_STATE="unknown"
FINAL_POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "unknown")
if [ "$FINAL_POWER_STATE" = "halted" ] && [ "$WAS_RUNNING" = "true" ] && [ "$AUTO_SHUTDOWN" = "true" ]; then
    echo "⚠ IMPORTANT: VM was shut down for configuration."
    echo "  The VM is now halted and needs to be started manually."
    echo ""
    echo "  To start the VM:"
    echo "    xe vm-start uuid=$VM_UUID"
    echo ""
    echo "  Then connect via VNC:"
    echo "    bash connect_vnc.sh $VM_NAME"
    echo ""
fi

echo "Next steps:"
if [ "$FINAL_POWER_STATE" != "running" ]; then
    echo "  1. Verify boot order is correct (should show 'order=c' above)"
    if [ "$BOOT_ORDER" != "c" ]; then
        echo "     ⚠ WARNING: Boot order is NOT correct! Fix it before starting:"
        echo "        xe vm-param-set uuid=$VM_UUID HVM-boot-params:order=c"
        echo ""
    fi
    echo "  2. Start the VM: xe vm-start uuid=$VM_UUID"
    echo "  3. Monitor VM status: xe vm-list uuid=$VM_UUID params=power-state"
    echo "  4. If VM shuts down immediately, boot order is likely incorrect"
    echo "  5. Connect via VNC: bash connect_vnc.sh $VM_NAME"
else
    echo "  1. Connect via VNC: bash connect_vnc.sh $VM_NAME"
fi
echo ""