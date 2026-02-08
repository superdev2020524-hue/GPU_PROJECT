#!/bin/bash
#================================================================================
#  VM Creation Workaround: Works without requiring new ISO SR
#================================================================================
#
# This script attempts to:
#  1. Fix existing SMB SR if possible, OR
#  2. Use any accessible ISO SR, OR  
#  3. Provide clear instructions if neither works
#
#================================================================================

set -euo pipefail

VM_NAME="Test-3"
VM_IP="10.25.33.13"
GATEWAY="10.25.33.254"
DNS="8.8.8.8"
DISK_SIZE="40GiB"
MEMORY="4GiB"
VCPUS=4
CD_DEVICE=3

echo "========================================================================"
echo "  Creating VM: $VM_NAME (Workaround Method)"
echo "========================================================================"
echo ""

# Mount VGS and find ISO (same as before)
echo "STEP 1: Mounting VGS and finding ISO..."
echo "----------------------------------------"

LV_LINE=$(lvs --noheadings -o vg_name,lv_name 2>/dev/null | grep -i iso_download | head -1)
VG_NAME=$(echo "$LV_LINE" | awk '{print $1}')
LV_PATH="/dev/$VG_NAME/iso_download"
MOUNT_POINT="/mnt/iso-storage"

mkdir -p "$MOUNT_POINT"
if ! mountpoint -q "$MOUNT_POINT"; then
    mount "$LV_PATH" "$MOUNT_POINT"
fi

ISO_FILE=$(find "$MOUNT_POINT" -name "*ubuntu*server*.iso" -type f 2>/dev/null | head -1)
[ -z "$ISO_FILE" ] && ISO_FILE=$(find "$MOUNT_POINT" -name "*ubuntu*.iso" -type f 2>/dev/null | head -1)
ISO_NAME=$(basename "$ISO_FILE")

echo "✓ Found ISO: $ISO_NAME"
echo ""

# Try to fix SMB SR
echo "STEP 2: Attempting to fix SMB ISO SR..."
echo "----------------------------------------"

SMB_SR_UUID="097e2b8c-af1a-d945-1432-8c0e7d0163fa"
PBD_UUID=$(xe pbd-list sr-uuid="$SMB_SR_UUID" params=uuid --minimal | head -1)

if [ -n "$PBD_UUID" ]; then
    echo "Found SMB SR PBD: $PBD_UUID"
    
    # Try unplug and replug
    echo "Attempting to fix SMB SR mount..."
    xe pbd-unplug uuid="$PBD_UUID" 2>/dev/null || true
    sleep 2
    xe pbd-plug uuid="$PBD_UUID" 2>&1 | head -5 || true
    sleep 3
    
    # Check if it worked
    SR_MOUNT="/var/run/sr-mount/$SMB_SR_UUID"
    if [ -d "$SR_MOUNT" ] && mount | grep -F "$SR_MOUNT" >/dev/null 2>&1; then
        echo "✓ SMB SR is now accessible!"
        
        # Copy ISO to SMB SR if not already there
        if [ ! -f "$SR_MOUNT/$ISO_NAME" ]; then
            echo "Copying ISO to SMB SR..."
            cp "$ISO_FILE" "$SR_MOUNT/$ISO_NAME"
            xe sr-scan uuid="$SMB_SR_UUID"
            sleep 2
        fi
        
        # Find ISO VDI
        ISO_VDI_UUID=$(xe vdi-list sr-uuid="$SMB_SR_UUID" name-label="$ISO_NAME" params=uuid --minimal 2>/dev/null | head -1)
        
        if [ -n "$ISO_VDI_UUID" ]; then
            echo "✓ ISO available in SMB SR: $ISO_VDI_UUID"
            ISO_SR_UUID="$SMB_SR_UUID"
        fi
    else
        echo "✗ SMB SR still not accessible (network issue to 10.25.33.33)"
    fi
fi

# If SMB didn't work, check other ISO SRs
if [ -z "${ISO_SR_UUID:-}" ]; then
    echo ""
    echo "STEP 3: Checking other ISO SRs..."
    echo "----------------------------------"
    
    for SR_UUID in $(xe sr-list content-type=iso params=uuid --minimal | tr ',' '\n'); do
        SR_UUID=$(echo "$SR_UUID" | xargs)
        [ -z "$SR_UUID" ] && continue
        
        SR_TYPE=$(xe sr-list uuid="$SR_UUID" params=type --minimal 2>/dev/null || true)
        [ "$SR_TYPE" = "udev" ] && continue
        
        SR_MOUNT="/var/run/sr-mount/$SR_UUID"
        if [ -d "$SR_MOUNT" ] && mount | grep -F "$SR_MOUNT" >/dev/null 2>&1; then
            echo "Found accessible SR: $SR_UUID"
            
            # Copy ISO there
            echo "Copying ISO to accessible SR..."
            cp "$ISO_FILE" "$SR_MOUNT/$ISO_NAME"
            xe sr-scan uuid="$SR_UUID"
            sleep 2
            
            ISO_VDI_UUID=$(xe vdi-list sr-uuid="$SR_UUID" name-label="$ISO_NAME" params=uuid --minimal 2>/dev/null | head -1)
            if [ -n "$ISO_VDI_UUID" ]; then
                ISO_SR_UUID="$SR_UUID"
                echo "✓ ISO registered in SR: $SR_UUID"
                break
            fi
        fi
    done
fi

# If still no ISO SR, we can't proceed
if [ -z "${ISO_SR_UUID:-}" ] || [ -z "${ISO_VDI_UUID:-}" ]; then
    echo ""
    echo "========================================================================"
    echo "  ERROR: Cannot proceed without accessible ISO SR"
    echo "========================================================================"
    echo ""
    echo "The ISO file is ready at: $ISO_FILE"
    echo "But no ISO Storage Repository is accessible."
    echo ""
    echo "SOLUTIONS:"
    echo "----------"
    echo "1. Fix SMB server connectivity (10.25.33.33 must be reachable)"
    echo "2. Create VGS ISO SR via CLI: bash create_vgs_iso_sr_cli.sh"
    echo "3. Use XCP-ng Center GUI (if you can get network access working)"
    echo ""
    exit 1
fi

# Continue with VM creation (same as create_test3_vm.sh from here)
echo ""
echo "STEP 4: Creating VM..."
echo "----------------------"

TEMPLATE_UUID=$(xe template-list name-label="Other install media" params=uuid --minimal)
LOCAL_SR_UUID=$(xe sr-list type=lvm content-type=user params=uuid --minimal | head -1)
NET_UUID=$(xe network-list bridge=xenbr0 params=uuid --minimal | head -1)

VM_UUID=$(xe vm-install template-uuid="$TEMPLATE_UUID" new-name-label="$VM_NAME")
xe vm-param-set uuid="$VM_UUID" affinity=$(xe host-list --minimal)
xe vm-memory-limits-set uuid="$VM_UUID" static-min="$MEMORY" static-max="$MEMORY" dynamic-min="$MEMORY" dynamic-max="$MEMORY"
xe vm-param-set uuid="$VM_UUID" VCPUs-max="$VCPUS"
xe vm-param-set uuid="$VM_UUID" VCPUs-at-startup="$VCPUS"

DISK_VDI=$(xe vdi-create sr-uuid="$LOCAL_SR_UUID" name-label="${VM_NAME}-disk0" virtual-size="$DISK_SIZE" type=user)
DISK_VBD=$(xe vbd-create vm-uuid="$VM_UUID" vdi-uuid="$DISK_VDI" device=0 mode=RW type=Disk bootable=true)
VIF_UUID=$(xe vif-create vm-uuid="$VM_UUID" network-uuid="$NET_UUID" device=0)
CD_VBD=$(xe vbd-create vm-uuid="$VM_UUID" device="$CD_DEVICE" type=CD mode=RO)

xe vbd-insert uuid="$CD_VBD" vdi-uuid="$ISO_VDI_UUID"
xe vm-param-set uuid="$VM_UUID" HVM-boot-policy="BIOS order"
xe vm-param-set uuid="$VM_UUID" HVM-boot-params:order=dc

echo "✓ VM configured"
echo ""

echo "STEP 5: Starting VM..."
echo "----------------------"

xe vm-start uuid="$VM_UUID"
sleep 3

DOMID=$(xe vm-param-get uuid="$VM_UUID" param-name=dom-id 2>/dev/null || echo "-1")
if [ "$DOMID" != "-1" ]; then
    echo "✓ VM started! Domain ID: $DOMID"
    echo "VNC socket: /var/run/xen/vnc-$DOMID"
else
    echo "✗ VM failed to start"
    exit 1
fi
