#!/bin/bash
# Script to delete all VMs except DOM0, Test-1, and Test-2

set -e

echo "=========================================="
echo "VM Deletion Script"
echo "=========================================="
echo ""

# VMs to keep (case-insensitive matching)
KEEP_VMS=("DOM0" "Test-1" "Test-2")

# Function to check if VM should be kept
should_keep() {
    local vm_name="$1"
    for keep in "${KEEP_VMS[@]}"; do
        if [ "${vm_name,,}" = "${keep,,}" ]; then
            return 0
        fi
    done
    return 1
}

# List all VMs (excluding control domains)
echo "=== Listing all VMs (excluding control domains) ==="
xe vm-list is-control-domain=false params=uuid,name-label,power-state

echo ""
echo "=== VMs to keep: ${KEEP_VMS[*]} ==="
echo ""

# Get list of VMs to delete
VMS_TO_DELETE=()

# Get all VM UUIDs (excluding control domains)
ALL_VM_UUIDS=$(xe vm-list is-control-domain=false params=uuid --minimal | tr ',' '\n')

for uuid in $ALL_VM_UUIDS; do
    if [ -z "$uuid" ]; then
        continue
    fi
    # Get VM name
    vm_name=$(xe vm-list uuid=$uuid params=name-label --minimal 2>/dev/null)
    if [ -n "$vm_name" ]; then
        if ! should_keep "$vm_name"; then
            VMS_TO_DELETE+=("$uuid|$vm_name")
        fi
    fi
done

if [ ${#VMS_TO_DELETE[@]} -eq 0 ]; then
    echo "No VMs to delete. All VMs are in the keep list."
    exit 0
fi

echo "=== VMs to be deleted ==="
for vm_info in "${VMS_TO_DELETE[@]}"; do
    IFS='|' read -r uuid name <<< "$vm_info"
    state=$(xe vm-param-get uuid=$uuid param-name=power-state 2>/dev/null || echo "unknown")
    echo "  - $name (UUID: $uuid, State: $state)"
done

echo ""
read -p "Are you sure you want to delete these VMs? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Deletion cancelled."
    exit 0
fi

echo ""
echo "=== Starting deletion process ==="
echo ""

# Delete each VM
for vm_info in "${VMS_TO_DELETE[@]}"; do
    IFS='|' read -r uuid name <<< "$vm_info"
    
    echo "Processing: $name (UUID: $uuid)"
    
    # Get current power state
    state=$(xe vm-param-get uuid=$uuid param-name=power-state 2>/dev/null || echo "unknown")
    
    # Shutdown if running
    if [ "$state" = "running" ]; then
        echo "  Shutting down VM..."
        xe vm-shutdown uuid=$uuid --force || true
        # Wait for shutdown
        max_wait=30
        wait_count=0
        while [ "$(xe vm-param-get uuid=$uuid param-name=power-state 2>/dev/null || echo 'halted')" != "halted" ] && [ $wait_count -lt $max_wait ]; do
            sleep 1
            wait_count=$((wait_count + 1))
        done
        echo "  VM shutdown complete"
    fi
    
    # Delete VM
    echo "  Deleting VM..."
    if xe vm-destroy uuid=$uuid 2>/dev/null; then
        echo "  ✓ Successfully deleted: $name"
    else
        echo "  ⚠️  Warning: Failed to delete $name (may already be deleted)"
    fi
    echo ""
done

echo "=== Deletion complete ==="
echo ""
echo "=== Remaining VMs ==="
xe vm-list is-control-domain=false params=uuid,name-label,power-state
