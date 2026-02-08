#!/bin/bash
# Delete existing Test-3 VM

set -euo pipefail

VM_NAME="Test-3"

echo "========================================================================"
echo "  Deleting existing VM: $VM_NAME"
echo "========================================================================"
echo ""

# Find VM UUID
VM_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal 2>/dev/null | head -1)

if [ -z "$VM_UUID" ]; then
    echo "VM '$VM_NAME' not found. Nothing to delete."
    exit 0
fi

echo "Found VM: $VM_NAME"
echo "UUID: $VM_UUID"
echo ""

# Check power state
POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "halted")
echo "Power state: $POWER_STATE"

# Shutdown if running
if [ "$POWER_STATE" != "halted" ]; then
    echo "VM is running. Shutting down..."
    xe vm-shutdown uuid="$VM_UUID" force=true || xe vm-shutdown uuid="$VM_UUID"
    echo "Waiting for shutdown..."
    sleep 5
    
    # Force stop if still running
    POWER_STATE=$(xe vm-param-get uuid="$VM_UUID" param-name=power-state 2>/dev/null || echo "halted")
    if [ "$POWER_STATE" != "halted" ]; then
        echo "Force stopping..."
        xe vm-destroy uuid="$VM_UUID"
    fi
fi

# Delete VM
echo "Deleting VM..."
xe vm-uninstall uuid="$VM_UUID" force=true

echo ""
echo "========================================================================"
echo "  ✓ VM '$VM_NAME' deleted successfully"
echo "========================================================================"
