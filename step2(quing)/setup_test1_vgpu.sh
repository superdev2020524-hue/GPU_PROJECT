#!/bin/bash
================================================================================
# Script: Setup TEST-1 VM with vGPU-Stub Device
# Purpose: Configure and start TEST-1 with vGPU stub device
# Config: pool_id=A, priority=high, vm_id=1
# Based on: Successful TEST-2 implementation (pool_id=B, priority=high, vm_id=200)
================================================================================

set -e  # Exit on error

echo "=========================================="
echo "vGPU Stub Setup for TEST-1"
echo "=========================================="
echo ""

# Configuration
VM_NAME="Test-1"
POOL_ID="A"
PRIORITY="high"
VM_ID="1"

echo "Target VM: $VM_NAME"
echo "Configuration:"
echo "  - pool_id: $POOL_ID"
echo "  - priority: $PRIORITY"
echo "  - vm_id: $VM_ID"
echo ""

# Step 1: Get VM UUID
echo "[Step 1] Finding VM UUID..."
VM_UUID=$(xe vm-list name-label="$VM_NAME" params=uuid --minimal)

if [ -z "$VM_UUID" ]; then
    echo "❌ ERROR: VM '$VM_NAME' not found!"
    echo "Available VMs:"
    xe vm-list is-control-domain=false params=name-label,uuid
    exit 1
fi

echo "✅ Found VM UUID: $VM_UUID"
echo ""

# Step 2: Check current power state
echo "[Step 2] Checking VM power state..."
POWER_STATE=$(xe vm-list uuid=$VM_UUID params=power-state --minimal)
echo "Current power state: $POWER_STATE"

if [ "$POWER_STATE" = "running" ]; then
    echo "⚠️  VM is currently running. Shutting down to apply configuration..."
    xe vm-shutdown uuid=$VM_UUID --force
    echo "Waiting for shutdown..."
    sleep 5
fi
echo ""

# Step 3: Configure device-model-args
echo "[Step 3] Configuring vGPU stub device..."
DEVICE_ARGS="-device vgpu-stub,pool_id=$POOL_ID,priority=$PRIORITY,vm_id=$VM_ID"
echo "Device arguments: $DEVICE_ARGS"

xe vm-param-set uuid=$VM_UUID platform:device-model-args="$DEVICE_ARGS"

# Verify the setting
CONFIGURED=$(xe vm-param-get uuid=$VM_UUID param-name=platform param-key=device-model-args)
echo "✅ Configuration applied: $CONFIGURED"
echo ""

# Step 4: Store configuration in xenstore (will be written when VM starts)
echo "[Step 4] Configuration will be written to xenstore on VM start"
echo ""

# Step 5: Start the VM
echo "[Step 5] Starting VM..."
xe vm-start uuid=$VM_UUID

echo "Waiting for VM to initialize..."
sleep 8

# Step 6: Verify VM started successfully
echo ""
echo "[Step 6] Verifying VM startup..."
POWER_STATE=$(xe vm-list uuid=$VM_UUID params=power-state --minimal)
DOM_ID=$(xe vm-list uuid=$VM_UUID params=dom-id --minimal)

echo "Power state: $POWER_STATE"
echo "Domain ID: $DOM_ID"

if [ "$POWER_STATE" != "running" ] || [ "$DOM_ID" = "-1" ]; then
    echo "❌ ERROR: VM failed to start properly!"
    echo "Check /var/log/daemon.log for errors:"
    echo "  tail -50 /var/log/daemon.log | grep qemu-dm-$DOM_ID"
    exit 1
fi

echo "✅ VM started successfully with Domain ID: $DOM_ID"
echo ""

# Step 7: Verify xenstore
echo "[Step 7] Verifying xenstore configuration..."
XENSTORE_ARGS=$(xenstore-read /local/domain/$DOM_ID/platform/device-model-args 2>/dev/null || echo "Not found")
echo "XenStore device-model-args: $XENSTORE_ARGS"

if [ "$XENSTORE_ARGS" = "$DEVICE_ARGS" ]; then
    echo "✅ XenStore configuration matches"
else
    echo "⚠️  XenStore value differs or not found"
fi
echo ""

# Step 8: Check QEMU command line in logs
echo "[Step 8] Checking QEMU command line in logs..."
echo "Searching for vgpu-stub in daemon.log..."

# Check for device-model-args being read
if grep -q "qemu-dm-$DOM_ID.*Adding device-model-args from xenstore" /var/log/daemon.log; then
    echo "✅ qemu-wrapper successfully read device-model-args from xenstore"
    grep "qemu-dm-$DOM_ID.*Adding device-model-args" /var/log/daemon.log | tail -1
else
    echo "⚠️  Warning: Device-model-args reading not found in logs"
fi

echo ""

# Check for device in QEMU exec line
if grep -q "qemu-dm-$DOM_ID.*vgpu-stub.*pool_id=$POOL_ID" /var/log/daemon.log; then
    echo "✅ vGPU stub device found in QEMU command line"
    echo ""
    echo "Full QEMU command line (last 200 chars):"
    grep "qemu-dm-$DOM_ID.*Exec:.*vgpu-stub" /var/log/daemon.log | tail -1 | rev | cut -c1-200 | rev
else
    echo "⚠️  Warning: vGPU stub device not found in QEMU command line"
    echo "Recent QEMU logs:"
    grep "qemu-dm-$DOM_ID" /var/log/daemon.log | tail -5
fi

echo ""
echo ""

# Step 9: Create verification summary
echo "=========================================="
echo "SETUP SUMMARY"
echo "=========================================="
echo ""
echo "VM Configuration:"
echo "  Name:      $VM_NAME"
echo "  UUID:      $VM_UUID"
echo "  Domain ID: $DOM_ID"
echo "  State:     $POWER_STATE"
echo ""
echo "vGPU Stub Configuration:"
echo "  Pool ID:   $POOL_ID"
echo "  Priority:  $PRIORITY"
echo "  VM ID:     $VM_ID"
echo ""
echo "Verification Status:"
echo "  [✓] VM started successfully"
echo "  [✓] Domain ID assigned (not -1)"
echo "  [✓] device-model-args configured"

if grep -q "qemu-dm-$DOM_ID.*vgpu-stub" /var/log/daemon.log; then
    echo "  [✓] Device in QEMU command line"
else
    echo "  [?] Device in QEMU command line - CHECK MANUALLY"
fi

echo ""
echo "Next Steps:"
echo "==========="
echo ""
echo "1. Verify device in guest (SSH into TEST-1):"
echo "   ssh user@test-1-ip"
echo "   lspci | grep -i 'processing accelerators\\|red hat\\|1af4:1111'"
echo ""
echo "2. Get detailed device info in guest:"
echo "   lspci -vvv | grep -A 20 'Processing accelerators'"
echo ""
echo "3. Test MMIO registers in guest (create and run test program):"
echo "   # See step2(quing)/vgpu-stub_enhance/complete.txt lines 652-696"
echo ""
echo "4. Compare with TEST-2 configuration:"
echo "   xe vm-list name-label=\"Test-2\" params=dom-id --minimal"
echo "   # TEST-2 should have: pool_id=B, priority=high, vm_id=200"
echo "   # TEST-1 now has: pool_id=A, priority=high, vm_id=1"
echo ""
echo "5. Check logs for any errors:"
echo "   tail -50 /var/log/daemon.log | grep -i error"
echo ""
echo "=========================================="
echo "Setup completed!"
echo "=========================================="
echo ""
echo "Log files for reference:"
echo "  - QEMU logs: /var/log/daemon.log"
echo "  - XenStore: xenstore-ls /local/domain/$DOM_ID"
echo ""
