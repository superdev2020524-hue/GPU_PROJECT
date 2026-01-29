#!/bin/bash
################################################################################
# Guest VM vGPU Stub Verification Script
# Purpose: Run this script INSIDE the guest VM to verify vGPU stub device
# Location: Copy to /tmp/ on the guest VM and run
################################################################################

echo "=========================================="
echo "vGPU Stub Device Verification (Guest)"
echo "=========================================="
echo ""

# Step 1: Check if device exists
echo "[Step 1] Searching for vGPU stub device..."
DEVICE_LINE=$(lspci | grep -i 'processing accelerators\|1af4:1111\|red hat')

if [ -z "$DEVICE_LINE" ]; then
    echo "❌ ERROR: vGPU stub device NOT found!"
    echo ""
    echo "Troubleshooting:"
    echo "1. Check if device was added to VM configuration (on Dom0):"
    echo "   xe vm-param-get uuid=<VM_UUID> param-name=platform param-key=device-model-args"
    echo ""
    echo "2. Check QEMU command line (on Dom0):"
    echo "   grep qemu-dm-<domid>.*vgpu-stub /var/log/daemon.log"
    echo ""
    echo "3. Try rescanning PCI bus:"
    echo "   sudo sh -c 'echo 1 > /sys/bus/pci/rescan'"
    echo ""
    echo "All PCI devices:"
    lspci
    exit 1
fi

echo "✅ Device found: $DEVICE_LINE"
DEVICE_ADDR=$(echo "$DEVICE_LINE" | awk '{print $1}')
echo "Device address: $DEVICE_ADDR"
echo ""

# Step 2: Get detailed device information
echo "[Step 2] Getting detailed device information..."
echo ""
lspci -vvv -s $DEVICE_ADDR

echo ""
echo ""

# Step 3: Check for MMIO region
echo "[Step 3] Verifying MMIO region..."
MMIO_REGION=$(lspci -vvv -s $DEVICE_ADDR | grep "Region 0:" | head -1)

if [ -z "$MMIO_REGION" ]; then
    echo "⚠️  Warning: MMIO Region 0 not found"
else
    echo "✅ $MMIO_REGION"
    MMIO_ADDR=$(echo "$MMIO_REGION" | grep -oP 'at \K[0-9a-f]+')
    MMIO_SIZE=$(echo "$MMIO_REGION" | grep -oP 'size=\K[^]]+')
    echo "   Address: 0x$MMIO_ADDR"
    echo "   Size: $MMIO_SIZE"
fi

echo ""

# Step 4: Check sysfs
echo "[Step 4] Checking sysfs entries..."
SYSFS_PATH="/sys/bus/pci/devices/0000:$DEVICE_ADDR"

if [ -d "$SYSFS_PATH" ]; then
    echo "✅ Sysfs entry exists: $SYSFS_PATH"
    
    echo ""
    echo "Device attributes:"
    echo "  Vendor: $(cat $SYSFS_PATH/vendor 2>/dev/null || echo 'N/A')"
    echo "  Device: $(cat $SYSFS_PATH/device 2>/dev/null || echo 'N/A')"
    echo "  Class:  $(cat $SYSFS_PATH/class 2>/dev/null || echo 'N/A')"
    
    if [ -f "$SYSFS_PATH/resource0" ]; then
        RESOURCE_SIZE=$(stat -c%s "$SYSFS_PATH/resource0" 2>/dev/null || echo "0")
        echo "  Resource0 size: $RESOURCE_SIZE bytes"
    fi
else
    echo "⚠️  Warning: Sysfs entry not found at $SYSFS_PATH"
fi

echo ""

# Step 5: Create and compile test program
echo "[Step 5] Creating MMIO register test program..."

cat > /tmp/test_vgpu_stub.c << 'TESTPROG'
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <string.h>

int main() {
    char resource_path[256];
    
    // Try to find the device automatically
    printf("=== vGPU Stub MMIO Register Test ===\n\n");
    
    // Search for the device
    FILE *lspci = popen("lspci | grep -i '1af4:1111\\|processing accelerators' | awk '{print $1}'", "r");
    if (!lspci) {
        fprintf(stderr, "Failed to run lspci\n");
        return 1;
    }
    
    char device_addr[32];
    if (!fgets(device_addr, sizeof(device_addr), lspci)) {
        fprintf(stderr, "Device not found\n");
        pclose(lspci);
        return 1;
    }
    pclose(lspci);
    
    // Remove newline
    device_addr[strcspn(device_addr, "\n")] = 0;
    
    snprintf(resource_path, sizeof(resource_path), 
             "/sys/bus/pci/devices/0000:%s/resource0", device_addr);
    
    printf("Device address: %s\n", device_addr);
    printf("Resource path: %s\n", resource_path);
    printf("\n");
    
    int fd = open(resource_path, O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("❌ Failed to open resource0 (need root?)");
        printf("\nTry running with: sudo %s\n", __FILE__);
        return 1;
    }
    
    volatile uint32_t *mmio = mmap(NULL, 4096, PROT_READ | PROT_WRITE, 
                                    MAP_SHARED, fd, 0);
    if (mmio == MAP_FAILED) {
        perror("❌ mmap failed");
        close(fd);
        return 1;
    }
    
    printf("✅ Successfully mapped MMIO region\n\n");
    
    // Read registers
    uint32_t cmd_reg = mmio[0x000/4];
    uint32_t status_reg = mmio[0x004/4];
    uint32_t pool_id_raw = mmio[0x008/4];
    uint32_t priority_raw = mmio[0x00C/4];
    uint32_t vm_id = mmio[0x010/4];
    
    char pool_id_char = (char)(pool_id_raw & 0xFF);
    
    printf("Register Values:\n");
    printf("  0x000 (Command):  0x%08x\n", cmd_reg);
    printf("  0x004 (Status):   0x%08x\n", status_reg);
    printf("  0x008 (Pool ID):  0x%08x = '%c'\n", pool_id_raw, pool_id_char);
    printf("  0x00C (Priority): 0x%08x = ", priority_raw);
    switch(priority_raw) {
        case 0: printf("low\n"); break;
        case 1: printf("medium\n"); break;
        case 2: printf("high\n"); break;
        default: printf("unknown\n"); break;
    }
    printf("  0x010 (VM ID):    0x%08x = %u\n", vm_id, vm_id);
    
    printf("\n");
    printf("Interpretation:\n");
    printf("  Pool ID:  '%c'\n", pool_id_char);
    printf("  Priority: %s\n", priority_raw == 2 ? "high" : 
                               priority_raw == 1 ? "medium" : "low");
    printf("  VM ID:    %u\n", vm_id);
    
    printf("\n✅ Test completed successfully!\n");
    
    munmap((void*)mmio, 4096);
    close(fd);
    return 0;
}
TESTPROG

# Try to compile
if command -v gcc &> /dev/null; then
    echo "Compiling test program..."
    if gcc /tmp/test_vgpu_stub.c -o /tmp/test_vgpu_stub 2>/dev/null; then
        echo "✅ Test program compiled successfully"
        echo ""
        echo "[Step 6] Running MMIO register test..."
        echo ""
        
        if [ "$EUID" -eq 0 ]; then
            /tmp/test_vgpu_stub
        else
            echo "Running with sudo (MMIO access requires root)..."
            sudo /tmp/test_vgpu_stub
        fi
    else
        echo "⚠️  Compilation failed. Install gcc: sudo apt-get install build-essential"
        echo "Test program saved to: /tmp/test_vgpu_stub.c"
    fi
else
    echo "⚠️  gcc not found. Install with: sudo apt-get install build-essential"
    echo "Test program saved to: /tmp/test_vgpu_stub.c"
    echo ""
    echo "To compile and run later:"
    echo "  gcc /tmp/test_vgpu_stub.c -o /tmp/test_vgpu_stub"
    echo "  sudo /tmp/test_vgpu_stub"
fi

echo ""
echo "=========================================="
echo "Verification Summary"
echo "=========================================="
echo ""
echo "Device Status:"
if [ -n "$DEVICE_LINE" ]; then
    echo "  [✓] Device detected by lspci"
    echo "  [✓] Device address: $DEVICE_ADDR"
    
    if [ -n "$MMIO_REGION" ]; then
        echo "  [✓] MMIO region allocated"
    else
        echo "  [?] MMIO region status unknown"
    fi
    
    if [ -d "$SYSFS_PATH" ]; then
        echo "  [✓] Sysfs entries present"
    else
        echo "  [?] Sysfs entries status unknown"
    fi
else
    echo "  [✗] Device NOT detected"
fi

echo ""
echo "Files created:"
echo "  - /tmp/test_vgpu_stub.c (source code)"
if [ -f /tmp/test_vgpu_stub ]; then
    echo "  - /tmp/test_vgpu_stub (compiled binary)"
fi

echo ""
echo "=========================================="
echo ""
