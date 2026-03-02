#!/bin/bash
# Verify GPU Operations Script
# This script verifies that actual GPU compute operations are being forwarded
# to the physical H100 GPU through the mediation layer.

set -e

echo "=========================================="
echo "GPU Operations Verification Test"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're on the VM or host
if [ -f /sys/bus/pci/devices/0000:00:05.0/vendor ]; then
    echo -e "${YELLOW}Running on VM (guest)${NC}"
    IS_VM=1
    VM_IP="10.25.33.111"
    HOST_IP="10.25.33.10"
else
    echo -e "${YELLOW}Running on host${NC}"
    IS_VM=0
    HOST_IP="localhost"
fi

echo ""
echo "Step 1: Check if mediator is running on host..."
if [ "$IS_VM" = "1" ]; then
    # Check from VM
    ssh -o StrictHostKeyChecking=no root@${HOST_IP} "ps aux | grep -E 'mediator_phase3|mediator' | grep -v grep" || {
        echo -e "${RED}ERROR: Mediator not running on host${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Mediator is running on host${NC}"
else
    # Check on host
    ps aux | grep -E 'mediator_phase3|mediator' | grep -v grep || {
        echo -e "${RED}ERROR: Mediator not running${NC}"
        exit 1
    }
    echo -e "${GREEN}✓ Mediator is running${NC}"
fi

echo ""
echo "Step 2: Run a compute-intensive Ollama query..."
if [ "$IS_VM" = "1" ]; then
    ssh -o StrictHostKeyChecking=no test-11@${VM_IP} "timeout 30 ollama run llama3.2:1b 'Calculate 123*456 and show your work step by step' 2>&1 | tail -10" || true
else
    timeout 30 ollama run llama3.2:1b 'Calculate 123*456 and show your work step by step' 2>&1 | tail -10 || true
fi

echo ""
echo "Step 3: Check mediator logs for GPU operations..."
if [ "$IS_VM" = "1" ]; then
    echo "Checking host mediator logs..."
    ssh -o StrictHostKeyChecking=no root@${HOST_IP} "tail -50 /tmp/mediator.log 2>/dev/null || tail -50 /var/log/mediator_phase3.log 2>/dev/null || echo 'No mediator log found'" | grep -E 'cuLaunchKernel|cuMemcpy|cuMemAlloc|cuda-executor.*SUCCESS|cuda-executor.*FAILED' | tail -20 || {
        echo -e "${YELLOW}No GPU operation logs found. This could mean:${NC}"
        echo "  1. Operations are not being forwarded (check transport)"
        echo "  2. Mediator logs are in a different location"
        echo "  3. Operations are happening but not logged"
    }
else
    tail -50 /tmp/mediator.log 2>/dev/null || tail -50 /var/log/mediator_phase3.log 2>/dev/null | grep -E 'cuLaunchKernel|cuMemcpy|cuMemAlloc|cuda-executor.*SUCCESS|cuda-executor.*FAILED' | tail -20 || {
        echo -e "${YELLOW}No GPU operation logs found${NC}"
    }
fi

echo ""
echo "Step 4: Check VM logs for CUDA transport calls..."
if [ "$IS_VM" = "1" ]; then
    ssh -o StrictHostKeyChecking=no test-11@${VM_IP} "journalctl -u ollama.service --since '2 minutes ago' --no-pager | grep -E 'cuda_transport_call|CUDA_CALL_LAUNCH|CUDA_CALL_MEMCPY' | tail -10" || {
        echo -e "${YELLOW}No transport call logs found in VM${NC}"
    }
else
    echo "Cannot check VM logs from host. Run this script from the VM to see transport logs."
fi

echo ""
echo "=========================================="
echo "Verification Complete"
echo "=========================================="
echo ""
echo "Expected evidence of GPU operations:"
echo "  ✓ cuMemAlloc: Memory allocated on physical GPU"
echo "  ✓ cuMemcpyHtoD: Data copied to physical GPU"
echo "  ✓ cuLaunchKernel: Kernels launched on physical GPU"
echo "  ✓ cuMemcpyDtoH: Results copied back from physical GPU"
echo ""
echo "If you see these logs, GPU operations are working!"
echo "If not, check:"
echo "  1. Mediator is running and connected"
echo "  2. Transport layer is forwarding calls"
echo "  3. CUDA executor is initialized"
