#!/bin/bash
# Test script for vGPU CUDA detection
# This script compiles and runs a test program that detects vGPU

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_PROGRAM="${SCRIPT_DIR}/test_cuda_detection"
TEST_SOURCE="${SCRIPT_DIR}/test_cuda_detection.c"

echo "============================================================"
echo "vGPU CUDA Detection Test"
echo "============================================================"
echo ""

# Compile test program
echo "[1] Compiling test program..."
gcc -o "${TEST_PROGRAM}" "${TEST_SOURCE}" -ldl
if [ $? -ne 0 ]; then
    echo "ERROR: Compilation failed"
    exit 1
fi
echo "  ✓ Compilation successful"
echo ""

# Run test (without LD_PRELOAD - should work via system library)
echo "[2] Running CUDA detection test..."
echo "  (Running without LD_PRELOAD - using system library resolution)"
echo ""

unset LD_PRELOAD
"${TEST_PROGRAM}"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================================"
    echo "✓✓✓ TEST PASSED: vGPU detected successfully!"
    echo "============================================================"
else
    echo "============================================================"
    echo "✗ TEST FAILED: vGPU not detected"
    echo "============================================================"
fi

exit $EXIT_CODE
