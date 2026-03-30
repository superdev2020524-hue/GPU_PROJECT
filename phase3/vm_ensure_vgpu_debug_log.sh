#!/bin/sh
# Ensure /tmp/vgpu_next_call.log is writable by the Ollama user (e.g. test-4) for shim debug.
# Run once on the VM with sudo: sudo sh vm_ensure_vgpu_debug_log.sh
set -e
rm -f /tmp/vgpu_next_call.log
touch /tmp/vgpu_next_call.log
chmod 666 /tmp/vgpu_next_call.log
