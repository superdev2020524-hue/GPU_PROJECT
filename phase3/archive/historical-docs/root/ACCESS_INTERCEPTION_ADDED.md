# access() Interception Added

## What Was Done

Added early interception to `access()` function for `/proc/driver/nvidia/version` to ensure Ollama's prerequisite checks pass.

## Current Status

✅ **access() interception added** - Now intercepts `/proc/driver/nvidia/version` before process check
✅ **open()/openat() interception** - Already intercepts `/proc/driver/nvidia/version`
✅ **stat() interception** - Already intercepts `/proc/driver/nvidia/version`
❌ **Still no NVML function calls** - Ollama still not calling device count functions
❌ **GPU mode still CPU** - `library=cpu`

## Key Finding

**Ollama is NOT checking `/proc/driver/nvidia/version` at all!**

Evidence:
- No interception logs for `/proc/driver/nvidia/version` in any form (open, openat, stat, access)
- Ollama only accesses PCI device files (`/sys/bus/pci/devices/0000:00:05.0/*`)
- This means Ollama's discovery doesn't use `/proc/driver/nvidia/version` as a prerequisite

## What This Means

Ollama's discovery mechanism is different than expected. It:
1. Scans PCI devices directly (✓ working)
2. But doesn't check `/proc/driver/nvidia/version` first
3. And doesn't call NVML/CUDA device query functions

## Next Steps

1. **Check if discovery happens in subprocess** - Maybe runner subprocess doesn't have LD_PRELOAD
2. **Check if Ollama uses different discovery mechanism** - Maybe not using standard NVML/CUDA API
3. **Check Ollama source code** - Understand exactly how discovery works
4. **Check if there's a different prerequisite** - Maybe checks something else we're not intercepting
