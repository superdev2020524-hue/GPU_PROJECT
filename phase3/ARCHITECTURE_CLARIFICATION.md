# Architecture Clarification: Software vs Hardware Virtualization

## Date: 2026-02-27

## The Confusion

**You have TWO components:**
1. **PCI VGPU-STUB device** (hardware-level) - appears in `lspci`
2. **CUDA API shims** (software-level) - intercept CUDA calls

**The key insight:** Ollama doesn't use PCI directly - it uses CUDA APIs!

## What We're Actually Doing (Software-Level Virtualization)

### ✅ Current Implementation: Option 2 (Software-Level)

**We ARE doing software-level virtualization** - exactly what ChatGPT recommended and what Momik wants:

1. **Replace libcuda.so.1** with our shim (`libvgpu-cuda.so`)
   - Intercepts CUDA Driver API calls (`cuInit`, `cuDeviceGetCount`, etc.)
   - No kernel driver needed
   - Pure userspace interception

2. **Replace libcudart.so.12** with our shim (`libvgpu-cudart.so`)
   - Intercepts CUDA Runtime API calls (`cudaGetDeviceCount`, `cudaGetDeviceProperties`, etc.)
   - No kernel driver needed
   - Pure userspace interception

3. **How it works:**
   ```
   Ollama → calls cudaGetDeviceCount()
          → Our shim intercepts (via LD_PRELOAD)
          → Returns count=1 (fake GPU)
          → Ollama thinks there's a GPU
   ```

### The PCI Device's Role

**The PCI VGPU-STUB device is NOT for CUDA** - it's for:
- **Discovery**: Finding the device in `/sys/bus/pci/devices/`
- **Transport**: MMIO communication to the mediator
- **NOT for CUDA**: Ollama never directly accesses PCI

**Ollama's path:**
```
Ollama → CUDA Runtime API → Our shim → (eventually) Transport → Mediator → Physical H100
```

**NOT:**
```
Ollama → PCI device → (this doesn't happen)
```

## Hardware vs Software Virtualization

### Hardware-Level Virtualization (Option 1 - NOT what we're doing)
- Make the kernel think there's a real GPU
- Implement NVIDIA kernel driver interface
- Handle BAR0 MMIO, registers, ioctls
- **Extremely complex** - reverse engineering NVIDIA driver
- **We're NOT doing this**

### Software-Level Virtualization (Option 2 - What we ARE doing)
- Replace CUDA libraries with shims
- Intercept CUDA API calls at userspace
- No kernel driver needed
- **Much simpler** - just implement CUDA API functions
- **This is what we're doing** ✅

## Why This Matches Momik's Requirements

**Momik wants:**
> "The layer that you're developing, it's invisible to the instance user"
> "VM with Ollama running a model and it's purely computing through virtual GPU"

**Our approach:**
- ✅ **Invisible**: User just runs Ollama - doesn't know about shims
- ✅ **Software-level**: Intercept CUDA APIs (not PCI)
- ✅ **VM-level**: Works in VM without kernel driver
- ✅ **Matches ChatGPT's Option 2**: Exactly what was recommended

## The Architecture

```
┌─────────────────────────────────────────────────────────┐
│ VM (Guest)                                              │
│                                                         │
│  Ollama                                                 │
│    ↓                                                    │
│  CUDA Runtime API (cudaGetDeviceCount, etc.)           │
│    ↓                                                    │
│  Our shim (libvgpu-cudart.so) ← SOFTWARE-LEVEL        │
│    ↓                                                    │
│  Our shim (libvgpu-cuda.so) ← SOFTWARE-LEVEL          │
│    ↓                                                    │
│  Transport (MMIO via PCI VGPU-STUB) ← HARDWARE-LEVEL   │
│    ↓                                                    │
└─────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────┐
│ Host                                                     │
│                                                         │
│  Mediator                                               │
│    ↓                                                    │
│  Physical H100 GPU                                      │
└─────────────────────────────────────────────────────────┘
```

**Key point:** The software shims are what make Ollama think there's a GPU. The PCI device is just for transport.

## Conclusion

**✅ We ARE doing software-level virtualization (Option 2)**

**✅ This matches what Momik wants:**
- Invisible layer ✅
- VM-level integration ✅
- CUDA API interception ✅
- No kernel driver needed ✅

**✅ This matches what ChatGPT recommended:**
- Option 2 (software-level) ✅
- Replace libcuda.so with shim ✅
- Intercept CUDA API calls ✅

**The PCI device is just for transport/discovery - not for CUDA itself.**
