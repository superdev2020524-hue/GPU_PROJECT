# GPU Operations Verification

This document describes how to verify that actual GPU compute operations are being forwarded to the physical H100 GPU.

## Overview

The system architecture:
```
VM (Guest)                    Host
─────────────────────────────────────────────
Ollama
  ↓
libvgpu-cuda.so (shim)
  ↓ (intercepts CUDA calls)
cuda_transport_call()
  ↓ (MMIO to vGPU-stub)
vGPU-stub PCI device
  ↓ (Unix socket)
Mediator Daemon
  ↓
CUDA Executor
  ↓ (real CUDA Driver API)
Physical H100 GPU
```

## Verification Steps

### 1. Check Mediator is Running

On the **host** (10.25.33.10):
```bash
ps aux | grep mediator_phase3 | grep -v grep
```

Expected output:
```
root     12345  0.1  0.2  ...  ./mediator_phase3
```

### 2. Run Compute-Intensive Model Query

On the **VM** (10.25.33.111):
```bash
ollama run llama3.2:1b 'Calculate 123*456 and show your work step by step'
```

This should:
- Trigger actual GPU compute operations
- Launch CUDA kernels for matrix operations
- Transfer data to/from GPU memory
- Return correct results

### 3. Check Mediator Logs for GPU Operations

On the **host**, check mediator logs:
```bash
tail -100 /tmp/mediator.log | grep -E 'cuLaunchKernel|cuMemcpy|cuMemAlloc'
```

**Expected logs:**
```
[cuda-executor] cuMemAlloc: allocating 545947648 bytes on physical GPU (vm=11)
[cuda-executor] cuMemAlloc SUCCESS: allocated 0x7f8a00000000 on physical GPU (vm=11)
[cuda-executor] cuMemcpyHtoD: dst=0x7f8a00000000 size=545947648 bytes (vm=11)
[cuda-executor] cuMemcpyHtoD SUCCESS: data copied to physical GPU (vm=11)
[cuda-executor] cuLaunchKernel: grid=(256,1,1) block=(256,1,1) shared=0 params=4 vm=11
[cuda-executor] cuLaunchKernel SUCCESS: kernel executed on physical GPU (vm=11)
[cuda-executor] cuMemcpyDtoH: src=0x7f8a00000000 size=545947648 bytes (vm=11)
[cuda-executor] cuMemcpyDtoH SUCCESS: data copied from physical GPU (vm=11)
```

### 4. Check VM Logs for Transport Calls

On the **VM**, check Ollama service logs:
```bash
journalctl -u ollama.service --since '5 minutes ago' | grep -E 'cuda_transport_call|CUDA_CALL'
```

**Expected evidence:**
- `cuda_transport_call()` being invoked
- CUDA_CALL_LAUNCH_KERNEL operations
- CUDA_CALL_MEMCPY_HTOD operations
- CUDA_CALL_MEMCPY_DTOH operations

## What Each Operation Means

### cuMemAlloc
- **What:** Allocates memory on the physical GPU
- **Evidence:** Log shows allocation size and GPU pointer
- **Success:** Memory is actually allocated on H100

### cuMemcpyHtoD (Host-to-Device)
- **What:** Copies data from VM host memory to physical GPU memory
- **Evidence:** Log shows source/destination and size
- **Success:** Data is physically transferred to H100

### cuLaunchKernel
- **What:** Launches a CUDA kernel on the physical GPU
- **Evidence:** Log shows grid/block dimensions and parameters
- **Success:** Kernel actually executes on H100 compute units

### cuMemcpyDtoH (Device-to-Host)
- **What:** Copies results from physical GPU back to VM
- **Evidence:** Log shows data being copied back
- **Success:** Results are physically transferred from H100

## Troubleshooting

### No GPU Operation Logs

**Possible causes:**
1. **Mediator not running** - Check with `ps aux | grep mediator`
2. **Transport not connected** - Check VM logs for connection errors
3. **Operations not forwarded** - Check if `cuda_transport_call()` is being called
4. **Logs in different location** - Check `/var/log/mediator_phase3.log` or stderr

### Operations Logged But Model Fails

**Possible causes:**
1. **Kernel launch errors** - Check for `cuLaunchKernel FAILED` in logs
2. **Memory allocation failures** - Check for `cuMemAlloc FAILED` in logs
3. **Context errors** - Check for CUDA context initialization errors

### Model Works But No GPU Logs

**Possible causes:**
1. **Model using CPU fallback** - Check Ollama logs for "library: cpu"
2. **Shim not intercepting** - Verify LD_PRELOAD is set correctly
3. **Operations cached** - Some operations may be cached locally

## Automated Verification

Run the verification script:
```bash
cd /home/david/Downloads/gpu/phase3
./verify_gpu_operations.sh
```

This script:
1. Checks if mediator is running
2. Runs a compute-intensive query
3. Checks mediator logs for GPU operations
4. Reports verification status

## Success Criteria

✅ **GPU operations are verified if:**
- Mediator logs show `cuMemAlloc SUCCESS`
- Mediator logs show `cuMemcpyHtoD SUCCESS`
- Mediator logs show `cuLaunchKernel SUCCESS`
- Mediator logs show `cuMemcpyDtoH SUCCESS`
- Model produces correct results
- No CPU fallback messages in Ollama logs

## Next Steps

Once GPU operations are verified:
1. Performance benchmarking
2. Multi-VM testing
3. Larger model testing
4. Stress testing
