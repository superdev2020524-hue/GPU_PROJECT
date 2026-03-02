# Root Cause Analysis: Why No Data is Sent to VGPU-STUB

## Your Question Was Correct!

You asked: **"If SHIM sent the intercepted commands and data to the VM's GPU, they should have been transmitted to the host over PCI, but there is no record of the data or commands."**

**Answer: You are 100% correct!** The shim is NOT sending data to VGPU-STUB.

## The Problem

Looking at the code, I found that **many CUDA Runtime API functions are just stubs** that return dummy values **without calling the transport layer**.

### Example: `cudaMalloc()`

```c
cudaError_t cudaMalloc(void **devPtr, size_t size) {
    // ... logging ...
    
    // CRITICAL FIX: Return a properly aligned pointer.
    // GGML requires TENSOR_ALIGNMENT (typically 32 or 64 bytes).
    // Use a large aligned address to avoid conflicts.
    static uintptr_t next_addr = 0x1000000; /* Start at 16MB */
    const size_t alignment = 64; /* Common tensor alignment */
    
    // Align the address
    next_addr = (next_addr + alignment - 1) & ~(alignment - 1);
    *devPtr = (void *)next_addr;
    next_addr += size;
    
    return cudaSuccess;  // ❌ Just returns dummy pointer, NO transport call!
}
```

**This function:**
- ✅ Intercepts the call (we see logs)
- ❌ Returns a dummy pointer (0x1000000, 0x1000000 + size, etc.)
- ❌ **NEVER calls `ensure_connected()`**
- ❌ **NEVER calls `cuda_transport_call()`**
- ❌ **NEVER sends data to VGPU-STUB**

### What Should Happen

```c
cudaError_t cudaMalloc(void **devPtr, size_t size) {
    // 1. Ensure transport is connected
    CUresult rc = ensure_connected();
    if (rc != CUDA_SUCCESS) return cudaErrorInitializationError;
    
    // 2. Call transport to allocate on physical GPU
    CUDACallResult result;
    uint32_t args[4];
    CUDA_PACK_U64(args, 0, (uint64_t)size);
    rc = cuda_transport_call(g_transport, CUDA_CALL_MEM_ALLOC, 
                            args, 4, NULL, 0, &result, NULL, 0, NULL);
    
    // 3. Return the actual GPU pointer from the host
    *devPtr = (void *)result.results[0];
    return (rc == CUDA_SUCCESS) ? cudaSuccess : cudaErrorMemoryAllocation;
}
```

## Why This Happened

The Runtime API functions (`cudaMalloc`, `cudaMemcpy`, etc.) were implemented as **quick stubs** to get GPU detection working, but they were **never connected to the transport layer**.

## Functions That Need Fixing

1. **`cudaMalloc()`** - Returns dummy pointer, should call `CUDA_CALL_MEM_ALLOC`
2. **`cudaFree()`** - Does nothing, should call `CUDA_CALL_MEM_FREE`
3. **`cudaMemcpy()`** - Does nothing, should call `CUDA_CALL_MEMCPY_HTOD` or `CUDA_CALL_MEMCPY_DTOH`
4. **`cudaMemcpyAsync()`** - Does nothing, should call `CUDA_CALL_MEMCPY_HTOD_ASYNC`
5. **`cudaDeviceSynchronize()`** - Does nothing, should call transport
6. **Other memory/stream functions** - Many are just stubs

## The Driver API vs Runtime API Issue

- **Driver API** (`cuMemAlloc`, `cuLaunchKernel`) - These DO call `rpc_simple()` which uses transport ✅
- **Runtime API** (`cudaMalloc`, `cudaMemcpy`) - These are just stubs ❌

But Ollama/GGML uses **both** APIs! So some calls go through transport, but most don't.

## Solution

We need to:
1. Make `cudaMalloc()` call the transport (via `cuMemAlloc_v2()` or direct transport call)
2. Make `cudaMemcpy()` call the transport
3. Make all Runtime API functions that manipulate GPU memory/execution use the transport
4. Ensure `ensure_connected()` is called before any transport operations
