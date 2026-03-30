# VM CUDA Activity Summary

## Test Performed

Triggered Ollama request to generate CUDA activity and verify shim interception.

## Results

### Transport Activity

**Commands sent to VGPU-STUB:**
- `call_id=0x0030` (CUDA_CALL_INIT) - Device initialization
- Additional calls may appear depending on Ollama's operations

**Transport logs show:**
- ✅ SENDING to VGPU-STUB
- ✅ RINGING DOORBELL
- ✅ RECEIVED from VGPU-STUB with status=DONE

### CUDA Functions Intercepted

**Driver API:**
- `cuMemCreate()` - Unified memory allocation
- `cuMemAddressReserve()` - Memory address reservation  
- `cuMemMap()` - Memory mapping
- `cuMemSetAccess()` - Memory access control
- `cuMemRelease()` - Memory release

**Runtime API:**
- `cudaGetDevice()` - Device queries
- `cudaGetLastError()` - Error checking
- `cudaStreamBeginCapture()` / `cudaStreamEndCapture()` - Stream capture
- `cudaGraphDestroy()` - Graph cleanup
- `cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags()` - Occupancy queries

**CUBLAS:**
- `cublasSgemm_v2()` - Matrix multiplication
- `cublasSetStream_v2()` - Stream setting

## Next Steps

1. **Check host-side logs** (see HOST_SIDE_VERIFICATION_INSTRUCTIONS.md):
   - Verify VGPU-STUB received doorbell rings
   - Verify mediator received CUDA calls
   - Verify CUDA executor processed them

2. **If host-side is working:**
   - Investigate Ollama crash (likely shim response handling)
   - Verify unified memory API implementations
   - Test with simpler CUDA operations

3. **If host-side not receiving:**
   - Check QEMU/VGPU-STUB logging
   - Verify socket connection
   - Check mediator is listening
