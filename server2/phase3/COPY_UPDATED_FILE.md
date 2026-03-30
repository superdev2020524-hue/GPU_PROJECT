# Copy Updated libvgpu_cuda.c to VM

The file on the VM needs to be updated with the logging code. Since direct file transfer through the connection script is problematic, here are the steps:

## Option 1: Manual Copy via SSH (Recommended)

From your local machine:
```bash
scp phase3/guest-shim/libvgpu_cuda.c test-11@10.25.33.111:~/phase3/guest-shim/libvgpu_cuda.c
```

## Option 2: Apply Changes Manually on VM

The changes needed are:

1. **In `cuLaunchKernel()` function (around line 5001):**
   - Add logging at the start of the function
   - Add success/failure logging after `cuda_transport_call()`

2. **In `cuMemcpyHtoD_v2()` function (around line 4653):**
   - Add logging at the start
   - Add success logging after `cuda_transport_call()`

3. **In `cuMemcpyDtoH_v2()` function (around line 4698):**
   - Add logging at the start
   - Add success logging after `cuda_transport_call()`

See `VM_GPU_OPERATIONS_VERIFICATION.md` for the exact code changes.

## Option 3: Use install.sh

If the install.sh script copies files, you could:
1. Update the local file
2. Run the install script which should copy it to the VM
