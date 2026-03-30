# Host GPU memory during VM model load

When the VM sends model data via HtoD (cuMemcpyHtoD_v2), the **host** mediator forwards the call to **cuda_executor**, which performs a **real** `cuMemcpyHtoD(host_dst, data, copy_len)` to the **physical GPU** (`cuda_executor.c` case `CUDA_CALL_MEMCPY_HTOD`). So if the path is working, host GPU memory usage should increase.

**If nvidia-smi on the host does not show increased GPU memory:**

1. **Confirm the host is applying HtoD**
   - On the **host**, check mediator/executor stderr (or log file) for:
     - `[cuda-executor] HtoD progress: N MB total (vm=Y)` — logged every **10 MB** of HtoD (no env needed).
     - With `VGPU_EXECUTOR_DEBUG=1`: `[cuda-executor] cuMemcpyHtoD: dst=...` and `cuMemcpyHtoD SUCCESS`.
   - If these lines never appear while the VM is loading, the mediator is not receiving this VM’s RPCs, or the executor is not being invoked for this vm_id.

2. **Which process holds the GPU context**
   - nvidia-smi shows memory per **process**. The process that allocated the GPU memory is the one that runs the executor (usually the **mediator** or the binary that links `cuda_executor`).
   - Use:
     - `nvidia-smi pmon` (per-process monitoring), or
     - `nvidia-smi --query-compute-apps=pid,used_memory --format=csv`
   - to see which PID is using GPU memory and whether that usage grows during the VM load.

3. **VM ID and stub connection**
   - The stub passes `vm_id` (from device model args, e.g. `vm_id=9` for Test-4). The mediator must dispatch requests from this VM to the executor with the same vm_id so that `vm_find_mem()` finds the host allocations created by this VM’s earlier cuMemAlloc.

4. **cuMemAlloc on host**
   - If host `cuMemAlloc` failed for this VM, `vm_find_mem()` returns 0 and the executor falls back to `host_dst = (CUdeviceptr)dst` (guest pointer), which is invalid on the host and can cause HtoD to fail or not touch real GPU memory. Check for `[cuda-executor] cuMemAlloc FAILED` in host logs.

**Reference:** `phase3/src/cuda_executor.c` — `CUDA_CALL_MEMCPY_HTOD` (lines ~985–1023), `vm_find_mem`, `htod_total_bytes`, `HTOD_PROGRESS_LOG_INTERVAL` (10 MB).
