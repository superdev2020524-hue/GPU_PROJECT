# cuBLAS: route Ollama/GGML through **libvgpu-cublas** (RPC to host)

## Root cause (Mar 2026)

- **Real** NVIDIA `libcublas.so.12` in the guest calls **`cublasCreate_v2`** against the **vGPU `libcuda` / `libcudart` shims**.
- That combination returns **`CUBLAS_STATUS_INTERNAL_ERROR` (14)** in **`test_cublas_vm`** (seconds, no long load).
- **Same** failure mode as the long Ollama run at **`cublasCreate_v2`** in ggml-cuda.

## Verified fix (VM test)

Prefix **`LD_LIBRARY_PATH`** so **`libcublas.so.12` resolves to **`/opt/vgpu/lib/libvgpu-cublas.so.12`** before **`cuda_v12`**:

```bash
mkdir -p /tmp/cublas_shim_first
ln -sf /opt/vgpu/lib/libvgpu-cublas.so.12 /tmp/cublas_shim_first/libcublas.so.12
LD_LIBRARY_PATH=/tmp/cublas_shim_first:/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama \
  ./test_cublas_vm
```

**Result:** **`cublasCreate_v2` → SUCCESS**, SGEMM **SUCCESS** (RPC / host path).

So GGML must use **`libvgpu-cublas`**, not the vendor **`libcublas.so.12`** symlink alone.

---

## Permanent guest layout

1. Keep the **vendor** math library on disk with a **versioned** name, e.g.  
   **`/usr/local/lib/ollama/cuda_v12/libcublas.so.12.3.2.9`** (real NVIDIA file).

2. Make **`libcublas.so.12`** a symlink to the shim:  
   **`libcublas.so.12` → `/opt/vgpu/lib/libvgpu-cublas.so.12`**

3. **`libvgpu_cublas.c`** `init_real_cublas()` must **`dlopen` the versioned path first** (implemented in repo) so the fallback path loads **NVIDIA** cuBLAS, not the shim again.

---

## Deploy order (important)

1. **Build and install** updated **`libvgpu-cublas.so.12`** (with versioned **`dlopen`** path) to **`/opt/vgpu/lib/`**.
2. **Then** repoint **`/usr/local/lib/ollama/cuda_v12/libcublas.so.12`** → shim (or use **`LD_LIBRARY_PATH`** prefix in **`ollama` service** until step 1 is on the VM).

If you flip the symlink **before** installing a shim that prefers **`libcublas.so.12.3.2.9`**, old shim code could **`dlopen`** **`libcublas.so.12`** and load **itself**.

---

## Relation to **`GPU_MODE_DO_NOT_BREAK.md`**

That doc warns against a **broken** cuBLAS **stub** in **`/opt/vgpu/lib`** that fakes discovery. **`libvgpu-cublas`** here is the **RPC** implementation: **`cublasCreate`** and GEMM can be **forwarded** to the host mediator. Discovery still uses **`libcuda`** / **`libcudart`** shims as before.

Re-check **`/api/tags`** and **`inference compute`** after deploy.

---

## Mediator reset

Not required to **prove** the fix (short **`test_cublas_vm`**). Recommended before **long** regression runs for clean logs.
