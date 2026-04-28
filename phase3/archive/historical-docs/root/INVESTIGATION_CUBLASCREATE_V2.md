# Investigation: `cublasCreate_v2` failure (Mar 2026)

**Symptom (from `FATBIN_TRACE_RECORD.md` / journal):**  
After long model load (`model load progress` ~**0.92**), the llama runner dies with:

- `CUDA error: an internal operation failed`
- `cublasCreate_v2(&cublas_handles[device])` in **ggml-cuda** (`common.cuh` / `ggml-cuda.cu`)

**HTTP:** `POST /api/generate` → **500** after ~**1h25m**.

---

## 1. Which code path is failing? (critical)

There are **two** different `cublasCreate_v2` paths in this stack:

| Path | When it runs | Guest diagnostics |
|------|----------------|---------------------|
| **A — Real NVIDIA `libcublas`** | `libggml-cuda.so` has `DT_NEEDED` on `libcublas.so.12`; loader resolves it from `LD_LIBRARY_PATH` (typically **`cuda_v12`**). | **No** `[libvgpu-cublas]` lines. NVIDIA cuBLAS uses **`cuCtxGetCurrent()`** from the loaded **`libcuda`**. |
| **B — `libvgpu-cublas` shim** | Only if `libcublas.so.12` resolves to **`/opt/vgpu/lib/libvgpu-cublas.so.12`** (or similar). | **`[libvgpu-cublas]`**, `/tmp/vgpu_cublas_*`, RPC to host `CUDA_CALL_CUBLAS_CREATE`. |

### What we verified on the VM (`connect_vm.py`)

1. **No cuBLAS shim in `/opt/vgpu/lib`**  
   `ls /opt/vgpu/lib/libcublas*` → **no such file** (matches **`GPU_MODE_DO_NOT_BREAK.md`** — shims must **not** shadow real cuBLAS).

2. **`libggml-cuda.so` + runner `LD_LIBRARY_PATH`** resolves:
   - `libcublas.so.12` → `/usr/local/lib/ollama/cuda_v12/libcublas.so.12` → **12.3.2.9** (real NVIDIA).
   - `libcublasLt.so.12` → **12.3.2.9**.
   - `libcudart.so.12` → `/opt/vgpu/lib/libcudart.so.12` (shim).
   - `libcuda.so.1` → `/opt/vgpu/lib/libcuda.so.1` (shim).

**Conclusion:** The failing call is **path A** — **NVIDIA `cublasCreate_v2` in the guest**, not the `libvgpu_cublas.c` RPC wrapper.  
Therefore **`INVESTIGATION_CUBLASCREATE_V2.md`** and host `CUDA_CALL_CUBLAS_CREATE` logs are **orthogonal** unless something forces the shim (e.g. wrong `LD_LIBRARY_PATH`).

### Symlink hygiene (risk)

- **`/usr/local/lib/ollama/libcublas.so.12`** may point at a **different** minor (e.g. **12.8.4.1**) than **`cuda_v12/libcublas.so.12`** → **12.3.2.9**.
- With correct wrapper order (`cuda_v12` **before** `/usr/local/lib/ollama`), **dynamic linking** of `libggml-cuda` still picks **`cuda_v12`**.  
- Anything that **`dlopen`s** by path under **`/usr/local/lib/ollama/`** only could load the **wrong** cuBLAS — worth aligning symlinks for safety.

---

## 2. Why GGML reports “internal operation failed”

GGML maps **cuBLAS** errors (and sometimes **CUDA** driver errors) into a **CUDA** error string. A failed **`cublasCreate_v2`** often surfaces as a **generic** CUDA failure after **`cudaGetLastError()`**-style checks.

**Likely classes of root cause (path A):**

1. **Invalid / missing current context** when NVIDIA `libcublas` initializes (e.g. `cuCtxGetCurrent` **NULL**, or context not compatible with cuBLAS expectations after long async work).
2. **Driver / context state** after **very large** HtoD + module activity (resource limits, corrupted state — **hypothesis**, needs host/guest correlation).
3. **ABI / version skew** between **`libcudart`** (shim) and **`libcublas`** (real 12.3.x) — `ldd` may show *“no version information available”* for `libcudart`; worth a controlled experiment if (1–2) are ruled out.

---

## 3. What the guest shim logs **did not** show

After the failed run, on the VM:

- **`/tmp/vgpu_debug.txt`**, **`/tmp/vgpu_cublas_context_diag.txt`**, **`/tmp/vgpu_cublas_init_diag.txt`**, **`/tmp/vgpu_status`** → **missing** or empty for cublas-specific writes.

That is **consistent** with path **A**: **`libvgpu_cublas.c`** is not the library answering `cublasCreate_v2` for **libggml-cuda**.

**`/tmp/vgpu_next_call.log`** existed with transport-oriented lines (`htod`, `stream_sync`, …) — useful for **transport**, not for NVIDIA cuBLAS create.

---

## 4. Host executor (`cuda_executor.c`) — when it matters

On the **host**, `CUDA_CALL_CUBLAS_CREATE` runs **real** `cublasCreate_v2` in the **mediator** process against the **physical** GPU (`cuCtxSetCurrent(exec->primary_ctx)`).

That path is used when the **guest** uses **RPC** (shim path **B**). For path **A**, the **host** may still show **no** `cublasCreate_v2` lines for the **guest** failure — check **`/tmp/mediator.log`** for:

```text
[cuda-executor] vm_id=... cublasCreate_v2 rc=...
```

- **If absent** during the failure window → confirms guest-side NVIDIA cuBLAS is failing, not host RPC create.

---

## 5. Next steps (ordered)

### 5.1 Minimal guest repro (no Ollama, ~seconds) — **RESULT: FAILS**

Use the repo’s **`guest-shim/test_cublas_vm.c`** (VM-only SGEMM smoke test):

```bash
# On a machine that has the VM source tree, copy to VM, then on VM:
cd /path/to
gcc -O2 -Wall -o /tmp/test_cublas_vm test_cublas_vm.c -ldl
LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama \
  /tmp/test_cublas_vm
```

**Executed (assistant, via `connect_vm.py`):** build + run with the `LD_LIBRARY_PATH` above.

**Observed (initial test, libcuda → libcublas only):**

```text
  cublasCreate_v2 returned: 14 (INTERNAL_ERROR)
  handle: (nil)
RC=1
```

**Follow-up A — preload vGPU `libcudart.so.12` before `libcuda` / `libcublas` (matches Ollama wrapper order):**  
Still **`INTERNAL_ERROR` (14)**.

**Follow-up B — after `cuCtxSetCurrent`, call `cudaSetDevice(0)` + `cudaGetLastError()` from shim:**  
`cudaSetDevice(0)` → **0**, `cudaGetLastError` → **0**, but **`cublasCreate_v2`** → **still 14**.

So simple **load order** and **runtime device selection** are **not** sufficient; the problem is deeper (how **NVIDIA libcublas** interacts with the **shimmed** driver/runtime stack, or a limitation of that stack for cuBLAS init).

**Interpretation (important):**

- The failure is **not** specific to “after ~1h25m of tensor upload.” A **minimal** process that **`dlopen`s** `libcuda.so.1` (shim) + **`libcublas.so.12`** (real NVIDIA **12.3.2.9**), then **`cuDevicePrimaryCtxRetain` + `cuCtxSetCurrent`**, still gets **`CUBLAS_STATUS_INTERNAL_ERROR` (14)** from **`cublasCreate_v2`**.
- So the **Ollama** failure is **the same class of bug**: **guest NVIDIA cuBLAS cannot create a handle** on top of the **vGPU `libcuda` / primary context** as currently implemented.
- **Next engineering direction:** treat this as **cuBLAS ↔ shim/driver context contract** (what real `libcublas` expects from `cuCtxGetCurrent` / primary context / device), not as “fatbin / HtoD size.” Host **`CUDA_CALL_CUBLAS_CREATE`** (mediator) is a **different** path (real host driver); it may succeed even when **guest** `cublasCreate_v2` fails — **do not** assume host logs will show guest create.

From repo root (if `connect_vm` + `scp` available):

```bash
cd phase3
scp guest-shim/test_cublas_vm.c test-4@10.25.33.12:/tmp/
CONNECT_VM_COMMAND_TIMEOUT_SEC=120 python3 connect_vm.py \
  'gcc -O2 -o /tmp/test_cublas_vm /tmp/test_cublas_vm.c -ldl && LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama /tmp/test_cublas_vm'
```

### 5.2 Align “parent” `libcublas` symlink (low risk)

Ensure **`/usr/local/lib/ollama/libcublas.so.12`** does not point at a **different** minor than the **`cuda_v12`** pair you intend (e.g. both **12.3.2.9**):

```bash
readlink -f /usr/local/lib/ollama/libcublas.so.12
readlink -f /usr/local/lib/ollama/cuda_v12/libcublas.so.12
```

If they differ, fix the **parent** symlink to match **`cuda_v12`** (see **`CUBLAS_HOST_TO_VM_SCP.md`**).

### 5.3 Short Ollama run after **clean** restart

Restart **mediator** + **`ollama`**, then **`curl`** with **short** timeout / small model — does **`cublasCreate_v2`** fail **immediately** or only **after** long HtoD?

- **Immediate** → environment / library / context setup.
- **Only after long load** → state / resource / ordering bug.

### 5.4 Host log slice

On **mediator host**, around the journal timestamp of the failure:

```bash
grep -E "cublasCreate_v2|CUBLAS|CUDA_CALL_CUBLAS" /tmp/mediator.log
```

### 5.5 Optional: `CUBLAS_DEBUG` / `VGPU_DEBUG`

Only helps if **`libvgpu-cublas`** is actually loaded (path **B**). For path **A**, use **`test_cublas_vm`** and NVIDIA tools instead.

---

## 6. Summary

| Question | Answer |
|----------|--------|
| Is the **1h25m** run “for `ldd`”? | **No** — it was **real load + transport**; `ldd` is seconds. |
| Is **`cublasCreate_v2`** important? | **Yes** — it pinpoints **guest-side NVIDIA cuBLAS** vs **transport/fatbin** issues. |
| Is **`libvgpu_cublas.c`** on the hot path? | **Not** for current **`ldd`** of **`libggml-cuda.so`** — **real** `libcublas` from **`cuda_v12`**. |
| Does **`test_cublas_vm`** reproduce the issue? | **Yes** — **`cublasCreate_v2` → 14 (`INTERNAL_ERROR`)** in **seconds** with shim + real cuBLAS. |
| Next bottleneck | **Fix or route guest cuBLAS** so **`cublasCreate_v2`** sees a **valid** context for NVIDIA’s `libcublas` (shim/driver contract), or **force RPC path** (`libvgpu-cublas`) for **all** GGML cuBLAS usage — **large** design change. |

---

---

## 7. Resolution (Mar 2026)

**Cause:** Guest **NVIDIA `libcublas`** cannot initialize on the **shimmed** CUDA stack (`cublasCreate_v2` → **14**).

**Fix:** Ensure dynamic loading uses **`libvgpu-cublas.so.12`** as **`libcublas.so.12`** (RPC to host for create/GEMM). Verified with **`LD_LIBRARY_PATH`** prefix directory + symlink to shim — **`test_cublas_vm`** **full SUCCESS**.

**Code:** `libvgpu_cublas.c` **`init_real_cublas()`** now tries **`libcublas.so.12.3.2.9`** **before** **`libcublas.so.12`** so fallback **`dlsym`** loads the **vendor** library when the public name is the shim.

**Deploy:** **`CUBLAS_VENDOR_SYMLINK_DEPLOY.md`**.

---

*Last updated: root cause + fix path (assistant).*
