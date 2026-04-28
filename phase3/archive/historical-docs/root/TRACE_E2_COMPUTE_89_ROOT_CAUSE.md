# Trace: Ollama logs `compute=8.9` on H100 vGPU (E2)

*Mar 22, 2026 — systematic trace per `SYSTEMATIC_ERROR_TRACKING_PLAN.md`.*

## Checkpoints (this session)

| Checkpoint | Result |
|------------|--------|
| **A** | `ollama` **active**; `library=CUDA` present |
| **B** | Shims present; **mismatch**: journal **`compute=8.9`** vs repo **`libvgpu_cudart.c`** intending **9.0** |
| **C** | **E1** still present: **`401312` → `INVALID_IMAGE`** in `mediator.log` |

## Trace chain (where `8.9` comes from)

1. **`discover/types.go`** → `slog.Info("inference compute", …, "compute", dev.Compute(), …)`
2. **`ml/device.go`** → `Compute()` = `strconv.Itoa(ComputeMajor) + "." + strconv.Itoa(ComputeMinor)` (non-ROCm)
3. **`ml/backend/ggml/ggml.go`** → `BackendDevices()` fills `ComputeMajor` / `ComputeMinor` from **`C.ggml_backend_dev_get_props`** → GGML **`props.compute_major` / `compute_minor`**
4. **`ggml-cuda.cu`** → `ggml_backend_cuda_device_get_props` uses **`ctx->major` / `ctx->minor`** (non-HIP)
5. **`ggml_backend_cuda_reg()`** (same file ~5120–5130) sets:
   ```cpp
   CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
   …
   dev_ctx->major = prop.major;
   dev_ctx->minor = prop.minor;
   ```

So the logged **`8.9`** is **`prop.major` / `prop.minor`** as seen by **GGML’s** `cudaDeviceProp` layout at **`cudaGetDeviceProperties`**, not necessarily the shim’s intended `computeCapabilityMajor` fields.

## Root cause hypothesis (confirmed direction)

**CUDA `cudaDeviceProp` ABI skew:** `libvgpu-cudart.so` is compiled with **one** toolkit’s `cudaDeviceProp` layout; **libggml-cuda.so** is compiled with **another**. Writing `prop->major` / `prop->computeCapabilityMajor` in the shim uses **the shim’s** offsets; GGML reads **`prop.major`** at **GGML’s** offsets → **wrong values (observed 8.9)** despite `gpu_properties.h` **9.0**.

The **`patch_ggml_cuda_device_prop()`** raw-offset patches help GGML **direct reads** but **do not** fix **`prop.major` as seen by GGML’s compiled field access** if the struct sizes differ.

## Implementation applied on VM (source only)

On **`/home/test-4/ollama/ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu`** (backup `*.bak_cc_YYYYMMDD`):

- Replaced `dev_ctx->major = prop.major; dev_ctx->minor = prop.minor;` with:
  ```cpp
  dev_ctx->major = 9;  /* Phase3 vGPU: shim/toolkit cudaDeviceProp layout skew; H100 = sm_90 */
  dev_ctx->minor = 0;
  ```

**Important:** `go build -o ollama.bin.new` **did not** embed this change into behavior: **`strings ollama.bin.new`** does **not** contain the comment string, and **`journalctl` still shows `compute=8.9`**.  

**Reason:** Discovery uses **`/usr/local/lib/ollama/cuda_v12/libggml-cuda.so`** (~187 MB, prebuilt). That **`.so`** contains **`ggml_backend_cuda_reg`**, not the Go binary. **You must rebuild and reinstall `libggml-cuda.so`** after editing `ggml-cuda.cu`.

## Next step (single)

**Rebuild `libggml-cuda.so`** on a machine with CUDA (VM or Docker per `build_libggml_cuda_hopper_docker.sh` / Ollama’s Linux build), install to **`/usr/local/lib/ollama/cuda_v12/libggml-cuda.so`**, **`systemctl restart ollama`**, re-run **Checkpoint B** (expect **`compute=9.0`**) then **Checkpoint C** (see if **E1**/`401312` changes).

## Related

- `FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md` — why **8.9** steers **sm_80** Lt fatbins  
- `SYSTEMATIC_ERROR_TRACKING_PLAN.md` — registry **E2**  
- `guest-shim/libvgpu_cudart.c` — still correct to advertise 9.0; GGML must read consistent CC via rebuilt **libggml-cuda.so** or further offset research  
