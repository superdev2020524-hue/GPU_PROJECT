# Work note (Mar 19): Second `cuModuleLoadFatBinary` → `CUDA_ERROR_INVALID_IMAGE`

## Summary

During a live GPU load (tinyllama), **host→device copies (0x0032)** completed successfully at scale. The hard failure was on **`CUDA_CALL_MODULE_LOAD_FAT_BINARY` (0x0042)** for a **second**, larger fat binary.

## Correlated evidence

| Layer | Observation |
|--------|----------------|
| **Host** `/tmp/mediator.log` | `module-load start … data_len=**28120**` → **`rc=0` SUCCESS** |
| **Host** | Next: `module-load start … data_len=**401312**` → **`rc=200` `CUDA_ERROR_INVALID_IMAGE`** — *device kernel image is invalid*, `module=(nil)` |
| **Host** | `CUDA result sent vm_id=9 **request_id=826** call_id=0x42 result.status=**200**` |
| **VM** `journalctl` | `[cuda-transport] **STATUS_ERROR**: call_id=**0x0042** seq=**826**` |
| **VM** | `MODULE_LOAD chunk failed at offset=0 chunk=401312 total=401312 rc=2` |

Protocol: **`0x0042` = `CUDA_CALL_MODULE_LOAD_FAT_BINARY`** (`include/cuda_protocol.h`).

## VM library check (same day, via `connect_vm.py`)

- **`/usr/local/lib/ollama/cuda_v12/libggml-cuda.so`** (~187 MB, dated Mar 15 on VM).
- **`strings`** shows many **`.target sm_90`** entries → the **on-disk** GGML CUDA library **does** claim Hopper SASS.
- **sha256:** `73e478b717095efe4a02382b9d6430b808e0d4bc347aab2256bf2fa4732babd9`

## Implication for root-cause

The naive story “bundle has no sm_90” is **weakened** for this VM: the installed `.so` contains sm_90 targets. Remaining hypotheses include:

1. **Chunked module transport / reassembly** — wrong or truncated bytes reaching `cuModuleLoadFatBinary` on the host (size matches **401312** in log, but content could still be wrong).
2. **Fat binary / CUDA version skew** — guest fatbin format vs host driver expectations.
3. **Multiple embedded images** — one small module loads; a **different** embedded image in the stream fails validation on H100.
4. **Stale or alternate blob** — runner loads a different code path than the file we hashed (less likely if sizes align).

## Permissions (assistant)

- **VM:** May deploy a replacement `libggml-cuda.so`, restart Ollama (`deploy_libggml_cuda_hopper.py`), add guest-side diagnostics.
- **Host:** Read logs only; executor/mediator changes are **user-operated**.

## Suggested next steps

1. **Guest (deployed):** **`cuda_transport.c`** — 64 KiB cap for large **FAT_BINARY**; chunk flags use **`chunk == send_len`** (not **`send_len <= limit`**) so flags match “this RPC carries the whole payload.” **`[vgpu-fp]`** on **stderr** + optional **`/var/tmp/...`**. Deploy: **`transfer_cuda_transport.py`**.
2. **Host correlation (read-only):** For **`data_len=401312`**, mediator logs **`module-chunk ... single=1 data_len=401312 first8=50ed55ba01001000`** then **`module-load done ... INVALID_IMAGE`**. So the host gets **one** full buffer in **one** RPC — **`INVALID_IMAGE`** is **CUDA** rejecting the **fatbin contents** for the **H100**, not a missing chunk.
3. **Next action — build machine with CUDA (not dom0 mediator):** Build **`libggml-cuda.so`** with **`CMAKE_CUDA_ARCHITECTURES=90`** (see **`BUILD_LIBGGML_CUDA_HOPPER.md`**), then run **`python3 deploy_libggml_cuda_hopper.py /path/to/libggml-cuda.so`** from a host that can reach the VM.

## Related

- `ERROR_TRACKING_STATUS.md` §10  
- `BUILD_LIBGGML_CUDA_HOPPER.md`  
- `GPU_MODE_DO_NOT_BREAK.md` (Hopper `.so` requirement)
