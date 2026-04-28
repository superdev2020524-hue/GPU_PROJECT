# Session continuity (latest) — do not lose this thread

*Updated: 2026-03-26 — **full resume snapshot:** **`PHASE3_RESUME_SESSION_2026-03-26.md`** (orchestration, preflight, long-run outcome, E1/E3/E4 pointers).*

*Earlier edit: 2026-03-25 — complements **`ERROR_TRACKING_STATUS.md`**.

## Operator context

- The operator is **not** a domain expert; notes from memory of assistant reports are **valid input** and were **cross-checked** against the VM (cuBLAS paths, Mar 2026).
- **Lt from host:** **`libcublasLt.so.12`** on the guest resolves to **`libcublasLt.so.12.3.2.9`** (large vendor file on disk) — consistent with **dom0 → VM** copy / align (same generation as **`libcublas.so.12.3.2.9`**).
- **Debugging preference (operator):** When we **suspect** a specific error mechanism, **dig into that first** until we **confirm or falsify** it — **then** hunt the “next” error. Avoid parallel hypotheses before the first is settled. (Aligns with **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`** — name **E1/E2/E3** with **log proof**, one layer at a time.)

## Facts to carry forward

1. **Long run (4h client / 4h server+transport):** Completed with **HTTP 500** ~**1h23m**; failure in **GGML CUDA** at **`cublasGemmBatchedEx`**: **“architectural feature absent from the device”** (Tensor Op / compute-type class). Not the old **`cublasCreate_v2`**-only story.
2. **`cublasCreate_v2` fix (Mar 23+):** Route **`libcublas.so.12` → `libvgpu-cublas`**, vendor **`.12.3.2.9`** for **`dlopen`** — **`INVESTIGATION_CUBLASCREATE_V2.md`**, **`CUBLAS_VENDOR_SYMLINK_DEPLOY.md`**.
3. **Live VM (test-4) layout:** **`cuda_v12/libcublas.so.12` → `/opt/vgpu/lib/libvgpu-cublas.so.12`**; **vendor files** **`libcublas.so.12.3.2.9`**, **`libcublasLt.so.12.3.2.9`** present (host copy). **Parent** **`/usr/local/lib/ollama/libcublas.so.12` → `12.8.4.1`** — **skew risk** vs **`cuda_v12`**.
4. **Logs:** **`LONG_RUN_4H_LOG_PATHS.md`**; host **`/tmp/mediator.log`**; VM **`journalctl -u ollama`**, **`/tmp/ollama_journal_longrun.log`**, **`/tmp/longrun_generate_4h.json`**.

## Next engineering (ordered, permissions per **`ASSISTANT_PERMISSIONS.md`**)

1. **VM:** Align **`/usr/local/lib/ollama/libcublas.so.12`** with **`cuda_v12`** policy (or document why 12.8.4.1 must exist).
2. **Host:** start with cheap log correlation (`grep -E 'cublas|Gemm|CUBLAS'` **`/tmp/mediator.log`** around failure time); **edit/build/restart on dom0** when needed per **`ASSISTANT_PERMISSIONS.md`** (not limited to read-only).
3. **Code:** Implement **RPC** for **`cublasGemmBatchedEx`** (and verify **`cublasGemmStridedBatchedEx`**) — see **confirmed gap** below.

---

## Confirmed (2026-03-25): why `cublasGemmBatchedEx` failed

**Source:** **`guest-shim/libvgpu_cublas.c`**

| API | Remote handle → host RPC? |
|-----|---------------------------|
| **`cublasGemmEx`** | **Yes** — `CUDA_CALL_CUBLAS_GEMM_EX` via `cuda_transport_call` |
| **`cublasGemmStridedBatchedEx`** | **No** — only **`RESOLVE_OR_FALLBACK`** → **real** `libcublas` on guest |
| **`cublasGemmBatchedEx`** | **No** — only **`RESOLVE_OR_FALLBACK`** → **real** `libcublas` on guest |

GGML’s failing stack uses **`cublasGemmBatchedEx`** with **Tensor Op** — that path **never** goes to the **mediator**; it runs **NVIDIA cuBLAS in the guest** on top of **shimmed `libcuda`**, which can produce **“architectural feature absent from the device”**.

**Host:** **`cuda_executor.c`** has **`CUDA_CALL_CUBLAS_GEMM_EX`** / **`SGEMM`**; **no** `GEMM_STRIDED_BATCHED_EX` / batched handler in repo grep — so **implement batched GEMM RPC + executor** (or force GGML to a path that uses **`cublasGemmEx`**) is the engineering fix, not more timeout tuning.

## Registry reminder

- **E1** — fatbin **401312** / **INVALID_IMAGE** (separate from this GEMM error).
- **E4** — **`rc=700`** **`CUDA_ERROR_ILLEGAL_ADDRESS`** after **`cublasGemmBatchedEx`** / **`cuCtxSynchronize`** on host (see **`ERROR_TRACKING_STATUS.md`** **2026-03-26**); primary when **Checkpoint C** is clean. Tracked in **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`** and **`ASSISTANT_ROLE_AND_ANTICOUPLING.md` §5**.
- **E3** — runner **exit 2** / guest native — still appears with **500** path above.
