# E1 — next process checklist (after tracing)

**Goal:** Turn tracing into **one fix line** — wrong **sm_80** blob on **H100** (**`FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`**, **`E1_ERROR_TRACING_NEXT_METHODS.md`**).

---

## A. Guest (VM) — verified **2026-03-22** (assistant)

| Check | Result |
|-------|--------|
| **`libcublasLt.so.12`** | **`…/libcublasLt.so.12.3.2.9`** (dom0-aligned) |
| **`libcudart.so.12`** | **`/usr/lib64` → `libvgpu-cudart.so`** (shim, not bare NVIDIA) |
| **`libvgpu-cuda.so`** | Strings show **CC=9.0** / H100 |
| **systemd `ollama`** | **`OLLAMA_LIBRARY_PATH`** includes **`cuda_v12`** first; **`OLLAMA_NUM_GPU=1`** |
| **Repo `fetch_gpu_info()`** | Forces **`GPU_DEFAULT_CC_MAJOR/MINOR` (9/0)** after host overlay |

**Conclusion:** Software path for **advertised CC** is in place; **E1** blob is still **sm_80** in **`cuobjdump`** — consistent with **embedded Ampere SASS inside cuBLAS Lt** for that kernel package, not a missing symlink in this checklist.

### A2. Runner-equivalent **`ldd`** (done **2026-03-22**)

With Ollama **`LD_LIBRARY_PATH`**, **`libggml-cuda.so`** resolves **`libcublas` / `libcublasLt`** to **`cuda_v12`** **12.3.2.9** and **`libcudart`** to the **shim**. See **`E1_VM_LDD_VERIFICATION_MAR22.md`**. **Other** **`libcublasLt`** copies exist under **`cuda_v13` / mlx** — avoid wrong **`OLLAMA_LLM_LIBRARY`**.

---

## B. Host (dom0) — **you** only if you want extra proof

| Step | Purpose |
|------|---------|
| **Optional:** tiny CUDA **`cuModuleLoadFatBinary("/tmp/fail401312.bin")`** | Confirms **driver** rejects same bytes **outside** mediator (**Method 5**). |
| **Optional:** **`cuobjdump`** on a **new** dump after any libcublas change | See if **sm_XX** changes. |

**Assistant:** read-only logs; **cannot** run CUDA on dom0.

---

## C. If E1 still appears after A — **product / matrix** (human)

1. **libcublas / libcublasLt** build must match **dom0 driver + H100** (NVIDIA compatibility matrix) — may need a **different** **12.x** build than **12.3.2.9** if embedded fatbins still ship **sm_80-only** for your call path.  
2. **mediator** rebuild only if you suspect **transport** (we already ruled out **chunk** reassembly errors for the logged E1 events).

---

## D. Do **not** use as primary fix

- **Longer** HTTP timeouts (does not change blob).  
- **Re-grep** `mediator.log` only (symptom already known).

---

*Use with **`E1_TRACE_TWO_MINUTE_RULE.md`** — stop a method if it gives no signal in ~2 minutes; static proof for E1 is already sufficient.*
