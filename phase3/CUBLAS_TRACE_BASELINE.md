# Baseline for cuBLAS / error tracing (avoid mistaken conclusions)

**Audience:** When switching between **short diagnostics** (e.g. `test_cublas_vm`) and **long Ollama loads**, it is easy to mix up **what** failed and **whether** stale state mattered. This note separates concerns.

---

## 1. Mediator on the **host** — when to restart

| Situation | Restart mediator? |
|-----------|-------------------|
| **Minimal `test_cublas_vm`** (seconds, no BAR1 HtoD storm) | **Usually not required** for interpreting **`cublasCreate_v2`** — failure reproduced **without** long mediator history. |
| **After a failed / hung long generate** (hours of HtoD, many `vm_id` calls) | **Recommended** before the **next** long run so host logs and executor state start clean. |
| **When correlating timestamps** between guest journal and `/tmp/mediator.log` | **Optional** clean restart **once**, then reproduce — reduces “which session is this line from?” |

**Practical rule:** For **iterating on cuBLAS** with **`test_cublas_vm`**, mediator reset is **optional**. For **full model load** traces comparable to **`FATBIN_TRACE_RECORD.md`**, **restart mediator + note wall-clock** so logs align.

---

## 2. VM “loading ladder” / Ollama — what is still broken

- **Yes:** Full **Ollama** path (long tensor upload → **`cublasCreate_v2`** in GGML) was still failing with the same class of error as the **minimal** test until a fix lands.
- **Short tests** do **not** exercise the **full ladder** (BAR1, async HtoD, `model load progress`). They answer: **does cuBLAS create work at all** on guest shim + real `libcublas`?

**Do not** assume a passing **`test_cublas_vm`** alone means **tinyllama** full load will succeed — you still need a **short** `curl` smoke after a fix. Conversely, if **`test_cublas_vm`** fails, **mediator long-run state** is **unlikely** to be the root cause.

---

## 3. Avoiding mistakes while tracing

1. **Label runs:** Note **mediator restart?** **ollama restart?** **timestamp** in your paste.
2. **Separate hypotheses:**  
   - **A** — Guest **real `libcublas`** + shim context (**`test_cublas_vm`**).  
   - **B** — Host **fatbin / INVALID_IMAGE** (**mediator `module-load`**).  
   - **C** — Transport / HtoD (timeouts, BAR1).  
   Changing **A** does not fix **B**; fix **B** does not fix **A** unless the same root cause.
3. **Use the right tool:** **`connect_vm.py`** for VM commands (see **`CONNECT_VM_README.md`**), not ad-hoc SSH unless keys are set up.

---

## 4. Ordered steps (assistant plan)

1. **Guest load order** — `libcudart` shim **before** `libcuda` / `libcublas` in **`test_cublas_vm.c`** — **done**; **still `INTERNAL_ERROR`**.  
2. **`cudaSetDevice(0)`** after primary context — **done**; **still `INTERNAL_ERROR`**.  
3. **Resolved:** route **`libcublas.so.12`** to **`libvgpu-cublas`** (RPC). See **`CUBLAS_VENDOR_SYMLINK_DEPLOY.md`**. **`libvgpu_cublas.c`** prefers **`libcublas.so.12.3.2.9`** for **`dlopen`** fallback.

---

*See also: **`INVESTIGATION_CUBLASCREATE_V2.md`***
