# E1 — next error tracing methods (beyond host log grep)

*Mar 22, 2026 — clarifies that **grep of `mediator.log`** alone is **not** sufficient to close **E1**; it only confirms the symptom repeats.*

**E1 (symptom):** `cuModuleLoadFatBinary` with **`data_len=401312`** → **`CUDA_ERROR_INVALID_IMAGE` (`rc=200`)** on the host.

**Why prior tracing was “ineffective”:** It established **that** the failure occurs and **how often**, but not **which embedded kernels** are wrong, **which guest library** produced the blob, or **whether** the mediator path is lossless — i.e. it did not narrow the fix enough.

---

## Method 1 — Binary forensics on the dumped payload (dom0)

**Artifact:** `[cuda-executor] dumped /tmp/fail401312.bin` (host).

| Step | Command / action | What you learn |
|------|------------------|----------------|
| 1 | `ls -la /tmp/fail401312.bin` | Size matches **401312** (integrity). |
| 2 | `cuobjdump -elf /tmp/fail401312.bin \| grep -E 'arch =|ptx'` | **sm_XX** actually inside the image (e.g. **sm_80** vs **sm_90**). |
| 3 | `cuobjdump --list-text /tmp/fail401312.bin` (or symbols) | Kernel naming (e.g. **cublasLt** / **GEMM** patterns vs GGML). |

**Already documented result (Mar 21):** **`FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`** — dump shows **sm_80** / Ampere-style names → points to **libcublasLt** selection, not GGML’s own TU set.

**Repeat when:** After **any** change to guest **cuBLAS**, **shim CC**, or **mediator** — confirm whether the **401312** blob **changes** (arch / symbols).

---

## Method 2 — Contrast with **successful** module loads (same log)

From **`mediator.log`**, loads with **`data_len` in {9360, 28120, 4432}** often show **`rc=0`**.

| Step | Action |
|------|--------|
| 1 | If the host can dump **those** payloads too (may require a one-line mediator tweak to dump on success — **human**), run the same **`cuobjdump`** on a **good** blob. |
| 2 | Diff: **magic**, **ELF vs fatbin**, **sm_XX** — proves what “good” looks like on **this** driver. |

*If dumping successes is not implemented, Method 1 on **`fail401312.bin`** plus **`FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`** remains the main binary signal.*

---

## Method 3 — Guest-side provenance (VM)

Trace **which DSO** supplies the failing path:

| Step | Command / check |
|------|-----------------|
| 1 | `ldd /usr/local/lib/ollama/cuda_v12/libcublasLt.so.12` and realpath of **`libcublasLt.so.12`** |
| 2 | `strings` / version symbols on that **`.so`**; compare to **dom0** CUDA **12.3.x** alignment (see **`ERROR_TRACKING`** / install scripts). |
| 3 | Confirm **`gpu_properties.h`** and **`libvgpu_cuda.c`** **force 9.0** after host overlay — wrong CC steers **Lt** fatbin choice (**`FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`**). |

---

## Method 4 — Mediator code path audit (repo + dom0 parity)

**File:** **`phase3/src/cuda_executor.c`** — **`load_host_module()`**.

| Check | Why |
|-------|-----|
| **`CUDA_CALL_MODULE_LOAD_FAT_BINARY`** uses **`use_fatbinary`** and **`0xBA55ED50`** raw then **wrapper** fallback | Confirms host tries both shapes (see comments ~L475–L511). |
| **Chunked reassembly** (same file, **MODULE_LOAD** handler) | If chunks are mis-ordered or truncated, magic is fine but payload is corrupt → **INVALID_IMAGE**. Worth verifying logs for **`module-chunk`** lines around failing **call_id**. |
| **Primary context** | **`HOST_FIX_MODULE_LOAD_PRIMARY_CTX.md`** — context mismatch can present as bad image. |

**Assistant can:** read/diff repo, document; **human** rebuilds mediator on dom0.

---

## Method 5 — Minimal host repro (dom0)

Load **`/tmp/fail401312.bin`** with a tiny **CUDA** program or **`cuda-memcheck` / driver test** using **`cuModuleLoadFatBinary`** outside the mediator — proves **driver + blob** interaction without VM variables.

---

## Method 6 — Stronger logging on failure (mediator change, human deploy)

Ideas (require **dom0** build):

- On **`cuModuleLoadFatBinary` failure**, log **`cuGetErrorString`** / extended driver message if available.
- Optionally dump **first/last 256 bytes** of payload to correlate with chunk assembly.

---

## Ordering (suggested)

1. **Re-run Method 1** on current **`fail401312.bin`** after latest guest libcublas align — **did sm_XX change?**  
2. **Method 3** on VM — **libcublasLt** realpath and versions.  
3. **Method 4** — chunk logs around **E1** timestamps vs **rc=0** loads.  
4. **Method 5** if still ambiguous.  
5. **Method 6** only if transport/path is suspected.

---

## Relationship to E2

**E2** (**`compute=8.9` vs 9.0** in discovery) is a **contributor** to wrong **Lt** kernel selection — see **`TRACE_E2_COMPUTE_89_ROOT_CAUSE.md`**. It **does not replace** E1 tracing; fix **CC reporting** and **re-validate E1** with **Method 1**.

---

## Execution log (assistant)

### Method 1 — **2026-03-22** (host, read-only)

| Item | Result |
|------|--------|
| **`/tmp/fail401312.bin`** | Present, **401312** bytes, mtime **Mar 20 11:52** (⚠ **older than** VM **cuBLAS 12.3.2.9** align **Mar 22** — **not** proven to be the blob from post-align runs). |
| **`sha256`** | `2a858c6781f946b6e2d209c7159cf31e19f877e2d3364d9bf6bebd7cc1203fc2` |
| **`cuobjdump -elf`** | **`arch = sm_80`**, **`sm=80`**, section names **`ampere_h16816gemm_*`** / **`ampere_h1688gemm_*`** — **cuBLAS Lt**-style **Ampere** SASS, **not** Hopper **sm_90**. |

**Interpretation:** Matches **`FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`** — H100 rejects **sm_80-only** image → **`INVALID_IMAGE`**. To see whether **12.3.2.9** + CC **9.0** still emits this, **re-dump** after a fresh failing load (or clear **`/tmp/fail401312.bin`** and reproduce so mediator writes a **new** file; compare **mtime** + **sha256**).

### Method 2 — **2026-03-22** (host log contrast)

**Note:** The mediator only writes **`/tmp/fail401312.bin`** on the failing path; there is **no** on-disk dump for **`rc=0`** loads in this setup — contrast is from **logged `data_len`** + outcome.

| `data_len` | Count (`module-load start`) | Outcome (paired `module-load done`) |
|------------|----------------------------|----------------------------------------|
| **4432** | 2 | **`rc=0`** |
| **9360** | 2 | **`rc=0`** |
| **28120** | 4 | **`rc=0`** |
| **401312** | 2 | **`rc=200`** **`INVALID_IMAGE`** |

**Read:** Smaller fatbins (**4–28 KB**) load successfully on the **same** `vm_id=9` / **`call_id=0x0042`** path; only the **~401312 B** payload fails — consistent with **wrong SASS arch** inside that blob (**Method 1**), not a universal transport break.

### Method 4 — **2026-03-22** (chunk / context around E1)

For **both** E1 occurrences in **`mediator.log`**, the failing load is **immediately preceded** by **chunked** `call_id=0x0042` reassembly:

| Segment | `data_len` | Notes |
|---------|------------|--------|
| First | **65536** | `flags` first=1, `first8=50ed55ba…` (fatbin magic) |
| Middle | **65536** × **5** | Continuation chunks |
| Last | **8096** | `flags` last=1 |

**Arithmetic:** \(6 \times 65536 + 8096 = 393216 + 8096 =\) **401312** — matches **`module-load start`** exactly.

**Interpretation:** **Reassembly is consistent** (size + leading magic). Failure is **not** explained by missing chunks or wrong total length in these two events; it points to **contents** (**sm_80** in **`cuobjdump`**) vs **H100** expectations.

**Surrounding lines:** Normal **`cuMemAlloc`** and **`MEDIATOR` `result.status=0`** before chunks; after E1, **`CUDA_CALL_INIT`** and routine **`result.status=0`** traffic — pipeline continues after reporting **`INVALID_IMAGE`** to guest.

**`module-chunk` line count** in log: **156** (many loads, not only 401312).

### Fresh probe — **2026-03-22** (background preload + host snapshot)

**Intent:** Exercise the stack **without** tying to a single long **`curl -m`** from the laptop; check whether a **new** **`401312`** / **`fail401312.bin`** appears post–**libcublas 12.3.2.9** align.

| Step | Result |
|------|--------|
| **VM** | **`nohup curl … /api/generate`** with **`{"model":"tinyllama:latest","keep_alive":-1}`** (empty preload) — **`curl`** still running after **90 s**; journal **`poll call_id=0x0032`** (HtoD) **seq=410** — transport active. |
| **Host `mediator.log`** | Lines **2175 → 2196** (**+21**); **`grep -c 'data_len=401312'`** still **2** — **no new** E1 **`module-load start`** lines in this window. |
| **`/tmp/fail401312.bin`** | **mtime** still **2026-03-20 11:52**; **sha256** unchanged vs prior capture — **not** rewritten (only written on dump at **INVALID_IMAGE** path). |

**Read:** This session **did not** produce a **new** fatbin failure record on the host; ongoing work is **weight HtoD**. To force a **fresh** **`fail401312`** dump, a code path must again execute the **401312** **Lt** load and fail — watch **`module-load start … 401312`**; if a **third** line appears, re-run **Method 1** on the new **`/tmp/fail401312.bin`**.

### Method 3 — **2026-03-22** (VM)

| Library | Resolved path |
|---------|----------------|
| **`libcublasLt.so.12`** | → **`/usr/local/lib/ollama/cuda_v12/libcublasLt.so.12.3.2.9`** |
| **`libcublas.so.12`** | → **`/usr/local/lib/ollama/cuda_v12/libcublas.so.12.3.2.9`** |

Symlinks dated **Mar 22**; binaries **Mar 22 13:43** — dom0-matched align **in place**.

---

*This doc supersedes any message that “E1 tracing is complete” based only on **grep** of **`mediator.log`**.*
