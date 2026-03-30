# Transmission outcomes and progress assessment

*Created: Mar 18, 2026 — honest summary of what happens when transfer completes or is interrupted, progress since early March, and how to describe failure.*

---

## 1. What results do we get if transmission completes or is interrupted?

### If transmission **completes** (HtoD runs to full model size)

From **CURRENT_STATE_AND_DIRECTION.md**, **PHASE3_VGPU_CURRENT_STATUS.md**, and **OLLAMA_VGPU_REVISIONS_STATUS.md**:

- **Host:** HtoD progress reaches ~1.2–2.5 GB (model weights). Then follow-up `cuMemAlloc` (4 GB, ~1.3 GB, 16 MB, etc.) all **SUCCESS**.
- **Next step:** Host loads the CUDA **module** (fat binary from VM’s `libggml-cuda.so`) via `cuModuleLoadFatBinary`.
  - **Current VM binary:** Built **without** sm_90 (Hopper). Host returns **CUDA_ERROR_INVALID_IMAGE (rc=200)** — "device kernel image is invalid".
- **Result:** Runner sees module-load failure (or no kernel for H100). Ollama reports **HTTP 500** and *"llama runner process has terminated: exit status 2"* (or similar). So **successful transfer is followed by a known, documented failure** at module load, not by working inference.

**Conclusion:** If this run’s transmission completes, we should expect the **same** outcome as in the docs: transfer finishes, then **module-load INVALID_IMAGE** (unless a Hopper-built `libggml-cuda.so` has already been deployed to the VM and is in use).

### If transmission is **interrupted** (timeout, disconnect, or reply path stuck)

From **ACTUAL_ERROR_VERIFICATION.md**, **HtoD_DIAGNOSIS_RESULTS.md**, **ACTUAL_ERROR_MAR16_STATUS_AND_SHMEM.md**:

- **Client timeout (e.g. 5 min):** Server logs *"Load failed ... timed out waiting for llama runner to start - progress 0.00"* → **HTTP 500**. Runner may still be blocked in transport (e.g. waiting for DONE on BAR0).
- **Reply path broken (guest never sees DONE):** Runner blocks in poll loop (status stays 0x01 BUSY), hits load timeout, exits with **exit status 2** → 500.
- **Connection closed / client disconnect:** *"client connection closed before server finished loading"*; request ends 499 or 500.

**Conclusion:** If transmission is interrupted, we get timeouts, 500, and exit 2 for the reasons above — no new “excuse,” just the same categories already documented.

---

## 2. “Such transmissions have taken place before”

Yes. The phase3 docs describe multiple runs where **bulk HtoD completed** (or nearly so):

- **test-3:** HtoD progress up to **~1250 MB**; then allocs (4 GB, ~1.3 GB, 16 MB) SUCCESS; then **runner exit 2, HTTP 500** (CURRENT_STATE_AND_DIRECTION.md, PHASE3_VGPU_CURRENT_STATUS.md).
- **test-4:** HtoD progress up to **~2.5 GB**; then **module-load done rc=200 INVALID_IMAGE** (CURRENT_STATE_AND_DIRECTION.md).
- **HtoD_DIAGNOSIS_RESULTS.md:** An older vm=9 run had **7672 MB** HtoD in the mediator log (full model transfer).

So the **pattern is established**: long transfer can complete; the failure is **after** transfer (module load or runner/CUBLAS step), not “transmission failed.”

---

## 3. What actual progress have we made in nearly two weeks?

**Reference window:** Docs and ERROR_TRACKING_STATUS are from **Mar 15–18, 2026**. “Nearly two weeks” is taken as progress since that state.

### Then (early March)

- **Runner often never reached alloc/HtoD:**  
  - LD_LIBRARY_PATH for runner lacked `/opt/vgpu/lib` (fixed).  
  - Sched: `num_gpu == 0` → `gpus = []` → CPU load (patch_sched_numgpu).  
  - GetRunner: cold-load/reload requests never enqueued to `pendingReqCh` (else-branch fix).  
  - Server blocked in `LoadModelFromFile` (VocabOnly); bypass with `model.NewTextProcessor(modelPath)` so runner can start.
- **After those fixes:** Runner reached alloc (cuMemAlloc 3×) and HtoD; commit and HtoD confirmed; timeouts aligned (OLLAMA_LOAD_TIMEOUT=40m, CUDA_TRANSPORT_TIMEOUT_SEC=2700).
- **Known blocker after transfer:** Host `cuModuleLoadFatBinary` → INVALID_IMAGE (VM’s libggml-cuda.so not built for sm_90). Fix documented: **BUILD_LIBGGML_CUDA_HOPPER.md** + deploy to VM.

### Now (current run)

- **Runner does send alloc and HtoD:** Host log shows HtoD progress **675+ MB**, request_id 735 for 0x32; VM sent 724 HtoD, received DONE through seq 730. Transfer is **in progress** and host/VM are in sync; no TIMEOUT/RESPONSE_LEN in verify log.
- **Blocker after transfer is unchanged:** We have **not** yet confirmed that a **Hopper-built** `libggml-cuda.so` was deployed to this VM and used for a full load. So when this run’s transfer completes, we should still expect **module-load INVALID_IMAGE** (or, if that were fixed, the next documented failure: CUBLAS / runner exit 2 per OLLAMA_VGPU_REVISIONS_STATUS.md).

### Summary of progress

| Area | Then | Now |
|------|-----|-----|
| Runner reaches alloc/HtoD | No (sched, GetRunner, LoadModelFromFile, env) | **Yes** — alloc and HtoD in progress |
| Transfer progress | Often 0 or stuck at first HtoD (reply path / status visibility issues) | **Ongoing** — 675+ MB, host and VM in sync |
| Post-transfer failure | Documented: INVALID_IMAGE or exit 2 | **Same** — fix (Hopper .so) documented but not confirmed deployed on this VM |

So the **actual progress** is: we moved from “runner never sends alloc/HtoD” and “transfer doesn’t really run” to “runner sends alloc/HtoD and transfer is actively progressing.” The **next** failure point (after transfer) is the same as in the docs; we have not yet closed that loop with a proven Hopper deploy and success.

---

## 4. If it fails, what do we say? (No “excuse” — the technical truth)

- **If transfer completes and then it fails:**  
  “Transfer completed (HtoD to ~X MB, host and VM in sync). Failure is at the **next** step: host `cuModuleLoadFatBinary` returns INVALID_IMAGE because the VM’s `libggml-cuda.so` is not built for sm_90 (Hopper/H100). Fix: build and deploy a Hopper-built library per BUILD_LIBGGML_CUDA_HOPPER.md. If module load were fixed, the next known failure is runner exit 2 after CUBLAS/alloc (see OLLAMA_VGPU_REVISIONS_STATUS.md).”

- **If transfer is interrupted:**  
  “Transfer was interrupted: client timeout, or runner blocked waiting for response (reply path / status visibility), or disconnect. That matches the failure modes in ACTUAL_ERROR_VERIFICATION.md and HtoD_DIAGNOSIS_RESULTS.md.”

- **One-line version:**  
  We don’t need an “excuse.” We report the **observed outcome** (transfer completed vs interrupted) and the **documented cause** (INVALID_IMAGE after transfer, or timeout/reply-path if interrupted). The root cause and the intended fix are already in the phase3 docs.
