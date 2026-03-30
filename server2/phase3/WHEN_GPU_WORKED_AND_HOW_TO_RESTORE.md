# When GPU mode worked — and how to restore it

**Purpose:** The docs say GPU mode (discovery + inference path) did work. This file ties that to the repo and gives a single place to restore from.

---

## 1. Where the docs say it worked

| Source | When | What worked |
|--------|------|-------------|
| **BREAKTHROUGH_SUMMARY.md** | 2026-02-25 | Discovery: 302ms, H100 detected, libggml-cuda loads. Inference “to be verified”. |
| **RESULTS_STATUS.md** | 2026-03-04 | **test-3**: “api/generate (llama3.2:1b) succeeds” with current shim. |
| **RESULTS_STATUS.md** (earlier) | — | “Model loaded and generated successfully.” Response: “How can I help you today?” with `done: true`. |
| **PHASE3_VGPU_CURRENT_STATUS.md** | Mar 6–7 | **test-3**: Mediator log shows “three cuMemAlloc (4 GB, ~1.3 GB, 16 MB) all SUCCESS” — host allocation path worked. Crash was guest-side (runner exit 2). |
| **SUCCESS_MODEL_EXECUTION.md** | 2026-02-27 | “Ollama successfully executed a model and returned a response!” (with CUBLAS shim + assertion intercept). |

So the **last clearly “working” setup in the docs is test-3**: discovery + host allocations (cuMemAlloc SUCCESS), with generate succeeding at least to the point of model load and allocs; later failure was guest-side.

---

## 2. This repo’s git history (no “working” tag)

Relevant commits (from `git log` in this repo):

```
53faaac  2026-03-02  feat:phase3 ollama->shim->vgpu-stub->midator->cuda->gpu first step confirmed
71912b6  2026-03-05  feat:phase3 2026-3-5 gpu-etect
c08d51f  2026-03-07  feat:phase3 host transfer erro
```

- **53faaac (Mar 2)** = “first step confirmed” (pipeline confirmed).
- **71912b6 (Mar 5)** = GPU-related.
- **c08d51f (Mar 7)** = host transfer error (later).

So the version that matched “when GPU worked” on test-3 is likely around **53faaac** or **71912b6**, not necessarily the current tree.

---

## 3. Do you need to “download the version from GitHub”?

- **If the “working” version is only on another machine or another repo (e.g. GitHub):**  
  Yes — get that tree (clone, or download archive) and use it. Then either run from that tree or diff it against this one to re-apply the working behaviour.

- **If the working version was from this same repo:**  
  You don’t have to download from elsewhere. Use this repo’s history:
  1. Try the commit from when test-3 worked, e.g.  
     `git checkout 53faaac -- phase3/`  
     (or `71912b6` if you know that’s the one).  
  2. Rebuild the **host mediator** (and executor) from that `phase3/` and run it.  
  3. If test-4 then works, you’ve found the working version in this repo; you can branch from there or re-apply the same changes on top of main.

So: **only “download from GitHub” if the known-good version lives there and not here.** Otherwise, restore from this repo’s commits above.

---

## 4. Why test-4 fails while test-3 “worked”

- **test-3 (vm_id=8):** Mediator log showed **cuMemAlloc SUCCESS** for the big allocs; failure was **guest-side** (runner exit 2).
- **test-4 (vm_id=9):** Fails earlier with **“unable to allocate CUDA0 buffer”** — first host allocation fails.

So either:

- The **host mediator/executor** that served test-3 was different (e.g. already using primary context for allocation), or  
- test-4’s mediator was built from a tree where allocation uses per-VM context and that path fails for vm_id=9.

The fix documented in **GPU_MODE_DO_NOT_BREAK.md** and **PHASE3_VGPU_CURRENT_STATUS.md** is: use the **primary context** for allocation (and related memory ops) in `cuda_executor.c`, then **rebuild the mediator** on the host. That matches how CUBLAS was fixed (primary context).

---

## 5. Concrete restore options

**Option A — Restore from this repo’s “working” commit**

```bash
cd /path/to/gpu
git stash
git checkout 53faaac -- phase3/
# Rebuild host mediator from phase3 (e.g. make in phase3 or your mediator tree)
# Restart mediator, test with test-4 (or test-3) again
```

If that fixes test-4, then branch or cherry-pick from 53faaac and re-apply any later changes you want to keep.

**Option B — Keep current code and fix allocation (recommended)**

- Current `phase3/src/cuda_executor.c` is already updated to use **primary context** for MEM_ALLOC, MEM_FREE, MEMCPY_*, MEMSET_*, MEM_GET_INFO (see **PHASE3_VGPU_CURRENT_STATUS.md**).
- **Rebuild the mediator** on the host with this tree and restart it.
- Run a generate on test-4 again; if the only problem was allocation, it should get past “unable to allocate CUDA0 buffer”.

**Option C — You have a known-good copy elsewhere (e.g. GitHub)**

- Clone or download that “working” tree.
- Build the mediator (and guest shims if needed) from that tree.
- Run mediator on the host and point test-4’s VGPU at it.
- Then diff that tree against this one to see what to re-apply here long term.

---

## 6. Summary

- **When it worked (from docs):** test-3, around Mar 4–7: discovery + host allocations (cuMemAlloc SUCCESS); generate succeeded at least through model load; later failure was guest-side.
- **Where to restore from:**  
  - Either this repo at commit **53faaac** (or **71912b6**), or  
  - A “working” version you have on GitHub/elsewhere.  
  You only need to “download from GitHub” if the known-good version is there and not in this repo.
- **Why test-4 fails now:** First allocation fails on the host for vm_id=9; fix is primary-context allocation in `cuda_executor.c` and **rebuilding the mediator** (Option B), or restoring a mediator/executor build from when test-3 worked (Option A or C).
