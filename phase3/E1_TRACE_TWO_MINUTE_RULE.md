# E1 tracing — simple rule & different approach

**Rule you asked for:** If a tracing step gives **no useful signal in ~2 minutes**, **stop** that step. **Do not** “just wait longer” as the method.

**E1 in one line:** The host gets a **401312-byte** fatbin, **`cuobjdump`** shows **sm_80** (Ampere). **H100** needs **Hopper** code → **`INVALID_IMAGE`**.

---

## What works without long runs (already done)

| Check | Time | Result |
|-------|------|--------|
| Host: **`cuobjdump`** on **`/tmp/fail401312.bin`** | &lt; 1 min | **sm_80** — wrong arch for H100 |
| Host: **`mediator.log`** grep | &lt; 1 min | **2** loads of **401312**; both failed |
| VM: **libcublas** symlinks | &lt; 1 min | Points to **12.3.2.9** |
| VM: **`grep` chunks** around E1 | &lt; 1 min | **6×65536+8096** = **401312** — data **assembled correctly** |

So: **not** a “needs 2+ hours to see” mystery — the blob **content** is the problem.

---

## Different approach (no long generate)

1. **Trust the static proof** — **sm_80** file on disk + **chunk math** + **smaller** fatbins **succeed** on the same path.  
2. **Fix direction** — make **libcublas Lt** pick **Hopper-compatible** kernels (CC **9.0**, right **Lt** build for driver/H100), not longer HTTP timeouts.  
3. **Optional 2-minute retest** — **`curl -m 10 /api/tags`**, **`systemctl is-active ollama`**, host **tail** — confirms services only; **does not** re-prove E1 (that needs a **module load**, which may take longer — but **you already have** the dump + **cuobjdump**).

---

## Do you need to do anything on the **host**?

| Action | Needed? | Who |
|--------|---------|-----|
| Read **`mediator.log`**, **`cuobjdump`** on **`fail401312.bin`** | For **proof** already done | Assistant read-only **or** you |
| **New** **`fail401312`** after a change | Only if you want a **fresh** file to compare **sha256** / **sm_XX** | Re-run load once (can be slow) **or** skip if old proof is enough |
| **Tiny CUDA program** calling **`cuModuleLoadFatBinary`** on **`fail401312.bin`** (Method 5) | **Optional** double-check | **You on dom0** — assistant cannot build/run there |
| **Mediator** code change (more log on failure) | **Optional** | **You** — rebuild mediator |

**Bottom line:** **No host action is required** to **accept** the current E1 diagnosis. **Host action is optional** if you want an extra **standalone** repro (Method 5) or **new** logs after changing **cuBLAS** / driver.

---

*Added Mar 22, 2026 — aligns with your 2-minute rule: stop blind long waits; use static/binary evidence + correct fix path.*
