# Why `/tmp/fail401312.bin` is stale — and how to get a **new** dump

*2026-03-23 — answers “what must happen for a dump” and “why none since Mar 20.”*

---

## 1. What **creates** `/tmp/fail401312.bin`?

It is **in-tree** in **`phase3/src/cuda_executor.c`** (dump block after **`memcpy`**, before **`cuModuleLoadFatBinary`**). Older trees used only **`host_fatbin_isolation_directive.sh`** to insert the same logic.

The **file write** is added by **`host_fatbin_isolation_directive.sh`**, which inserts code in **`load_host_module()`** **after** `memcpy(fatbin_copy, …)` and **before** `cuModuleLoadFatBinary`, roughly:

```c
if (data_len == 401312U) {
    FILE *df = fopen("/tmp/fail401312.bin", "wb");
    if (df) {
        fwrite(fatbin_copy, 1, data_len, df);
        fclose(df);
        fprintf(stderr, "[cuda-executor] dumped /tmp/fail401312.bin (%u bytes)\n", data_len);
    }
}
```

So a **new** `mtime` **only** happens when:

1. The **running** `mediator_phase3` binary was built **with** this patch, **and**
2. The guest completes forwarding a **module load** with **`data_len == 401312`** (magic `0xBA55ED50` fatbin path) so **`load_host_module`** runs and hits that block.

---

## 2. What we verified on **dom0** (read-only)

- **`sed -n '488,520p' /root/phase3/src/cuda_executor.c`** shows **`memcpy` → comment → `cuModuleLoadFatBinary`** with **no** `fopen("/tmp/fail401312.bin")` — **the dump patch is not present in current host source.**
- **`grep -c 'dumped /tmp/fail401312' /tmp/mediator.log`** → **2** (only two lines **ever**).

So: historical **`fail401312.bin`** and **`[cuda-executor] dumped …`** lines came from an **older** build that **had** the patch; the **current** tree **does not** re-dump the file when a new **401312** load runs.

---

## 3. Why **no** new dump during long VM loads?

Even if the **guest** eventually reached **`cuModuleLoadFatBinary(401312)`** again:

- The **current** mediator **would not** write **`/tmp/fail401312.bin`** without re-applying the patch and rebuilding.
- Separately, many sessions **never** reached that CUDA call: **HtoD** still running, **HTTP client timeout**, **`llama runner` exit status 2**, **context canceled** — **before** the **Lt** fatbin load.

---

## 4. How to **ensure** a **new** dump (ordered)

| Step | Action | Who |
|------|--------|-----|
| **A** | On **`dom0`**: apply **`host_fatbin_isolation_directive.sh`** (or patch **`cuda_executor.c`** by hand as in §1), **`make`**, **restart** `mediator_phase3`. | **You** (host edit; not assistant per **`ASSISTANT_PERMISSIONS.md`**) |
| **B** | Confirm **`grep -n fail401312 /root/phase3/src/cuda_executor.c`** shows the **`fopen`**. | You |
| **C** | Trigger a load that **reaches** a **401312** module load (full model path past weights, or **minimal** repro if you add one). Use **detached** client / no short **`curl -m`** (see **`PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md`**). | VM |
| **D** | **`stat /tmp/fail401312.bin`** — **mtime** should update; **`grep dumped /tmp/mediator.log | tail -1`** should show a **new** line. | **You** or **`connect_host.py`** |

**Without step A**, waiting longer on the VM **cannot** refresh a file the **current** mediator **never** writes.

---

## 5. Relation to **Temporary Stage 1** / **Method 1**

**Method 1** (`cuobjdump` on **`fail401312.bin`**) needs a **fresh** blob after **cuBLAS** (or CC) changes. That requires **both** a **dumping** mediator **and** a completed **401312** load.

---

## Related

- **`host_fatbin_isolation_directive.sh`**
- **`TEMPORARY_STAGE_E1.md`**
- **`E1_ERROR_TRACING_NEXT_METHODS.md`** — Method 1
