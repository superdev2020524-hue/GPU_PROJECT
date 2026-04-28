# BAR0 / BAR1 and fixed response value — reference (what was done)

This document ties together the steps you remember: **BAR0 failing → using BAR1 to proceed**, and **setting the response value to a fixed 1 or 2** in the context of the **40-minute transfer** and expediting it.

---

## 1. BAR0 failed → use BAR1 to proceed

Two separate uses of “BAR0 vs BAR1” exist in phase3:

### A. Status register (poll for completion)

- **Problem:** The guest’s MMIO read of the **status** register (BAR0 offset 0x004) returns **0x01 (BUSY)** even when the stub has set **0x02 (DONE)**. So “BAR0 status read” is broken on this Xen/qemu-dm path (guest never sees DONE).
- **Step taken:** Because BAR0 status was wrong, we **use BAR1 for status** instead:
  - **Guest:** `guest-shim/cuda_transport.c` maps BAR1 and reads status from the **BAR1 status mirror** (last 4 bytes of BAR1: `BAR1_STATUS_MIRROR_OFFSET = BAR1_SIZE - 4`). See comment: *“Workaround: BAR0 status read returns stale value on some Xen/qemu-dm; poll BAR1 tail instead”* (line ~313). Transport init: *“Always map BAR1 for status mirror (avoids broken BAR0 status path)”* (line ~828).
  - **Stub:** `phase3/src/vgpu-stub-enhanced.c` maintains `bar1_status_mirror` and returns it when the guest reads the last 4 bytes of BAR1.
- **Result:** Guest and host logs show BAR1 status reads do reach the stub and stub returns 0x2, but the **guest still sees 0x01**. So switching to BAR1 for status did **not** fix the visibility bug (same wrong value on delivery; see MMIO_MISMATCH_CAUSE_DIAGNOSIS.md, ACTUAL_ERROR_MAR16_STATUS_AND_SHMEM.md). The “use BAR1 to proceed” here means: we **tried** BAR1 for status because BAR0 failed; both paths still show the same bug.

**References:**  
`guest-shim/cuda_transport.c` (BAR1 status mirror, init), `MMIO_MISMATCH_CAUSE_DIAGNOSIS.md`, `ACTUAL_ERROR_MAR16_STATUS_AND_SHMEM.md`, `HtoD_DIAGNOSIS_RESULTS.md`.

### B. Data path (payload: model/alloc/HtoD)

- **Problem:** Shared-memory registration can fail on the host (e.g. `cpu_physical_memory_map(H2G)` fails). Guest then cannot use shmem for large payloads.
- **Step taken:** When shmem registration fails, the guest **falls back to BAR1 for data** (legacy 16 MB BAR1 window, chunked). So “BAR0” (or rather the preferred path) “failed” and we **replace / fall back to BAR1** for the **data** path to keep sending requests (cuMemAlloc, HtoD, etc.).
- **Result:** Runs can proceed (e.g. 40-minute transfer with 295 HtoD RPCs) over BAR1 data path; the blocker remains **completion visibility** (guest not seeing DONE on status, whether read from BAR0 or BAR1).

**References:**  
`ACTUAL_ERROR_MAR16_STATUS_AND_SHMEM.md` (“vgpu-stub rejected shmem registration … — using BAR1”), `guest-shim/cuda_transport.c` (“Fallback: map BAR1 (legacy data region)”).

---

## 2. Setting the response value to a fixed 1 or 2

Two “fixed” values appear in the design:

### “Fixed 1” — `response_len = 1` (workaround for completion)

- **Stub:** When the stub applies a CUDA result (DONE), it sets **`s->response_len = 1`** so the guest can detect completion by reading BAR0+0x01C instead of the broken status register.
- **Location:** `phase3/src/vgpu-stub-enhanced.c` (in `VGPU_MSG_CUDA_RESULT` branch):  
  `s->response_len = 1;`  
  and when starting a new request (doorbell): `s->response_len = 0`.
- **Guest:** Poll loop checks `REG_RESPONSE_LEN` (BAR0+0x01C) after **30** poll iterations; if `rlen != 0`, treat as DONE. Do **not** use 3 iterations (that was reverted; see MMIO_WORKAROUND_RESPONSE_LEN.md, VERIFICATION_REPORT.md).

So “fixed 1” here is: **response_len set to 1** when the host has finished the request.

### “Fixed 2” — status value 0x02 (DONE)

- **Protocol:** Status register values are **0x01 = BUSY**, **0x02 = DONE**, **0x03 = ERROR** (see `include/vgpu_protocol.h` or guest `STATUS_BUSY` / `STATUS_DONE` / `STATUS_ERROR`).
- The stub sets status to **0x02 (DONE)** when the result is applied; the guest is supposed to see this value to exit the poll loop. On this stack the guest does **not** see 0x02 (it sees 0x01), which led to the `response_len = 1` workaround above.

So “fixed 2” is the **status value 2 (DONE)** that the host writes but the guest often does not observe.

---

## 3. 40-minute transfer and expediting

- **Observation:** A 40-minute run could send **295** HtoD (and other) RPCs; the host applied 292+ results. So the **data path** (including BAR1 fallback) and mediator→stub path were working; the guest still did **not** complete load because it never saw DONE on the status read (HtoD_DIAGNOSIS_RESULTS.md §10–11).
- **Expediting attempt:** To avoid relying on the broken status read, we added the **response_len** workaround (fixed 1): stub sets `response_len = 1` on DONE; guest checks BAR0+0x01C after 30 poll iterations. An earlier variant checked after **3** iterations (“Fix 1”); that was **reverted** because it broke long-duration transmission (guest could exit on stale data). The correct value is **30** (MMIO_WORKAROUND_RESPONSE_LEN.md, VERIFICATION_REPORT.md).

So: the long (40-minute) transfer was possible over BAR1 data path; “expediting” was attempted by using **response_len (fixed 1)** and the right poll threshold (30, not 3).

---

## 4. Where to find this in the repo

| Topic | File(s) / doc |
|-------|----------------|
| BAR0 status broken, use BAR1 for status | `guest-shim/cuda_transport.c` (BAR1_STATUS_MIRROR_OFFSET, “status mirror”), `MMIO_MISMATCH_CAUSE_DIAGNOSIS.md` |
| Shmem failed, fallback to BAR1 for data | `ACTUAL_ERROR_MAR16_STATUS_AND_SHMEM.md`, `guest-shim/cuda_transport.c` (setup_shmem, “Fallback: map BAR1”) |
| response_len = 1 (fixed 1) | `phase3/src/vgpu-stub-enhanced.c` (VGPU_MSG_CUDA_RESULT), `MMIO_WORKAROUND_RESPONSE_LEN.md` |
| Status 0x02 (DONE) / fixed 2 | `include/vgpu_protocol.h` (VGPU_STATUS_*), guest `STATUS_DONE` |
| 40-min run, 295 HtoD, reply path | `HtoD_DIAGNOSIS_RESULTS.md` §10–11 |
| Expedite: response_len, 30 vs 3 | `MMIO_WORKAROUND_RESPONSE_LEN.md`, `VERIFICATION_REPORT.md` |

---

## 5. GitHub

A search for “vgpu BAR0 BAR1 response_len phase3 Xen” on GitHub did not find this project’s internal docs (they live in your local phase3 tree). The design and history above are in the phase3 markdown files and source references listed in this document.
