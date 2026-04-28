# Host fix: event/stream lookup failures no longer report CUDA_SUCCESS

## Problem

In `phase3/src/cuda_executor.c`, several **event** and **stream wait** cases did nothing when `vm_find_event()` returned NULL (unknown handle), but left **`rc == CUDA_SUCCESS`**. The guest then believed the call succeeded (e.g. `cuEventRecord`, `cuStreamWaitEvent`) when the host performed **no operation**. That can desync GGML/CUDA state and contribute to **later crashes** (e.g. exit status 2) after long loads.

## Changes (non-destructive to successful paths)

1. **`CUDA_CALL_STREAM_WAIT_EVENT`**  
   - Require a resolved **event**; require a resolved **stream** when `stream_handle != 0` (handle `0` = default stream / NULL).  
   - On failure: `CUDA_ERROR_INVALID_HANDLE`.

2. **`CUDA_CALL_EVENT_DESTROY`, `EVENT_RECORD`, `EVENT_SYNCHRONIZE`, `EVENT_QUERY`**  
   - If the event is not in the map: `CUDA_ERROR_INVALID_HANDLE`.  
   - **EVENT_RECORD**: same stream rules as above (`0` => NULL).

3. **`CUDA_CALL_EVENT_ELAPSED_TIME`**  
   - If either event is missing: `CUDA_ERROR_INVALID_HANDLE`.

4. **`CUDA_CALL_EVENT_CREATE`**  
   - **`vm_add_event`** now returns `int` (0 if `MAX_EVENT_ENTRIES` full).  
   - If the table is full: **`cuEventDestroy`** the new event, return **`CUDA_ERROR_OUT_OF_MEMORY`**, do not return a handle to the guest.

## Deploy

This file is compiled into the **host** CUDA executor (mediator build). Rebuild and deploy **`cuda_executor`** / mediator on the **GPU host** per your usual process (e.g. `deploy_cuda_executor_to_host.py` or dom0 build). **VM / guest shim unchanged** for this fix.

## Anti-regression

- Successful event create/record/sync/query/destroy paths unchanged when handles are valid.  
- Default stream (`stream_handle == 0`) still maps to NULL for `cuEventRecord` / `cuStreamWaitEvent`.
