# Host fix: module load and module ops use primary context

*Created: Mar 15, 2026 — apply on the mediator host to address INVALID_IMAGE / CUBLAS resource allocation.*

## What was changed (in your local `phase3/src/cuda_executor.c`)

### 1. Primary context for module ops (already applied on host)

Module load and related operations now use the **primary CUDA context** instead of the per-VM context (`ensure_vm_context`), consistent with allocations and CUBLAS:

1. **CUDA_CALL_MODULE_LOAD_DATA / MODULE_LOAD_DATA_EX / MODULE_LOAD_FAT_BINARY**  
   Before calling `load_host_module()` (which calls `cuModuleLoadFatBinary`), the executor now does `cuCtxSetCurrent(exec->primary_ctx)` instead of `ensure_vm_context(exec, vm)`. Loading the module in the same context as allocations and CUBLAS avoids context mismatch and can resolve INVALID_IMAGE when the binary is valid (e.g. Hopper sm_90).

2. **CUDA_CALL_MODULE_UNLOAD**  
   Uses `cuCtxSetCurrent(exec->primary_ctx)` before `cuModuleUnload(mod)`.

3. **CUDA_CALL_MODULE_GET_FUNCTION**  
   Uses `cuCtxSetCurrent(exec->primary_ctx)` before `cuModuleGetFunction()`.

4. **CUDA_CALL_MODULE_GET_GLOBAL**  
   Uses `cuCtxSetCurrent(exec->primary_ctx)` before `cuModuleGetGlobal()`.

### 2. Fat binary passing (try raw, then wrapper)

In `load_host_module()`, when the payload has magic `0xBA55ED50` (raw fat binary from the guest), the code now tries **passing the raw fat binary pointer** to `cuModuleLoadFatBinary()` first. If that returns an error, it falls back to the **wrapper** (0x466243b1) shape. This can resolve INVALID_IMAGE when the driver accepts the raw format but not the wrapper.

No other files were changed. CUBLAS was already using primary context in this tree.

## What you need to do on the host

1. **Get the updated `cuda_executor.c` onto the host**  
   Copy from your local repo to the host (e.g. SCP or git pull):
   ```bash
   scp /home/david/Downloads/gpu/phase3/src/cuda_executor.c root@10.25.33.10:/root/phase3/src/
   ```
   (Adjust paths if your mediator host or phase3 path is different.)

2. **Rebuild the mediator**
   ```bash
   ssh root@10.25.33.10
   cd /root/phase3
   make clean
   make
   ```

3. **Restart the mediator**
   ```bash
   pkill -f mediator_phase3
   nohup ./mediator_phase3 >> /tmp/mediator.log 2>&1 &
   ```

4. **Re-test from the VM**  
   Trigger a generate from test-4 (e.g. `curl -X POST http://127.0.0.1:11434/api/generate ...`). Then check the host log:
   ```bash
   tail -100 /tmp/mediator.log | grep -E 'vm=9|module-load|CUBLAS|SUCCESS|FAILED'
   ```
   You should see `module-load done ... rc=0` (success) if the fix applies, and no INVALID_IMAGE. If CUBLAS was failing before, `cublasCreate_v2` should also succeed with primary context (already in this tree).

## Why this can fix the current error

- **INVALID_IMAGE:** The VM has a valid Hopper (sm_90) `libggml-cuda.so`. The failure at `cuModuleLoadFatBinary` can be due to loading the module in a **per-VM context** that is not the one used for allocations; the driver may reject the image in that context. Using the **primary context** for module load (and get-function / get-global / unload) aligns with allocations and CUBLAS and can clear the error.

- **CUBLAS "resource allocation failed":** If the host mediator was not yet built with primary-context CUBLAS, rebuilding with this tree (which already has it) fixes that. If module load now succeeds, CUBLAS init will run in the same context and should succeed.

## Criteria (from PHASE3_PURPOSE_AND_GOALS.md)

This change is **general**: it does not add Ollama-specific logic. It uses the same primary-context pattern already used for memory and CUBLAS, and does not require matching host CUDA version to VM CUDA.
