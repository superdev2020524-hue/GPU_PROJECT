# Actual errors found (Mar 16, 2026)

## 1. Journal / runner

- **Ollama journal** (VM): After a generate that returns 500, journal shows:
  - Goroutine stacks and a **register dump** (rax, rbx, rip=0x725738f1a109, …) from `ollama.bin[PID]`.
  - Then: `msg="waiting for server to become available" status="llm server error"`.
  - Then: `msg="Load failed" … error="llama runner process has terminated: exit status 2"`.
  - Then: `[GIN] … 500 | 8.19s | POST "/api/generate"`.

- **Interpretation:** The **llama runner** process (the one that loads the model and uses libvgpu-cuda) **exits with status 2**. The register dump may be from that process’s state when it died (or from the Go server dumping state). Exit status 2 is the **process exit code**, not a signal (signals yield 128+signal).

## 2. Where exit 2 can come from

- **Our transport:** We return `2` on timeout, on BAR STATUS_ERROR, or when `result->status != 0` (host CUDA error). In those cases we also call `cuda_transport_write_error(...)`, which writes `/tmp/vgpu_last_error` and `/tmp/vgpu_debug.txt`.
- **Observed:** After a 500 run, `/tmp/vgpu_last_error` and `/tmp/vgpu_debug.txt` are **not** present on the VM (even with sudo). So either:
  - The process that exited is **not** taking our error path (timeout / STATUS_ERROR / CUDA_CALL_FAILED), or
  - That process runs with a different view of `/tmp` (e.g. namespace) so we don’t see the files.

- **Host:** In `/var/log/daemon.log`, for vm_id=9 all applied results show **status=0** in the stub log (`CUDA result applied seq=N status=0 (DONE)`). So the **transport** is receiving DONE and the stub is applying **CUDA status 0** for those responses. If the mediator had sent a CUDA error (e.g. status=2), the stub would set VGPU_STATUS_ERROR and log `status=2 (CUDA_ERROR)`; we don’t see that in the tail.

## 3. Call sequence when it fails

- **VM** `/tmp/vgpu_call_sequence.log`: Last entries are **0x00ae** (CUBLAS_SET_STREAM) and **0x00b5** (CUBLAS_GEMM_EX). Many RPCs complete (HtoD, alloc, cuInit, 0x00ac, 0x00ae, 0x00b5, …). So BAR1 status path is working and the host is replying.

## 4. Diagnostics added

- **Stub:** When applying a CUDA result with `cr->status != 0`, the stub now logs explicitly:  
  `CUDA result applied seq=N status=X (CUDA_ERROR — guest will see ERROR)`.
- **Guest:** Right before returning any non-zero value from `cuda_transport_call`, the guest now writes one line to **`/tmp/vgpu_transport_returned_nonzero`** with `ret=… call_id=0x… seq=…`.

## 5. How to confirm the source of exit 2

1. **Redeploy** guest (transfer_cuda_transport.py) and, on the host, rebuild/install the stub (see HOST_STUB_REBUILD_INSTRUCTIONS.md).
2. Trigger a generate that returns 500.
3. **On the VM:**
   - `cat /tmp/vgpu_transport_returned_nonzero`  
     If this file exists, the transport **did** return non-zero; the line gives `ret`, `call_id`, and `seq`.
   - `sudo cat /tmp/vgpu_last_error /tmp/vgpu_debug.txt`  
     If these exist, the failure was timeout, STATUS_ERROR, or CUDA_CALL_FAILED and the report has details.
4. **On the host:**  
   `grep -a 'CUDA_ERROR\|status=2\|status=3' /var/log/daemon.log | tail -20`  
   If any line shows `(CUDA_ERROR)` or `status=2`, the mediator sent a non-zero CUDA status for that seq.

## 6. Summary

| Finding | Meaning |
|--------|--------|
| Runner exits with status 2 | Process calls `exit(2)` or returns 2 from main. |
| No vgpu_last_error / vgpu_debug.txt | Our transport error path (and thus write_error) was not executed, or /tmp is not shared. |
| Host log shows status=0 for applied results | Mediator is sending CUDA status 0 in the responses we inspected. |
| Call sequence ends with 0x00ae / 0x00b5 | Many RPCs succeed; failure happens during/after CUBLAS set_stream or gemm_ex. |

**Check done (after adding guest write):** After deploying the guest that writes `/tmp/vgpu_transport_returned_nonzero` when returning non-zero, a generate was run; the file was **not** created. So **the transport is not returning a non-zero value** when the runner exits with status 2. The exit 2 therefore comes from **elsewhere** in the runner/ollama/llama stack (e.g. Go runner exit code, or a path in the C/llama code that does not go through our transport return value).
