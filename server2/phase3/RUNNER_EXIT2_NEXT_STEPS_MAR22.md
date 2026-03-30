# Runner exit status 2 — next steps (Mar 22)

## Status (assistant executed)

| Item | Result |
|------|--------|
| **`/etc/systemd/system/ollama.service.d/coredump.conf`** | **Present** — `LimitCORE=infinity` |
| **`coredumpctl`** | **`/usr/bin/coredumpctl`** available |
| **Stored cores** | **`coredumpctl list`** → **none** (no core captured for past crashes yet) |
| **`/tmp/vgpu_call_sequence.log` (tail)** | Ends with **`0x0044 cuModuleGetFunction`** → **`0x0061 cuStreamCreateWithFlags`** → **`0x0063 cuStreamDestroy`** |
| **`/tmp/vgpu_current_call.txt`** | **`call_id=0x0063 cuStreamDestroy seq=846 pid=<runner>`** |

So the **last instrumented CUDA op** before the **exit 2** / register dump was **`cuStreamDestroy`** (see `cuda_transport.c` name for `CUDA_CALL_STREAM_DESTROY`).

## Interpretation

- Failure remains **native (C/C++/GGML/CUDA)** after a long **HtoD** + **module load** path; Go stack shows **WaitGroup.Wait** in the runner parent — child process **signaled/crashed** (typical **exit 2** / **SIGINT**-like behavior depends on how the loader reports it; journal showed **register dump** → treat as **SIGSEGV in native code** until a core proves otherwise).
- **No new core** yet: cores may not have been written (older crash), or **Apport**/policy blocked storage — after the next repro, run **`coredumpctl list`** immediately.

## VM package health (warning)

`apt-get install` hit **dpkg / DKMS / nvidia-driver** errors. **Avoid mass `apt upgrade`** on this VM until dom0/VM admin fixes the broken packages. **`systemd-coredump`** is installed despite partial failure output — verify with `which coredumpctl`.

## Ordered next steps

### 1) Capture a core on the **next** crash (VM)

1. Confirm drop-in: `sudo cat /etc/systemd/system/ollama.service.d/coredump.conf`
2. Re-run a **long** generate (same as Checkpoint D: `curl -m 7200` …) or wait for natural repro.
3. Right after **`llama runner terminated`**:  
   `coredumpctl list -n 20`  
   `coredumpctl dump -o /tmp/core.ollama ollama.bin.new` (adjust executable name if different)
4. On VM:  
   `gdb /usr/local/bin/ollama.bin.new /tmp/core.ollama` → **`bt full`**, **`info registers`**, **`info sharedlibrary`**

### 2) Correlate with transport + host

- Guest **`cuda_transport.c`** maps **`0x0063`** → **`cuStreamDestroy`**.
- **Repo fix (Mar 22):** **`phase3/src/cuda_executor.c`** — **`CUDA_CALL_STREAM_DESTROY`** previously left **`rc = CUDA_SUCCESS`** when **`vm_find_stream`** returned **NULL** (silent success → **desync**). Now returns **`CUDA_ERROR_INVALID_HANDLE`**. **You (dom0)** must **rebuild and redeploy `mediator_phase3` / `cuda_executor`** for this to take effect on the host.
- See **`WORK_NOTE_HOST_EVENT_STREAM_FIX.md`** for related event/stream handle semantics.

### 3) Host / libcublas (human, dom0) — plan §6 Step 3–4

If **`401312`** / **`INVALID_IMAGE`** reappears on **`module-load`** for the **cuBLASLt** blob: **`load_host_module()`** semantics + **`libcublasLt.so.12`** per **`FATBIN_CUBLAS_CC_ANALYSIS_MAR21.md`**.

### 4) Optional: faster repro experiments (VM)

- **`OLLAMA_DEBUG=1`** already on — keep.
- If scheduling allows, try a **smaller** model or **fewer GPU layers** (only if supported by your build) to reach **graph_reserve** sooner — same crash may reproduce with shorter wall time.

## References

- **`CRASH_SYMBOLICATION_AND_COREDUMPS.md`**
- **`enable_coredump_ollama_vm.sh`**
- **`ERROR_TRACKING_STATUS.md`** — § Mar 22 plan execution
- **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`** — E3
