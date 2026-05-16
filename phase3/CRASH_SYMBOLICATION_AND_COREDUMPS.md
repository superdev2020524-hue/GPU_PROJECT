# Runner crash symbolication and coredumps (Phase 3)

## Current crash finding (Mar 19)

- **VM journal:** Llama runner terminated with **exit status 2** after **83 minutes** in load.
- **Stack at crash:** goroutine 19 in `sync.WaitGroup.Wait` (llamarunner `runner.go:360`); **rip=0x7f88670969fc** (native code, likely GGML/CUDA .so).
- **Call sequence on VM:** `/tmp/vgpu_call_sequence.log` last entries included **0x0071** (cuEventCreateWithFlags) and **0x0030** (cuMemAlloc_v2). So the crash occurs during or after event/CUDA calls in the load path.
- **Host:** Had already completed HtoD (~1.6 GB), module-load rc=0, and post-module allocs for vm=9.

So the failure is **runner crash in C/native code** (rip in a loaded .so), not a transport timeout.

---

## 1. Stuck / current-call detection

When the guest is blocking in the transport poll loop, the **last call** being waited on is written to:

- **`/tmp/vgpu_current_call.txt`** — overwritten on each new request (call_id, name, seq, pid). Read this while the process is stuck to see which CUDA call is blocking.
- **`/tmp/vgpu_call_sequence.log`** — appended; last line is the last call sent before crash or hang.

After the transport change, 0x0071 is logged as **cuEventCreateWithFlags** (and other event/stream/module call names are present in the sequence log).

---

## 2. Enabling coredumps on the VM

To get a backtrace on the **next** runner crash:

1. **Systemd override for ollama** (run on VM as root or with sudo):

   ```bash
   sudo mkdir -p /etc/systemd/system/ollama.service.d
   echo -e '[Service]\nLimitCORE=infinity' | sudo tee /etc/systemd/system/ollama.service.d/coredump.conf
   sudo systemctl daemon-reload
   sudo systemctl restart ollama
   ```

2. **Core pattern** (optional; leave default if you only need one core):

   ```bash
   # List core location (often /var/lib/systemd/coredump or current dir)
   cat /proc/sys/kernel/core_pattern
   # To write core next to binary (for quick gdb):
   # echo '/tmp/core.%e.%p' | sudo tee /proc/sys/kernel/core_pattern
   ```

3. After the **next** crash, cores may be in:
   - `coredumpctl list` then `coredumpctl info <pid>` / `coredumpctl dump -o /tmp/core.ollama`
   - Or the path set in `core_pattern` (e.g. `/tmp/core.ollama.bin.new.<pid>`).

### 2.1 Phase 3 / Ubuntu 22.04 findings (**VM-6**, **E3** / **`SIGFPE`**, **2026-05**)

Applied in order, with **reproducible** **`SIGFPE: floating-point exception`** immediately after **`cublasGemmEx() RETURN ok`** (runner child PID in shim logs, parent **`ollama.bin[serve]`** logs the `SIGFPE` line):

| Step | Result |
|------|--------|
| **`LimitCORE=infinity`** on **`ollama.service`** (`coredump.conf` drop-in) | **`systemctl show`** reports **`LimitCORE=infinity`**; **`Max core file size`** **unlimited** in **`/proc/<mainpid>/limits`**. |
| **`systemd-coredump`** installed; **`kernel.core_pattern`** → **`|/usr/lib/systemd/systemd-coredump …`** | **`coredumpctl list`** **empty** after reproduce; **`journalctl -u systemd-coredump`** **no entries**. |
| **`/etc/systemd/coredump.conf.d/phase3-large-cores.conf`**: **`Storage=external`**, **`ProcessSizeMax=16G`**, **`ExternalSizeMax=16G`**, **`Compress=no`** | Still **no** stored cores under **`/var/lib/systemd/coredump`** (empty dir). |
| **`kernel.core_pattern=/tmp/core.%e.%p.%t`** (plain file, sysctl drop-in) | Still **no** **`/tmp/core.*`** after reproduce — kernel **did not** write a core file for this crash class (not only a size limit issue). |
| **`GOTRACEBACK=all`** in **`ollama.service.d/gotraceback.conf`** | **No** **`goroutine`** / **`runtime.`** lines in **`journalctl -u ollama`** — consistent with fault in **C/CUDA** (GGML/cuBLAS path) after Gemm returns, not a Go panic with traceback. |

**Interpretation for the registry:** treat **native backtrace** as **still missing**; next actions remain **`gdb`** on the **runner** (§4), **`libSegFault.so`**-style preload (if acceptable), or a **controlled build swap** (e.g. **with/without** **`-DGGML_CUDA_FORCE_CUBLAS=ON`**) to bisect **E5** vs **SIGFPE** per **`BUILD_AND_DEPLOY_LIBGGML_CUDA_PHASE3.md`**.

---

## 3. Getting a backtrace from a core

On the VM (with the same binary and libs that were running):

```bash
# If core is in coredumpctl:
coredumpctl list
coredumpctl dump -o /tmp/core.ollama   # latest matching process

# Load core in gdb (adjust paths if ollama is elsewhere)
gdb /usr/local/bin/ollama.bin.new /tmp/core.ollama

# In GDB:
(gdb) bt full          # full backtrace
(gdb) info sharedlibrary   # see which .so contains 0x7f88670969fc
(gdb) list             # if symbols are present
```

If the crash is in a shared library (e.g. libllama.so, libvgpu_cuda.so), the backtrace will show the C/CGo frame that called into it. Use that to see whether the crash is in GGML load, a CUDA shim, or the transport.

---

## 4. GDB: catch **SIGFPE** in the **runner** child (VM-6 / E3, **2026-05**)

Coredumps did not appear for this fault class (§2.1). A one-shot **interactive** session under **`gdb`** can still stop on **`SIGFPE`** if the debugger **follows the child** after **`ollama.bin`** forks the CUDA runner.

**1. Match the service environment** (from `systemctl show ollama -p Environment`); at minimum:

```bash
export LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama
export OLLAMA_NUM_GPU=1 OLLAMA_LLM_LIBRARY=cuda_v12
export OLLAMA_LIBRARY_PATH="$LD_LIBRARY_PATH"
export OLLAMA_LOAD_TIMEOUT=4h CUDA_TRANSPORT_TIMEOUT_SEC=14400
export GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_DISABLE_GRAPH_RESERVE=1
export GGML_CUDA_DISABLE_BATCHED_CUBLAS=1 GGML_CUDA_FORCE_CUBLAS=yes
```

**2. Stop the unit**, run **`serve`** under **`gdb`** as **root** (same **file caps** / user model as production if you mirror `User=` / **`AmbientCapabilities`** from the unit):

```bash
sudo systemctl stop ollama
sudo gdb -q --args /usr/local/bin/ollama.bin serve
```

**3. In GDB** (repro → **tinyllama** load from a second shell: `curl` **`/api/generate`** as usual):

```text
set pagination off
set confirm off
set startup-with-shell off
set follow-fork-mode child
set detach-on-fork off
catch signal SIGFPE
run
# After SIGFPE:
thread apply all bt full
```

**GDB / Go:** **`set startup-with-shell off`** avoids immediate exit during startup. **`vfork`:** **`follow-fork-mode child`** may attach to the short-lived **`runner --ollama-engine`** child first; use **`info inferiors`** / switch inferiors, or **`set follow-fork-mode parent`** + **`set detach-on-fork on`** if you must keep **`serve`** attached (**SIGFPE** still usually fires in a **different** runner PID — **`attach`** to the PID printed in **`[libvgpu-cublas] … pid=…`** may be required).

**`LD_PRELOAD`** handlers for **`SIGFPE`** are typically **ignored** when **`ollama.bin`** has **`AT_SECURE`** (file caps); do not rely on preload for the capped service binary.

**4. Restore service** when done: **`sudo systemctl start ollama`**.

**Caution:** Changing **`fs.suid_dumpable`** globally to force cores from **file-capped** binaries is a **host-wide** security knob; prefer **GDB** (above) unless you explicitly accept that tradeoff.

### 4a. **`strace -f`** timing observation (**Heisenbug class**, VM-6, **2026-05**)

In one probe, **`sudo strace -f`** around **`ollama.bin serve`** (same library env as **`sudo -u ollama`**) allowed **`POST /api/generate`** (**`tinyllama`**) to complete with **HTTP 200** (~**2.2 s**), while the normal **`systemd`** path remained **HTTP 500** + **`SIGFPE`** right after **`cublasGemmEx() RETURN ok`**. Treat this as **timing/order-sensitive** native behavior, not a production wrapper. **`CUDA_LAUNCH_BLOCKING=1`** on the unit **did not** remove **`SIGFPE`** in a follow-up A/B. Remove experimental **`systemd` drop-ins** after tests; restore the service with **`systemctl daemon-reload`** + **`restart`**.

### 4b. Automated attach script (repo **`vm_gdb_attach_sigfpe.sh`**)

On the VM (after **`scp`** or **`deploy_to_test3.scp_file`** to **`/tmp/`**), with **`ollama` active:

```bash
chmod +x /tmp/vm_gdb_attach_sigfpe.sh
bash /tmp/vm_gdb_attach_sigfpe.sh /tmp/gdb_attach_sigfpe.log /tmp/gdb_attach_gen.json
```

The script starts a bounded **`curl`** **`/api/generate`**, polls for **`ollama.bin runner`** whose **`cmdline`** does **not** contain **`ollama-engine`**, then runs **`gdb -batch -p …`** with **`catch signal SIGFPE`**, **`x/16i $pc`**, sample **`info registers`**, and **`thread apply all bt full`**. Use **LF** line endings only (not CRLF).

**Result class (VM-6, May 2026):** **`SIGFPE`** at **`launch_fattn<64,8,8>(…)`** in **`libggml-cuda.so`**, called from **`ggml_cuda_flash_attn_ext_mma_f16_case`** → **`evaluate_and_capture_cuda_graph`** → **`ggml_backend_cuda_graph_reserve`** → **`llama_init_from_model`** / **`llama-context.cpp`**, even when service env sets **`GGML_CUDA_DISABLE_GRAPHS=1`** (graph-**reserve** path still runs).

**Related trial:** A **`GGML_CUDA_FA=OFF`** **`libggml-cuda.so`** rebuild (**CMake** patch **`phase3/patches/phase3_ollama_cmake_ggml_cuda_fa_overridable.patch`**) did **not** show this **`SIGFPE`** in a short generate on **VM-6**, but failed with wide **`cublasGemmEx`** **`cublas_status=13`** — registry **E7** (**`SYSTEMATIC_ERROR_TRACKING_PLAN.md`**); guest was rolled back to **FA-on** **`.so`**.

---

## 5. Summary

| Action | Purpose |
|--------|--------|
| Read `/tmp/vgpu_current_call.txt` while stuck | See which CUDA call is blocking (call_id and name). |
| Read last lines of `/tmp/vgpu_call_sequence.log` after crash | See last call sent (e.g. 0x0071 cuEventCreateWithFlags, 0x0030 cuMemAlloc_v2). |
| Enable coredumps (LimitCORE=infinity + daemon-reload + restart) | So the next crash produces a core. |
| `gdb /usr/local/bin/ollama.bin.new /path/to/core` then `bt full` | Get exact C/Go backtrace for the crash (rip 0x7f88670969fc). |
| `bash vm_gdb_attach_sigfpe.sh` (**§4b**) | Capture **`thread apply all bt full`** at **`SIGFPE`** (attach to non-**ollama-engine** runner). |

Next step after a backtrace: identify the failing line (e.g. in GGML event/stream handling or in the vGPU shim) and fix or add a host/guest sync or error path as needed.
