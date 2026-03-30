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

## 4. Optional: run under GDB to catch next crash

To avoid waiting for a repeat 83‑minute run, you can run the **runner** under GDB only if you can start it in a debugger (e.g. manual run of the runner binary with the same env as the server uses). For a normal “serve” run, enabling coredumps (above) and reproducing once is usually simpler.

---

## 5. Summary

| Action | Purpose |
|--------|--------|
| Read `/tmp/vgpu_current_call.txt` while stuck | See which CUDA call is blocking (call_id and name). |
| Read last lines of `/tmp/vgpu_call_sequence.log` after crash | See last call sent (e.g. 0x0071 cuEventCreateWithFlags, 0x0030 cuMemAlloc_v2). |
| Enable coredumps (LimitCORE=infinity + daemon-reload + restart) | So the next crash produces a core. |
| `gdb /usr/local/bin/ollama.bin.new /path/to/core` then `bt full` | Get exact C/Go backtrace for the crash (rip 0x7f88670969fc). |

Next step after a backtrace: identify the failing line (e.g. in GGML event/stream handling or in the vGPU shim) and fix or add a host/guest sync or error path as needed.
