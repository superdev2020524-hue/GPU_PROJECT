# Investigation: transfer_libvgpu_cuda.py and VM connection loss

**Issue:** After running `transfer_libvgpu_cuda.py`, the VM (TEST-4, 10.25.33.12) lost connectivity; SSH from the user's local PC no longer works.

This document describes exactly what the script does and what could cause the VM to become unresponsive.

---

## What the script does (step by step)

All commands are run **on the VM** via SSH using `connect_vm.py` (which uses `vm_config.py`: `test-4@10.25.33.12`, password `Calvin@123`).

### 1. Local preparation

- Reads **guest-shim/libvgpu_cuda.c** from the phase3 tree (~397 KB, ~9600 lines).
- Computes SHA-256 of the file.
- Base64-encodes the entire file (~530 KB of text).

### 2. Clear remote temp file

- **SSH command:** `rm -f /tmp/combined.b64`
- Effect: Removes any previous partial transfer on the VM.

### 3. Send file in chunks (many SSH sessions)

- **Chunk size:** 40,000 characters of base64 per chunk (~14 chunks for this file).
- **Per chunk:** One SSH session running:
  ```bash
  echo -n '<40000 chars of base64>' >> /tmp/combined.b64
  ```
- **Risks:**
  - Each invocation passes a **~40 KB command line** to the VM shell.
  - **14 sequential SSH connections** in a short time.
  - High memory or CPU on the VM could make SSH or the shell slow or unstable; unlikely by itself to kill the VM unless the VM is very constrained.

### 4. Decode and verify on VM

- **SSH command:** `base64 -d /tmp/combined.b64 > /tmp/libvgpu_cuda_new.c && wc -c /tmp/libvgpu_cuda_new.c`
- Effect: Writes ~397 KB to `/tmp/libvgpu_cuda_new.c`. Needs enough free space in `/tmp`.

- **SSH command:** `sha256sum /tmp/libvgpu_cuda_new.c | awk '{print "REMOTE_SHA256=" $1}'`
- Effect: Reads the file again; used only for verification.

### 5. Copy into phase3 tree

- **SSH command:** `cp /tmp/libvgpu_cuda_new.c <REMOTE_PHASE3>/guest-shim/libvgpu_cuda.c` (e.g. `/home/test-4/phase3/guest-shim/libvgpu_cuda.c`)
- Effect: Overwrites the guest-shim source on the VM.

### 6. Build on the VM (highest risk)

- **SSH command (single long run, timeout 300 s):**
  ```bash
  cd /home/test-4/phase3 && \
  gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
      -Iinclude -Iguest-shim \
      -o /tmp/libvgpu-cuda.so.1 \
      guest-shim/libvgpu_cuda.c guest-shim/cuda_transport.c \
      -ldl -lpthread
  ```
- **What this does:**
  - Compiles **libvgpu_cuda.c** (~9600 lines, very large single translation unit) and **cuda_transport.c** into one shared library.
- **Why this is risky on a small VM:**
  - **Memory:** GCC compiling a large file can use **hundreds of MB of RAM** (often 500 MB–1 GB+ peak). If the VM has limited RAM (e.g. 1–2 GB), the **OOM killer** may activate and kill processes (e.g. sshd, or the gcc run itself). That can make the VM unresponsive or break SSH.
  - **CPU:** Full optimization (-O2) on a large file can keep the CPU busy for a minute or more; combined with low memory, the system can thrash.
  - **Disk:** Build artifacts in `/tmp`; usually small, but /tmp full can cause odd failures.

### 7. Install and restart Ollama

- **SSH command:**
  ```bash
  echo <password> | sudo -S cp /tmp/libvgpu-cuda.so.1 /opt/vgpu/lib/libvgpu-cuda.so.1 && \
  echo <password> | sudo -S systemctl restart ollama
  ```
- Effect: Replaces the vGPU CUDA shim and restarts the Ollama service.
- **Risks:** Restarting ollama is normally safe. In theory a bug in the new shim could cause Ollama or a child to misbehave, but that would not typically take down the whole VM or SSH. So the **primary suspect for VM loss is step 6 (the gcc build)**.

---

## What actually happened on TEST-4 (from journal after VM restart)

After the user restarted TEST-4 and SSH worked again, the **previous boot’s** journal was checked. Findings:

1. **Kernel soft lockup (what made the VM unreachable)**  
   - **watchdog: BUG: soft lockup - CPU#2 stuck for 2608s! [kworker/2:2:86075]** (and again at 2634s).  
   - So **one CPU was stuck in kernel code for ~43 minutes**. The call trace points to:  
     `netstamp_clear` → `static_key_enable` → `jump_label_update` → `text_poke_bp_batch` → `smp_call_function_many_cond`.  
   - Once a CPU is stuck that long, the system becomes unresponsive (SSH, etc.). That matches “VM lost connection.”

2. **RCU / CPU starvation**  
   - Many repeated messages: **`rcu: Unless rcu_preempt kthread gets sufficient CPU time, OOM is now expected behavior.`** (from Mar 16 00:07 through 00:55).  
   - That indicates **very heavy CPU load**: RCU wasn’t getting enough CPU, so the kernel warned that OOM could follow. So the system was overloaded (e.g. gcc build + other work), then one CPU hit the soft lockup.

3. **No direct “OOM: Killed process gcc”** in the log. So the immediate cause of the outage was the **soft lockup**, not the OOM killer killing sshd. The load from the **gcc** build (and possibly the chunked SSH transfers) likely pushed the VM into a state where the kernel hit that lockup.

4. **VM specs (after restart):** 3.8 GiB RAM, 4 CPUs. So the VM is small; a large gcc build can still drive high CPU and memory use and contribute to lockups under load.

**Conclusion:** The VM became unreachable because of a **kernel soft lockup** (one CPU stuck for ~43 minutes), not because the OOM killer killed SSH. The **transfer script’s gcc build** (and possibly the many SSH chunk transfers) is the most plausible source of the load that led to that lockup. Recommendations below still apply: avoid building on the VM when possible, or give the VM more resources and run the build when the system is otherwise idle.

---

## Recommendations for a new VM (or to avoid recurrence)

1. **Give the VM more RAM** (e.g. at least 2–4 GB) so that a full `gcc` build of libvgpu_cuda.c is safe.
2. **Build the library elsewhere and copy the .so:**
   - On a machine with more RAM (or your PC), build the same command (gcc ... libvgpu_cuda.c cuda_transport.c -o libvgpu-cuda.so.1).
   - Copy only **libvgpu-cuda.so.1** to the VM (e.g. with scp) and run:
     ```bash
     sudo cp /path/to/libvgpu-cuda.so.1 /opt/vgpu/lib/libvgpu-cuda.so.1
     sudo systemctl restart ollama
     ```
   - This avoids running gcc on the VM at all. A small script could do: build locally → scp .so → ssh 'sudo cp ... && sudo systemctl restart ollama'.
3. **Optional: reduce gcc memory use on VM** (if you must build on the VM):
   - Add `-j1` (already serial) and consider `-O1` instead of `-O2` to reduce compiler memory use (at the cost of some performance of the shim). Not guaranteed to avoid OOM on very small VMs.
4. **Before running the transfer script:** Check VM RAM and free memory (`free -h`) and ensure `/tmp` has enough free space (`df /tmp`).

---

## Script location and dependencies

- **Script:** `phase3/transfer_libvgpu_cuda.py`
- **Uses:** `connect_vm.py`, `vm_config.py` (VM_USER, VM_HOST, VM_PASSWORD, REMOTE_PHASE3)
- **Source file transferred:** `phase3/guest-shim/libvgpu_cuda.c`
- **Also needs on VM:** `phase3/guest-shim/cuda_transport.c`, `phase3/include/`, headers (script does not transfer cuda_transport.c; it assumes the VM already has the full phase3 tree under REMOTE_PHASE3).
