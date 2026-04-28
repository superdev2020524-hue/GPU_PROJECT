# Run transfer scripts from the mediator host (stable connection to VM)

## Why this fixes transfer failures

**Documented in CONNECT_VM_README.md:** Connection and transfer **succeeded earlier when scripts were run from a machine that could reach the VM**. The VM (e.g. test-4@10.25.33.12) is on a private network (10.25.33.x). If you run `connect_vm.py` or `transfer_cuda_transport.py` from a machine that has **no route** to that IP (e.g. Cursor cloud, or a PC on another network), you get:

- **SCP:** "timeout waiting for password" (no output from SSH)
- **connect_vm:** "Connection timed out" or "No route to host"

The **mediator host** (root@10.25.33.10) is on the same network as the VM. So running the transfer **from the mediator host** gives a stable path to the VM and avoids these failures.

See also: **VM_TEST3_GPU_MODE_STATUS.md** ("Deploy (from host)"), **SETUP_TEST4_CHECKLIST.md** (run from a machine that can SSH to the VM), **HOST_SETUP_BEGINNER_GUIDE.md** (phase3 on the host at `/root/phase3`).

---

## Solution: run transfer on the mediator host

### 1. Ensure phase3 is on the host

If phase3 is not already on the host (e.g. you only use the host for the mediator), copy it once from your local machine:

```bash
# From your local machine (where the gpu repo lives)
cd /path/to/gpu
scp -r phase3 root@10.25.33.10:/root/
```

If you already deploy mediator sources to the host, you may have `/root/phase3` with `src/`, `include/`, etc. Ensure the **guest-shim** files are there too (e.g. `guest-shim/cuda_transport.c`, `guest-shim/cuda_transport.h`, `guest-shim/libvgpu_cuda.c`). If not, copy the full phase3 tree as above.

### 2. On the host: stop Ollama on the VM (optional but recommended)

Stopping Ollama on the VM before transfer frees memory and reduces load (see TRANSFER_LIBVGPU_CUDA_SCRIPT_INVESTIGATION.md). From the host:

```bash
ssh root@10.25.33.10
# If you have connect_vm on the host, you can run:
# cd /root/phase3 && python3 connect_vm.py "echo Calvin@123 | sudo -S systemctl stop ollama"
# Or SSH from host to VM directly:
ssh -o StrictHostKeyChecking=no test-4@10.25.33.12 "echo Calvin@123 | sudo -S systemctl stop ollama"
```

(If you run `transfer_cuda_transport.py` on the host, the script will stop Ollama via connect_vm for you.)

### 3. On the host: run the transfer script

SSH to the mediator host, then run the transfer from there:

```bash
ssh root@10.25.33.10
cd /root/phase3
python3 transfer_cuda_transport.py
```

The script will use `connect_vm.py` and SCP **from the host** to the VM (test-4@10.25.33.12). Because the host can reach the VM, SCP and SSH should get the password prompt and complete.

### 4. Alternative: only copy transport files and build on VM

If you prefer not to run Python on the host, you can SCP the two files from the host to the VM and then build on the VM:

```bash
# On the mediator host (root@10.25.33.10)
cd /root/phase3
scp -o StrictHostKeyChecking=no guest-shim/cuda_transport.c guest-shim/cuda_transport.h test-4@10.25.33.12:/home/test-4/phase3/guest-shim/
ssh test-4@10.25.33.12 'cd /home/test-4/phase3 && gcc -shared -fPIC -O2 -std=c11 -D_GNU_SOURCE -Iinclude -Iguest-shim -o /tmp/libvgpu-cuda.so.1 guest-shim/libvgpu_cuda.c guest-shim/cuda_transport.c -ldl -lpthread && echo BUILD_OK'
ssh test-4@10.25.33.12 'echo Calvin@123 | sudo -S cp /tmp/libvgpu-cuda.so.1 /opt/vgpu/lib/libvgpu-cuda.so.1 && sudo systemctl start ollama'
```

(You will be prompted for the VM password unless you use sshpass or keys.)

---

## Summary

| Where you run the script | Result |
|--------------------------|--------|
| Machine with **no route** to 10.25.33.12 (e.g. Cursor backend, wrong network) | SCP/SSH timeout; transfer fails. |
| **Mediator host** (10.25.33.10) or any machine on same LAN/VPN as the VM | Stable connection; transfer and build succeed. |

**Stable transmission previously took place on the host** because the host could reach the VM. Run `transfer_cuda_transport.py` (or `deploy_to_test3.py`) from the mediator host when your current environment cannot reach the VM.
