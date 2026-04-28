# Clean VM setup: test-3@10.25.33.11

This document records the **exact setup performed on a fresh VM** (test-3) so the error can be traced and reproduced from a known state.

## VM config (vm_config.py)

- **VM_USER**: test-3  
- **VM_HOST**: 10.25.33.11  
- **VM_PASSWORD**: Calvin@123  
- **REMOTE_PHASE3**: /home/test-3/phase3  

Scripts `connect_vm.py` and `transfer_libvgpu_cuda.py` read these from `vm_config.py`.

## Steps performed on test-3

1. **Directories**
   - `/home/test-3/phase3/guest-shim`
   - `/home/test-3/phase3/include`
   - `/opt/vgpu/lib` (sudo)

2. **Build deps**
   - `sudo apt-get update && sudo apt-get install -y gcc make`

3. **Files copied**
   - `guest-shim/libvgpu_cuda.c` → `REMOTE_PHASE3/guest-shim/libvgpu_cuda.c`
   - `guest-shim/cuda_transport.c` → `REMOTE_PHASE3/guest-shim/cuda_transport.c`
   - `guest-shim/cuda_transport.h` → `REMOTE_PHASE3/guest-shim/cuda_transport.h`
   - `guest-shim/gpu_properties.h` → `REMOTE_PHASE3/guest-shim/gpu_properties.h`
   - `include/cuda_protocol.h` → `REMOTE_PHASE3/include/cuda_protocol.h`

4. **Build**
   ```bash
   cd /home/test-3/phase3 && gcc -shared -fPIC -O2 -std=c11 -D_GNU_SOURCE \
     -Iinclude -Iguest-shim -o /tmp/libvgpu-cuda.so.1 \
     guest-shim/libvgpu_cuda.c guest-shim/cuda_transport.c -ldl -lpthread
   ```

5. **Install shim**
   - `sudo cp /tmp/libvgpu-cuda.so.1 /opt/vgpu/lib/libcuda.so.1`

6. **Ollama (snap)**
   - Ollama was not installed; installed with `sudo snap install ollama`.
   - Service: `snap.ollama.listener.service` (no `snap.ollama.ollama.service`).
   - Override added: `/etc/systemd/system/snap.ollama.listener.service.d/vgpu.conf` with `Environment=LD_LIBRARY_PATH=/opt/vgpu/lib`.
   - Model pulled: `ollama pull llama3.2:1b`.

7. **Test**
   - With **ollama serve** started as: `LD_LIBRARY_PATH=/opt/vgpu/lib /snap/ollama/105/bin/ollama serve` (so the shim is loaded), **api/generate** for `llama3.2:1b` prompt "Hi" → **succeeds** (no "unexpectedly reached end of file").
   - The runner process inherits `LD_LIBRARY_PATH=/snap/ollama/105/lib/ollama:/opt/vgpu/lib`; the snap does not ship `libcuda.so.1` in its lib dir, so the runner loads our shim from `/opt/vgpu/lib`. The previous model-load error seen on earlier VMs **does not reproduce** on test-3 with the current shim code.

## How to re-run from your machine

```bash
cd /home/david/Downloads/gpu/phase3
# Use test-3 (already in vm_config.py)
python3 connect_vm.py "whoami; hostname"
python3 setup_new_vm.py   # full deploy + build + install (requires gcc on VM)
# Or only transfer and build shim:
python3 transfer_libvgpu_cuda.py
```

## File locations on test-3 (after setup)

| Path | Description |
|------|-------------|
| `/home/test-3/phase3/guest-shim/libvgpu_cuda.c` | Shim source |
| `/home/test-3/phase3/guest-shim/cuda_transport.c` | Transport source |
| `/opt/vgpu/lib/libcuda.so.1` | Built shim (used when LD_LIBRARY_PATH=/opt/vgpu/lib) |
| `/etc/systemd/system/snap.ollama.listener.service.d/vgpu.conf` | LD_LIBRARY_PATH override for ollama |

No other ad-hoc edits or legacy files from test-11 exist on test-3; all steps above are the only manipulations performed.
