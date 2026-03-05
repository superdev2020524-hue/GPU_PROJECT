# Phase3 deploy and Ollama GPU mode (test-3 only)

**Target VM:** test-3@10.25.33.11 (see `vm_config.py`). test-11 is no longer used.

## Correct transfer and build method

- **Use SCP for all file transfers.** Do not use chunked base64 transfer for the guest shim sources; it can corrupt large files (e.g. `libvgpu_cuda.c`) and cause follow-on errors that look like code bugs.
- **Shims are always built on the VM.** The host may not have a GCC environment. Never copy host-built .so files to the VM. Deploy source only; build with `make guest` on the VM so the .so files are produced in the VM’s environment.
- **Preferred:** run `python3 deploy_to_test3.py`. It:
  - Copies the full phase3 **source tree** to the VM via **scp -r**
  - Runs **`make guest` on the VM** (builds .so files there)
  - Installs the **VM-built** shims from `~/phase3/guest-shim/` to `/opt/vgpu/lib` with correct symlinks (libcuda.so.1, libcudart.so.12, libnvidia-ml.so.1, and CUBLAS stubs if built)
  - Ensures Ollama systemd override has `LD_LIBRARY_PATH=/opt/vgpu/lib` and `OLLAMA_NUM_GPU=1`
  - Restarts the Ollama service

## After deploy

1. Check GPU discovery:  
   `python3 connect_vm.py "journalctl -u ollama -n 50 --no-pager | grep -E 'library=|total_vram|inference compute'"`

2. Run a short inference:  
   `python3 connect_vm.py "ollama run llama3.2:1b 'Hi'"`

## Service runs ollama.bin directly (full LD_PRELOAD)

To avoid SEGV when the bash script loads the CUDA shim, the service runs **`ollama.bin`** directly. The drop-in `ollama.service.d/vgpu.conf` is generated from `phase3/ollama.service.d_vgpu.conf` (SCP’d and installed by the deploy script). It clears `ExecStart` and sets `ExecStart=/usr/local/bin/ollama.bin serve` plus full `LD_PRELOAD` and `LD_LIBRARY_PATH`.

## If you change only one file

For a single-file update (e.g. `libvgpu_cuda.c`), use **scp** from your host to copy **source** to the VM (do not copy host-built .so; build on the VM):

```bash
sshpass -p 'Calvin@123' scp -o StrictHostKeyChecking=no \
  phase3/guest-shim/libvgpu_cuda.c \
  test-3@10.25.33.11:/home/test-3/phase3/guest-shim/
```

Then on the VM (or via `connect_vm.py`):  
`cd ~/phase3 && make guest` and re-run the install steps (copy the **VM-built** .so from `~/phase3/guest-shim/` to `/opt/vgpu/lib`, restart ollama), or run `deploy_to_test3.py` again.

## Requirements

- **sshpass** (optional): If installed, deploy uses it for non-interactive scp/ssh. Otherwise the script uses **pexpect** for scp and **connect_vm.py** for ssh.
- VM must be reachable: `python3 connect_vm.py "echo ok"` should succeed before running deploy.

## Avoiding past mistakes

1. **Transfer:** Always use the script that uses SCP (`deploy_to_test3.py`) or explicit `scp`; avoid base64/chunked transfer for shim sources so code is not lost or corrupted.
2. **Progress:** When fixing an error, keep the fix and move forward; do not revert to the starting point or conclude "this error is not correct" and undo the change without verifying the real cause.
