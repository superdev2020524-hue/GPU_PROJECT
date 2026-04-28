# VM GPU mode status and next steps

## Verified on VM (Mar 2)

**Ollama is currently in CPU mode.**

Evidence from `journalctl -u ollama`:

```
time=... source=types.go:60 msg="inference compute" id=cpu library=cpu compute="" name=cpu ...
time=... source=routes.go:1768 msg="vram-based default context" total_vram="0 B" default_num_ctx=4096
```

- **id=cpu**, **library=cpu**, **total_vram="0 B"**, **pci_id=""** → discovery did not find a GPU, so Ollama chose the CPU backend.

Our shim **does** run in the main process (we see `cuInit`, "GPU defaults applied (H100 80GB CC=9.0 VRAM=81920 MB)", "device found at 0000:00:05.0"). So the process has the CUDA shim, but the **discovery path** that feeds `types.go` (inference compute) is not getting device count / VRAM from our shim—or discovery runs in a context where our shim is not used (e.g. different library or child process).

## Goal

Get **id=gpu** (or equivalent), **library=cuda** (or cuda_v12), and **total_vram > 0** so Ollama selects **GPU** for inference and uses the vGPU remoting path.

## Tried on VM

- **NVML in LD_PRELOAD:** `vgpu.conf` has `LD_PRELOAD=.../libnvidia-ml.so.1:.../libcuda.so.1:.../libcudart.so.12`. Discovery still reported **id=cpu**.
- **Symlink cuda_v12 and parent ollama dir (Mar 2):** Replaced real libcuda.so.1, libcudart.so.12, libnvidia-ml.so.1 in both `/usr/local/lib/ollama/cuda_v12/` and `/usr/local/lib/ollama/` with symlinks to `/opt/vgpu/lib/` shims, so the runner loads our shims regardless of path order. **OLLAMA_NUM_GPU=999** added to vgpu.conf. After restart, discovery still reports **id=cpu**, **total_vram="0 B"**. The runner may be started with an env that does not use these paths, or discovery may not go through these libs.
- **CUDART no-op constructor (Mar 2):** `libvgpu_cudart.c` constructor was made a no-op (no cuInit/cudaGetDeviceCount during library load) to avoid SEGV when Go runtime is not ready. Deployed to VM and rebuilt; `/opt/vgpu/lib/libcudart.so.12` updated.
- **Wrapper script for runner env:** A shell wrapper at `/usr/local/bin/ollama` that sets `LD_PRELOAD`/`LD_LIBRARY_PATH` and `exec`s `ollama.real` was tried so that when the main process spawns `ollama runner`, the runner would get the same env. With the wrapper installed, **Ollama crashes with SEGV** immediately (with old CUDART: after "Runtime API shim ready"; with no-op CUDART: no shim logs). Running the same binary with the same env under GDB does **not** reproduce the crash, so the failure may be timing- or startup-order related when launched via the wrapper (bash → exec). **Original binary restored**; service runs again in CPU mode.
- **C wrapper (no shell):** A minimal C program (`phase3/ollama_vgpu_wrapper.c`) that sets vGPU env and `execve()`s `ollama.real` was built and installed as `/usr/local/bin/ollama` on the VM. **No SEGV** — Ollama starts and stays up. The main process (ollama.real) has `LD_PRELOAD` and `OLLAMA_NUM_GPU=999`. Logs still show **id=cpu**, **total_vram="0 B"**. The runner is spawned as `ollama.real runner` (Ollama resolves the real binary path), so the runner does not go through the wrapper; it should still inherit the parent's environment. **Current VM state:** C wrapper is installed; discovery still reports CPU.
- **Process env check:** The main Ollama process (started by systemd with vgpu.conf) has `LD_PRELOAD` and `OLLAMA_NUM_GPU=999` in its environment. GPU discovery and the "inference compute" result are driven by the **runner** subprocess; if the runner is spawned with a sanitized env (e.g. Go clears or does not pass `LD_PRELOAD`), it would not load our shims and would report CPU.

## Next steps (to get GPU mode)

1. **Trace discovery**
   - On the VM, run Ollama under `strace -f -e openat,open,read -o /tmp/ollama_discovery.log` (or similar) and trigger discovery; inspect which libraries and files are opened/read so you can see which path is used and why it might not call our shims.
2. **Ensure discovery calls our shim**
   - Confirm which API Ollama uses for “inference compute” (CUDA vs NVML vs both) and that the process doing discovery has our shims in the load order (LD_PRELOAD is set; if discovery runs in a child, ensure it inherits the same env).

2. **Confirm which process runs discovery**
   - Logs show "discovering available GPUs..." in the main process; the next line is "inference compute id=cpu".
   - Check whether discovery is done in the same process (then our preloaded libcuda/libcudart should be used) or in a child (then the child must inherit **LD_PRELOAD** and **LD_LIBRARY_PATH** so it loads our shims).

3. **Match what Ollama expects**
   - From past docs, discovery may require **PCI bus ID** and **VRAM** to match. Ensure:
     - Our NVML/CUDA shims report a non-empty **pci_id** (e.g. `0000:00:05.0` or the format Ollama expects).
     - **cuMemGetInfo_v2** / NVML memory info return **total > 0** (we already return 80 GB in the shim; confirm this is the path used by discovery).

4. **Re-check after changes**
   - Restart Ollama, then:
     - `journalctl -u ollama -n 80 --no-pager | grep -iE 'inference compute|total_vram|library=|discovering'`
   - Look for **id=gpu**, **library=cuda** (or cuda_v12), and **total_vram** non-zero.

5. **If logs still say CPU, confirm actual GPU usage**
   - Run a model on the VM (`ollama run llama3.2:1b 'hello'`) and on the host run `nvidia-smi` and `./vgpu-admin show-metrics`. If GPU memory or job count increases during the run, inference may be using the vGPU even though discovery logged CPU. See **GPU_MODE_FIX_SYMLINK_CUDA_V12.md** for full symlink steps and debugging.

## Host verification (for you)

See **HOST_VERIFY_GPU_MODE.md** for commands to run on the host to confirm whether inference is using the physical GPU (nvidia-smi, mediator/CUDA executor logs).
