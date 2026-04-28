# Constructor SEGV fix (wrapper + Ollama)

## What happened

With the **ollama wrapper** in place, the process crashed with **SEGV (status=11)** during library load. Two constructors were doing heavy work at load time:

1. **NVML** – constructor called `nvmlInit_v2()` / `nvmlDeviceGetCount_v2()` → fixed by making it log-only; init is lazy.
2. **CUDA** – after the NVML fix, SEGV still occurred right after `[libvgpu-cuda] constructor CALLED` and CUDART “Runtime API shim ready”. The CUDA constructor was doing nanosleep, getenv, dlsym, and `ensure_init()` (find_vgpu_device, etc.), which can be unsafe when the Go runtime isn’t fully up.

## Fixes applied (in repo)

- **`phase3/guest-shim/libvgpu_nvml.c`**  
  Constructor only logs one line and returns. NVML is initialized lazily on first `nvmlDeviceGetCount_v2()` / `nvmlInit_v2()`.

- **`phase3/guest-shim/libvgpu_cuda.c`**  
  Constructor only logs one line and returns. CUDA is initialized lazily in `ensure_init()` on first `cuInit` / `cuDeviceGetCount` (and CUDART constructor already triggers those).

## What you need to do on the VM

1. **Copy both updated sources to the VM** (from your host):
   ```bash
   scp phase3/guest-shim/libvgpu_nvml.c phase3/guest-shim/libvgpu_cuda.c test-11@10.25.33.111:~/phase3/guest-shim/
   ```

2. **On the VM**, rebuild guest shims and install **both** the NVML and CUDA shims:
   ```bash
   cd ~/phase3
   make guest
   sudo cp guest-shim/libvgpu-nvml.so /opt/vgpu/lib/libnvidia-ml.so.1
   sudo cp guest-shim/libvgpu-cuda.so.1 /opt/vgpu/lib/libcuda.so.1
   sudo systemctl restart ollama
   ```

3. **Check logs** (no SEGV; then check GPU mode):
   ```bash
   sleep 5
   sudo journalctl -u ollama -n 50 --no-pager | grep -E 'inference compute|total_vram|SEGV|segfault'
   ```

If the VM uses a different path than `~/phase3`, adjust the `cd` and `cp` paths. You must deploy **both** updated shims (NVML and CUDA) for the fix to take effect.
