# cuCtxGetFlags fix applied — next steps

## What was done (local)

- **File:** `phase3/guest-shim/libvgpu_cuda.c`
- **Change:** Added `__attribute__((visibility("default")))` to the existing `cuCtxGetFlags` definition (around line 5683) so the symbol is exported from the shared library.
- **Status:** File builds clean locally (no linter errors). The function was already declared and listed in `stub_funcs`; only the visibility attribute was missing.

## Deploy to VM and test

1. **Copy updated source to VM**  
   Copy `phase3/guest-shim/libvgpu_cuda.c` to the VM (e.g. to the same path under your phase3 tree, e.g. `~/phase3/guest-shim/` or `/home/test-11/phase3/guest-shim/`).

2. **On the VM, rebuild the CUDA shim**
   ```bash
   cd ~/phase3   # or your phase3 path
   make guest
   ```
   Or without make:
   ```bash
   cd ~/phase3
   gcc -shared -fPIC -O2 -Wall -Wextra -std=c11 -D_GNU_SOURCE \
       -Iinclude -Iguest-shim \
       -o guest-shim/libvgpu-cuda.so.1 \
       guest-shim/libvgpu_cuda.c guest-shim/cuda_transport.c -ldl -lpthread
   ```

3. **Verify the symbol is exported**
   ```bash
   nm -D guest-shim/libvgpu-cuda.so.1 | grep ' T cuCtxGetFlags'
   ```
   Expected: `0000000000000000 T cuCtxGetFlags` (or similar address).

4. **Install the new shim and restart Ollama**
   - If you use `install.sh`: run it from the guest-shim dir (or as your setup expects).
   - Restart the Ollama service: `sudo systemctl restart ollama`.

5. **Check Ollama**
   - `sudo systemctl status ollama`
   - Run a short model request and confirm it uses the vGPU (e.g. `ollama run <model>` or your usual test).

## If you still see "undefined symbol: cuCtxGetFlags"

- Confirm the **same** `libvgpu_cuda.c` (with the visibility attribute) was copied and that the binary you built is the one installed (e.g. under `/opt/vgpu/lib/` or wherever Ollama’s `LD_PRELOAD` points).
- Re-run step 3 on the **installed** `.so`:  
  `nm -D /opt/vgpu/lib/libcuda.so.1 | grep cuCtxGetFlags`  
  (adjust path if your install uses a different name/location.)
