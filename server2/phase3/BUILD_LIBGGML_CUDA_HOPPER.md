# Building libggml-cuda.so with Hopper (sm_90) Support

**Phase 3 (VM deploy + GGML CC=9.0 patch):** start with **`BUILD_AND_DEPLOY_LIBGGML_CUDA_PHASE3.md`** — single flow, patch file, and `run_libggml_docker_build_and_deploy.sh`.

**Quick reference:** On a machine with CUDA toolkit: `export CMAKE_CUDA_ARCHITECTURES=90 && cd ollama && make -j $(nproc)`. Then from phase3: `python3 deploy_libggml_cuda_hopper.py /path/to/ollama/build/lib/ollama/libggml-cuda.so`.

## What is sm_90?

**sm_90** is NVIDIA’s **compute capability** code for the **Hopper** architecture (e.g. H100). In CUDA/CMake:

- **Architecture 90** = Hopper = **compute capability 9.0**.
- The H100 GPU is Hopper, so it requires a `libggml-cuda.so` built with **sm_90** (or `CMAKE_CUDA_ARCHITECTURES=90`).
- Ollama’s default GGML CUDA build does **not** include 90 in its arch list (it has 50, 61, 70, 75, 80, 86, 89). So the bundled `.so` does not support H100; building with `CMAKE_CUDA_ARCHITECTURES=90` (and a CUDA toolkit that supports it, e.g. CUDA 11.8+ or 12.x) produces a library that **does** recognize and use the H100.

**Summary:** Build the `.so` on your local machine with CUDA 12.4 (or 11.8+) and `CMAKE_CUDA_ARCHITECTURES=90`, then copy it to the VM’s `/usr/local/lib/ollama/cuda_v12/` and restart Ollama. After that, GPU detection works (device_count=2, library=CUDA, H100 80GB).

## Why this is needed

On the test-3 vGPU guest, the bundled `libggml-cuda.so` (from Ollama's default build) was **not** compiled with support for compute capability 9.0 (Hopper / H100). That causes:

1. **Second-pass init validation** – Runners started with `GGML_CUDA_INIT=1` hit  
   `GGML_ASSERT(ggml_cuda_has_arch(info.devices[id].cc) && "ggml was not compiled with support for this arch")` and exit, so devices are filtered. We work around this by skipping CUDA init validation (`NeedsInitValidation` patch).

2. **Inference** – When the inference runner loads a real model, the same CUDA backend arch check runs and the process can abort, so `/api/generate` never completes.

To fix both, use or build a `libggml-cuda.so` that includes Hopper (sm_90).

---

## Option A: Build on a host with CUDA toolkit

Use a Linux (or Windows) machine where the NVIDIA CUDA toolkit is installed (including `nvcc`) and supports Hopper (e.g. CUDA 11.8+). The VM does not need nvcc; only the build host does.

### 1. Clone Ollama and deps

```bash
git clone https://github.com/ollama/ollama.git
cd ollama
```

### 2. Configure CMake with Hopper

Include architecture **90** (Hopper) in the CUDA architectures. You can target only Hopper or add it to the default set:

```bash
# Only Hopper (smaller build, H100/vGPU only)
export CMAKE_CUDA_ARCHITECTURES=90

# Or keep common archs and add 90 (e.g. for multi-GPU)
export CMAKE_CUDA_ARCHITECTURES="80;86;89;90"
```

Ollama's top-level `CMakeLists.txt` uses `CMAKE_CUDA_ARCHITECTURES` when set; the ggml-cuda `CMakeLists.txt` only sets defaults when it is **not** defined, so this env var is applied.

### 3. Build native deps (including libggml-cuda)

From the ollama repo root:

```bash
make -j $(nproc)
```

This runs CMake and builds the native LLM code. The exact output path depends on the Makefile; common locations are:

- `build/lib/ollama/libggml-cuda.so`, or  
- `dist/lib/ollama/...` if using a script like `scripts/build_linux.sh`.

If your tree uses a CMake preset:

```bash
cmake --preset linux
cmake --build build
```

Then look under the build directory for `lib/ollama/libggml-cuda.so` (see `CMakeLists.txt`: `OLLAMA_BUILD_DIR` = `${CMAKE_BINARY_DIR}/lib/ollama`).

### 4. Install on the VM

**Option 4a – Deploy script (from host where you have the built file)**

From the phase3 directory on a machine that has the built `libggml-cuda.so` and can reach the VM (vm_config.py target):

```bash
cd phase3
python3 deploy_libggml_cuda_hopper.py /path/to/build/lib/ollama/libggml-cuda.so
```

If you run it from phase3 and put the built library there as `libggml-cuda.so`:

```bash
python3 deploy_libggml_cuda_hopper.py
```

The script SCPs the file to the VM, backs up the existing library, installs the new one under `/usr/local/lib/ollama/cuda_v12/`, and restarts ollama.

**Option 4b – Manual SCP and SSH**

Copy the built library to the VM, then install and restart on the VM:

```bash
# On the build host (replace VM_USER and VM_HOST from vm_config.py)
scp build/lib/ollama/libggml-cuda.so VM_USER@VM_HOST:/tmp/libggml-cuda.so
```

On the VM:

```bash
# Backup existing library
sudo cp /usr/local/lib/ollama/cuda_v12/libggml-cuda.so /usr/local/lib/ollama/cuda_v12/libggml-cuda.so.bak

# Install new library
sudo cp /tmp/libggml-cuda.so /usr/local/lib/ollama/cuda_v12/libggml-cuda.so

# Restart Ollama
sudo systemctl restart ollama
```

If Ollama is installed under a different prefix (e.g. `/usr/lib/ollama` or a snap), copy `libggml-cuda.so` into the `cuda_v12` (or equivalent) directory used by that installation. You can set `OLLAMA_CUDA_DIR` in the deploy script if needed.

### 5. Re-enable CUDA init validation (optional)

After deploying a Hopper-capable `libggml-cuda.so`, you can re-enable second-pass init validation by reverting the `NeedsInitValidation` change in `ml/device.go` (restore `return d.Library == "ROCm" || d.Library == "CUDA"`), then rebuild the Ollama binary and restart the service.

---

## Option B: Build via Docker (no CUDA on host)

Use when the VM has no free disk for CUDA 12 or you prefer not to install the toolkit. Requires Docker (e.g. `sudo docker` or docker group). Run: `cd phase3 && ./build_libggml_cuda_hopper_docker.sh [OUTPUT_DIR]` (default output `./out/libggml-cuda.so`). Then: `python3 deploy_libggml_cuda_hopper.py out/libggml-cuda.so`. Uses image `nvidia/cuda:12.4.0-devel-ubuntu22.04`; first run can take a long time (image pull + build).

---

## Option C: Use Ollama's containerized build

Ollama's `scripts/build_linux.sh` can build Linux binaries (and libs) with CUDA. To add Hopper:

1. Set `CMAKE_CUDA_ARCHITECTURES` (e.g. `export CMAKE_CUDA_ARCHITECTURES="80;86;89;90"`) before or inside the script if it runs CMake.
2. Run the script and locate `libggml-cuda.so` in the produced `dist` (or equivalent) tree.
3. Copy that `libggml-cuda.so` to the VM as in Option A, step 4.

Check the script for how it invokes CMake and whether it respects `CMAKE_CUDA_ARCHITECTURES`.

---

## Option D: Prebuilt Ollama with Hopper (future)

If a future Ollama release or distro package is built with Hopper support in the default CUDA arch list, you can install that and use its `libggml-cuda.so` (or entire `cuda_v12` directory) on the VM instead of building from source.

---

## Verify

After replacing the library and restarting Ollama:

1. **Discovery** – `journalctl -u ollama` should show “inference compute” with `library=CUDA` and the H100 device.
2. **Init validation** – If you reverted the `NeedsInitValidation` patch, second-pass verification runners should no longer crash; discovery should still report the GPU.
3. **Inference** – A short request (e.g. `curl .../api/generate` or `ollama run llama3.2:1b 'Hi'`) should complete without timeout and use the GPU.

**Commands to run on the VM:**  
`ls -la /usr/local/lib/ollama/cuda_v12/libggml-cuda.so` (check timestamp);  
`sudo journalctl -u ollama -n 30 --no-pager | grep -E "inference compute|runner /info"` (expect library=CUDA);  
`curl -s -m 60 'http://127.0.0.1:11434/api/generate' -d '{"model":"llama3.2:1b","prompt":"Hi","stream":false,"options":{"num_predict":2}}' | jq -r '.response // .error'` (expect text, not timeout); then `ollama ps`.

---

## Documents before and after (PHASE3 flow)

**Before (why this is needed):**

- **VM_TEST3_GPU_MODE_STATUS.md** – Sections “Why verification runners exit” and “Inference and model load”: the bundled `libggml-cuda.so` was not built with Hopper (sm_90), so second-pass init validation and inference hit `ggml_cuda_has_arch` and aborted. Discovery was fixed by skipping CUDA init validation, but inference still needed a Hopper-capable library.
- **PHASE3_INFERENCE_ISSUES_AND_NEXT_STEPS.md** – D1: “Hopper (sm_90) missing in libggml-cuda.so”; fix is to build and deploy the Hopper lib.

**This document:** Build `libggml-cuda.so` with CUDA 12.4 (or host with CUDA 11.8+) and `CMAKE_CUDA_ARCHITECTURES=90`, then deploy to the VM with `deploy_libggml_cuda_hopper.py`.

**After (what to check once deployed):**

- **VM_TEST3_GPU_MODE_STATUS.md** – “Verification record”: `journalctl -u ollama` shows “runner /info response” device_count=2, “inference compute” library=CUDA, NVIDIA H100 80GB. “Inference verification”: server uses 80 GiB VRAM for context; runners report device_count=2. “Verify” section in this doc: discovery, optional re-enable of init validation, and a short inference test.

---

## VM reality check (test-4, Mar 2026)

See **`VM_LIBGGML_BUILD_VERIFIED.md`**: the VM has **Ollama source + Go** but **no `nvcc`**; Ubuntu’s **`nvidia-cuda-toolkit`** candidate is **11.5**, which is **not** the recommended baseline for Hopper in this doc. Prefer **Docker** (`build_libggml_cuda_hopper_docker.sh`) on a machine with Docker, or install **CUDA 12** from NVIDIA’s repo on the VM before native builds.

## References

- Ollama development: https://ollama.readthedocs.io/en/development/
- GGML CUDA arch list: `ml/backend/ggml/ggml/src/ggml-cuda/CMakeLists.txt` (defaults: 50, 61, 70, 75, 80, 86, 89 – no 90).
- Top-level `CMakeLists.txt`: can set `CMAKE_CUDA_ARCHITECTURES` to `"native"` if unset (CMake ≥ 3.24); passing `90` (or `"80;86;89;90"`) overrides that.
