# VM libggml-cuda (Hopper) — verified environment (Mar 19, 2026)

## What PHASE3 docs already say

- **`BUILD_LIBGGML_CUDA_HOPPER.md`**: Build `libggml-cuda.so` with **`CMAKE_CUDA_ARCHITECTURES=90`** on a machine that has the **CUDA toolkit (nvcc)** supporting Hopper (CUDA **11.8+** or **12.x**). The **guest VM does not need a GPU** to *compile* for `sm_90`; it needs **nvcc** (or build elsewhere).
- **`PHASE3_GPU_AND_TRANSPORT_STATUS.md`**: Alternative — **`build_libggml_cuda_hopper_docker.sh`** on a host with **Docker**, then **`deploy_libggml_cuda_hopper.py`** to the VM.

## Verified on test-4 (VM)

| Check | Result |
|--------|--------|
| **`/home/test-4/ollama`** | Present, **`CMakeLists.txt`** exists (Ollama tree ready for **Go** builds). |
| **`command -v nvcc`** | **Empty** — no CUDA compiler on PATH. |
| **`/usr/local/cuda`** | **Not present.** |
| **`apt` candidate `nvidia-cuda-toolkit`** | **11.5.1** (Jammy multiverse) — **too old** to rely on for **Hopper**; PHASE3 targets **11.8+ / 12.x**. |
| **Disk** | **~9.3 GiB** free on `/` — tight for a full **CUDA 12** toolkit install. |

**Conclusion:** The VM is prepared for **Go / Ollama binary** builds (`install_go_and_build_ollama_on_vm.py`), **not** for a **trusted Hopper `libggml-cuda.so`** build unless you install a **new enough** CUDA toolkit (e.g. **CUDA 12** from NVIDIA’s repo) or build **off-VM**.

## Fastest path that matches PHASE3 (recommended)

### A) Docker on a Linux machine **where you can run Docker** (workspace or other)

From **`phase3/`** (this repo includes **`ollama-src/`**):

```bash
cd phase3
# If docker requires sudo on your machine:
sudo env OLLAMA_SRC="$PWD/ollama-src" ./build_libggml_cuda_hopper_docker.sh ./out
python3 deploy_libggml_cuda_hopper.py ./out/libggml-cuda.so
```

Uses **`nvidia/cuda:12.4.0-devel-ubuntu22.04`** and **`CMAKE_CUDA_ARCHITECTURES=90`** per **`build_libggml_cuda_hopper_docker.sh`**.

### B) Install CUDA **12** toolkit on the VM (only if you accept size + repo setup)

Use NVIDIA’s **Ubuntu 22.04** CUDA repo (not only `nvidia-cuda-toolkit` 11.5 from multiverse). Follow NVIDIA’s current **cuda-downloads** instructions for **deb (network)** install, then:

```bash
export CMAKE_CUDA_ARCHITECTURES=90
cd /home/test-4/ollama
# Use Ollama’s documented native build (see BUILD_LIBGGML_CUDA_HOPPER.md)
make -j "$(nproc)"
# find build tree for libggml-cuda.so, then backup + copy:
sudo cp /usr/local/lib/ollama/cuda_v12/libggml-cuda.so /usr/local/lib/ollama/cuda_v12/libggml-cuda.so.bak.$(date +%s)
sudo cp path/to/built/libggml-cuda.so /usr/local/lib/ollama/cuda_v12/
sudo systemctl restart ollama
```

## Why this is still the right “next step”

Host logs showed **`cuModuleLoadFatBinary`** → **`CUDA_ERROR_INVALID_IMAGE`** for the **~401312-byte** GGML fatbin after **successful HtoD**. Even when the on-disk `.so` contains **`.target sm_90`** strings, a **fresh** build with **`CMAKE_CUDA_ARCHITECTURES=90`** and a **current** CUDA toolchain is the documented remediation path in **`BUILD_LIBGGML_CUDA_HOPPER.md`** and related notes.

## Related files

- **`BUILD_LIBGGML_CUDA_HOPPER.md`** — full rationale  
- **`build_libggml_cuda_hopper_docker.sh`** — Docker build  
- **`deploy_libggml_cuda_hopper.py`** — install on VM + restart Ollama  
- **`WORK_NOTE_MAR19_INVALID_IMAGE_SECOND_FATBINARY.md`** — failure analysis  
