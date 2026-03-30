# Quick: Hopper libggml-cuda.so → VM (one screen)

**Prereq:** Linux with **Docker** (use `sudo` if your user is not in the `docker` group). Repo has **`phase3/ollama-src/`**.

```bash
cd /path/to/gpu/phase3

sudo env OLLAMA_SRC="$PWD/ollama-src" ./build_libggml_cuda_hopper_docker.sh ./out

python3 deploy_libggml_cuda_hopper.py ./out/libggml-cuda.so
```

Then on the VM: `journalctl -u ollama -f` and run a **long** `/api/generate`.

**If Docker is not an option:** install **CUDA 12.x toolkit** on a build machine (or VM with enough disk) so **`nvcc --version`** ≥ 11.8, then `export CMAKE_CUDA_ARCHITECTURES=90` and build Ollama native deps per **`BUILD_LIBGGML_CUDA_HOPPER.md`**.

See **`VM_LIBGGML_BUILD_VERIFIED.md`** for what was verified on test-4.
