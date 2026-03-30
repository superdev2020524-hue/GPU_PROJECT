# Build `libggml-cuda.so` and deploy to the Phase 3 VM

**Audience:** Anyone repeating the Phase 1 / E2 fix (Ollama `compute=9.0` + Hopper kernels in GGML).

**Canonical technical detail (CUDA arch, Hopper):** **`BUILD_LIBGGML_CUDA_HOPPER.md`**

This document ties together **Phase3-specific source**, **build**, and **VM install** in one place.

---

## 1. Why two steps?

1. **`CMAKE_CUDA_ARCHITECTURES=90`** — GGML must contain **sm_90** SASS for H100 (see `BUILD_LIBGGML_CUDA_HOPPER.md`).
2. **Phase3 CC patch** — GGML reads `prop.major` / `prop.minor` after `cudaGetDeviceProperties`; with the vGPU shims, **ABI/layout skew** can make Ollama log **`compute=8.9`** and steer **wrong** kernel packages. The patch forces **`dev_ctx->major/minor = 9/0`** in `ggml_backend_cuda_reg()` (see **`TRACE_E2_COMPUTE_89_ROOT_CAUSE.md`**).

The patch lives in the repo as:

- **`patches/phase3_ggml_cuda_force_cc90.patch`**

Apply it to the **same Ollama tree** you use for the build (VM copy or fresh clone).

---

## 2. Option A — Docker build on the dev PC (recommended)

**Requires:** Docker (use `sudo docker` if your user is not in the `docker` group), network for image pull.

**Steps:**

```bash
cd /path/to/gpu/phase3

# 1) Ollama source (shallow clone is OK if patch applies cleanly)
git clone --depth 1 https://github.com/ollama/ollama.git ollama-src-phase3
cd ollama-src-phase3
patch -p1 < ../patches/phase3_ggml_cuda_force_cc90.patch
# If patch fails (upstream moved lines), apply the same two-line change manually:
#   dev_ctx->major = 9; dev_ctx->minor = 0;
# in ggml_backend_cuda_reg() after cudaGetDeviceProperties — see patch context.
cd ..

# 2) Build (writes ./out/libggml-cuda.so)
export OLLAMA_SRC="$PWD/ollama-src-phase3"
./build_libggml_cuda_hopper_docker.sh ./out

# 3) Deploy to VM (uses vm_config.py: test-4, etc.)
python3 deploy_libggml_cuda_hopper.py ./out/libggml-cuda.so
```

**Note:** If `patch` fails, your clone may differ from test-4’s tree. Either use the **VM’s** `ggml-cuda.cu` as the source of truth (copy file then build) or adjust the hunk to the current line numbers.

---

## 3. Option B — Sync patched `ggml-cuda.cu` from the VM, then Docker

When the VM already has the correct edit (e.g. `/home/test-4/ollama/ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu`):

1. Copy that file **over** `ollama-src-phase3/ml/backend/ggml/ggml/src/ggml-cuda/ggml-cuda.cu` in your clone **before** the Docker build (versions must match or the file may not compile).
2. Run **`build_libggml_cuda_hopper_docker.sh`** with `OLLAMA_SRC` pointing at that tree.

---

## 4. Option C — Native CUDA on a Linux host

See **`BUILD_LIBGGML_CUDA_HOPPER.md`** § “Option A”. After `make` / CMake, run **`deploy_libggml_cuda_hopper.py /path/to/libggml-cuda.so`**.

---

## 5. Verify on the VM (Checkpoints A / B / C)

```bash
# B: discovery should show compute=9.0 (not 8.9) after E2 fix + restart
journalctl -u ollama -b --no-pager | grep 'inference compute' | tail -3

# C: host mediator — module-load for 401312 (E1 may still need libcublasLt / host work)
# On dom0 (read-only for assistant): grep module-load /tmp/mediator.log | tail -10
```

---

## 6. Security

- **Do not** commit or paste **passwords** (VM, sudo, root) into this repo or into chat logs you archive.
- Use **`sudo`** / **`sshpass`** / env vars locally; rotate any password that was shared in plaintext.

---

## 7. Related PHASE3 docs

| Doc | Role |
|-----|------|
| `BUILD_LIBGGML_CUDA_HOPPER.md` | sm_90, CMake, verify |
| `TRACE_E2_COMPUTE_89_ROOT_CAUSE.md` | Why `8.9` appears |
| `deploy_libggml_cuda_hopper.py` | SCP + install + `systemctl restart ollama` |
| `build_libggml_cuda_hopper_docker.sh` | Containerized build |
| `SYSTEMATIC_ERROR_TRACKING_PLAN.md` | Checkpoints after deploy |
