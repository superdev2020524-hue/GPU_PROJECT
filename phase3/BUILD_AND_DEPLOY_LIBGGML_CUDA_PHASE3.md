# Build `libggml-cuda.so` and deploy to the Phase 3 VM

**Audience:** Anyone repeating the Phase 1 / E2 fix (Ollama `compute=9.0` + Hopper kernels in GGML).

**Canonical technical detail (CUDA arch, Hopper):** **`BUILD_LIBGGML_CUDA_HOPPER.md`**

This document ties together **Phase3-specific source**, **build**, and **VM install** in one place.

---

## 1. Why these build choices?

1. **`CMAKE_CUDA_ARCHITECTURES=90`** — GGML must contain **sm_90** SASS for H100 (see `BUILD_LIBGGML_CUDA_HOPPER.md`).
2. **`-DGGML_CUDA_GRAPHS=OFF`** and **`-DGGML_CUDA_FORCE_CUBLAS=ON`** in CMake (see **`build_libggml_cuda_hopper_docker.sh`**) — **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`** **E5** (`mmq_x_best=0` / `mmq.cuh` fatal) can still occur with **`GGML_CUDA_FORCE_CUBLAS=yes`** in **`ollama.service.d`** alone; compile-time **FORCE_CUBLAS** is the documented fallback when runtime env does not steer off the MMQ path.
3. **Phase3 CC patch** — GGML reads `prop.major` / `prop.minor` after `cudaGetDeviceProperties`; with the vGPU shims, **ABI/layout skew** can make Ollama log **`compute=8.9`** and steer **wrong** kernel packages. The patch forces **`dev_ctx->major/minor = 9/0`** in `ggml_backend_cuda_reg()` (see **`TRACE_E2_COMPUTE_89_ROOT_CAUSE.md`**).
4. **`GGML_CUDA_FA=OFF`** (**explicit** **`PHASE3_GGML_CUDA_FA=OFF`** on **`build_libggml_cuda_hopper_docker.sh`**; **default remains ON**) — **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`** **E6** mitigation hypothesis: native **`SIGFPE`** in **`launch_fattn`** / **`ggml_cuda_flash_attn_ext_mma_f16_case`** during **`ggml_backend_cuda_graph_reserve`**. Upstream Ollama **`CMakeLists.txt`** uses **`set(GGML_CUDA_FA ON)`**, which ignores **`-DGGML_CUDA_FA=OFF`** unless patched — apply **`patches/phase3_ollama_cmake_ggml_cuda_fa_overridable.patch`**. **VM-6 trial (May 2026):** **E6** / **SIGFPE** did not recur, but load failed with **`cublasGemmEx() … cublas_status=13`** (**`CUBLAS_STATUS_EXECUTION_FAILED`**), **`m=32000`**, and **`CUDA error: the function failed to launch on the GPU`** — recorded as **E7**; **do not** treat **FA=OFF** as a clean win until **E7** is resolved. **Restore** prior **`libggml-cuda.so`** if the guest regresses.
5. **`patches/phase3_ggml_fattn_launch_avoid_host_sigfpe.patch`** — **`fattn-common.cuh`** **`launch_fattn`** hardening for **Phase3 / vGPU**: (**a**) **`mb_eff`** when **`max_blocks=n_sm*max_blocks_per_sm`** is **0** (guest **`cudaDeviceProp.multiProcessorCount`** can report **0**) so stream‑k tile math never does **`idiv`** with a **zero** divisor; (**b**) **`blocks_per_wave`** floor at one SM’s worth when **`n_sm==0`**; (**c**) **`nbatch_fa==0`** guard on **`ntiles_KQ`**; (**d**) wave‑efficiency loop denominators guarded; (**e**) **`use_stream_k`** requires **`max_blocks>0`**. **`vm_gdb_attach_sigfpe.sh`** (with **`x/16i $pc`**) on **VM-6** showed **`SIGFPE`** at **`idiv`** with **zero** divisor inside **`launch_fattn<64,8,8>`** — consistent with **`n_sm==0`**. **May 2026:** **`libggml-cuda.so`** rebuilt with full patch (**v5** **`~199213624`** B) **removes** **`SIGFPE`** on bounded **`/api/generate`** but load then fails in **`ggml_cuda_op_mul_mat_cublas`** with **`cublasGemmEx`** **`cublas_status=7`** (**`CUBLAS_STATUS_INVALID_VALUE`**), **`m=5632`** — overlaps **E7**‑class **FA‑off** geometry; **global** **`std::max(1, nsm)`** alone was tried and produced **`CUDA_ERROR_INVALID_VALUE`** earlier (**skewed** launch geometry). **Guest rollback** to **FA‑on** baseline **`.so`** (**`…bak.pre_fattn_sigfpe_20260515`**, **`199193144`** B) for **E6** checkpoint **D** until **E7** / GGML **`mul_mat`** + executor alignment is solved; keep patch in tree for **operator‑approved** retests.
6. **`patches/phase3_ggml_cuda_init_nsm_fallback.patch`** — **`ggml_cuda_init()`**: if **`cudaGetDeviceProperties`** leaves **`multiProcessorCount` ≤ 0**, **`info.devices[id].nsm`** is forced to **132** (matches **`guest-shim/gpu_properties.h`** **`GPU_DEFAULT_SM_COUNT`**) with **`GGML_LOG_WARN`**. **Rationale:** **`launch_fattn`** reads **`ggml_cuda_info().devices[id].nsm`**; **0** yields host **SIGFPE** independent of **`fattn-common.cuh`** guards — **SYSTEMATIC_ERROR_TRACKING_PLAN.md** **Option B** “fix at source” for **E6** vs relying solely on **`launch_fattn`** math patches. Applied automatically by **`build_libggml_cuda_hopper_docker.sh`** when the patch file is present. **VM retest** after rebuild: bounded **`POST /api/generate`**; if the guest path still reports bogus **`multiProcessorCount`**, expect one **`GGML_LOG_WARN`** line at init.

The patch lives in the repo as:
- **`patches/phase3_ollama_cmake_ggml_cuda_fa_overridable.patch`** (CMake so **`-DGGML_CUDA_FA=OFF`** works)
- **`patches/phase3_ggml_fattn_launch_avoid_host_sigfpe.patch`** (**E6** host-math guard — optional but **on** by default in Docker build)
- **`patches/phase3_ggml_cuda_init_nsm_fallback.patch`** (**E6** **`nsm`** init fallback — **on** by default when present)

Apply it to the **same Ollama tree** you use for the build (VM copy or fresh clone).

### Bisect: `PHASE3_GGML_CUDA_FORCE_CUBLAS=OFF`

**When:** A Hopper build with **compile-time** **`-DGGML_CUDA_FORCE_CUBLAS=ON`** (script default) removes **`mmq_x_best` / `mmq.cuh`** (**E5**) but the runner still faults — e.g. **`SIGFPE`** after guest logs show **`cublasGemmEx() RETURN ok`** (**`ERROR_TRACKING_STATUS.md`**, **`CRASH_SYMBOLICATION_AND_COREDUMPS.md`**).

**Purpose:** Rebuild with **`FORCE_CUBLAS=OFF`** so the quantized path can exercise **MMQ** again. Compare outcomes:

- **E5 returns** → failure is tied to **cuBLAS-forced** matmul vs **MMQ** selection; record in **`ERROR_TRACKING_STATUS.md`**.
- **SIGFPE still** → fault is **not** specific to dropping E5; continue **native** trace (**`CRASH_SYMBOLICATION`** §4 / **`gdb`**).

**On the machine that runs Docker** (VM or dev PC), preserve the env through **`sudo`**, install the **`.so`**, **`systemctl restart ollama`**, then a **short** **`/api/generate`** (see **`PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md`** for client timeout vs load):

```bash
cd /path/to/phase3   # or ~/ggml-docker-build on the guest
sudo -E env PHASE3_GGML_CUDA_FORCE_CUBLAS=OFF ./build_libggml_cuda_hopper_docker.sh ./out-nocublas
sudo cp /usr/local/lib/ollama/cuda_v12/libggml-cuda.so{,.bak.$(date +%Y%m%d_%H%M%S)}
sudo cp ./out-nocublas/libggml-cuda.so /usr/local/lib/ollama/cuda_v12/libggml-cuda.so
sudo ldconfig
sudo systemctl restart ollama
```

---

## 2. Option A — Docker build on the dev PC (recommended)

**Requires:** Docker (use `sudo docker` if your user is not in the `docker` group), network for image pull.

### 2a. Dev PC has Docker

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

### 2b. Dev PC has no Docker — build on **VM-6** (or any Phase3 guest with Docker)

Use this when **`docker info`** fails on your laptop but the **guest** has Docker (often true for Phase3 images). The script pulls **`nvidia/cuda:12.4.0-devel-ubuntu22.04`** and runs the same containerized CMake build as **`build_libggml_cuda_hopper_docker.sh`**.

**On the VM (example: `test-6`):**

```bash
mkdir -p ~/ggml-docker-build/patches
# From your dev tree, copy these onto the VM (scp/rsync):
#   build_libggml_cuda_hopper_docker.sh  →  ~/ggml-docker-build/
#   patches/phase3_ggml_cuda_force_cc90.patch  →  ~/ggml-docker-build/patches/
cd ~/ggml-docker-build
chmod +x build_libggml_cuda_hopper_docker.sh
nohup bash -c 'echo YOUR_SUDO_PASSWORD | sudo -S ./build_libggml_cuda_hopper_docker.sh ./out' > build.log 2>&1 &
tail -f build.log
```

When **`Success. Library: …/out/libggml-cuda.so`** appears, install (still on VM):

```bash
sudo cp /usr/local/lib/ollama/cuda_v12/libggml-cuda.so{,.bak.$(date +%Y%m%d_%H%M%S)}
sudo cp ~/ggml-docker-build/out/libggml-cuda.so /usr/local/lib/ollama/cuda_v12/libggml-cuda.so
sudo systemctl restart ollama
```

**Or** from the dev PC, after `scp` of `out/libggml-cuda.so` back to the laptop: `python3 deploy_libggml_cuda_hopper.py …` as in §2a step 3.

**Note:** If `OLLAMA_SRC` is omitted, the script clones **`ollama`** into **`~/ggml-docker-build/ollama-src`** on the machine where you run it (first run downloads the Git repo).

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
