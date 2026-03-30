# Verification: role, authority, environments, Phase 1 (Mar 20, 2026)

## 1. Assistant role and authority (authoritative)

| Scope | Source | Allowed | Not allowed (this agent) |
|-------|--------|---------|---------------------------|
| **Host (dom0, e.g. xcp-ng / mediator machine)** | `ASSISTANT_PERMISSIONS.md` | Read logs, read file contents, run **non-destructive diagnostics** (e.g. `nvidia-smi`, `cuobjdump` on existing dumps) | Edit host files, copy to host, `make`/build, restart mediator |
| **VM (test-4)** | same | Full: commands, deploy, edit, rebuild services, restart `ollama` | — |

**Anti-coupling / triage:** `ASSISTANT_ROLE_AND_ANTICOUPLING.md` — VM+host health first; PHASE3 history first; then escalation.

---

## 2. Environment verification (this session)

### Host (dom0) — read-only scope

- **Hostname:** `xcp-ng-syovfxoz`
- **`cuda.h`:** present — `/usr/local/cuda/include/cuda.h`
- **`nvcc`:** `/usr/local/cuda/bin/nvcc`, **CUDA 12.3.52**
- **GPU:** `nvidia-smi` wrapper on host may use `-i 0` alias; use bare `nvidia-smi` if diagnosing
- **Mediator log:** shows prior Phase 1 signature: `dumped /tmp/fail401312.bin`, `CUDA_ERROR_INVALID_IMAGE` on `0x0042`

### VM (test-4) — full authority scope

- **Hostname:** `test4-HVM-domU`
- **`ollama`:** `active`
- **`nvcc`:** **not in PATH**; **`cuda.h`:** not under `/usr/local/cuda` or `/usr/include` (no dev toolkit for native CUDA builds)
- **Docker:** **not installed**
- **Go:** `/usr/local/go/bin/go` → **go1.26.1**
- **`libggml-cuda.so`:** `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so` (~179M, dated Mar 19 2026)
- **Strings spot-check:** many `sm_90`-related strings exist in the `.so`, but **individual fatbin modules can still be sm_80-only** (see §3).

### Workspace (David’s machine, `phase3/`)

- **Docker:** installed but **`docker run` fails with permission denied** on the daemon socket; **`sudo docker` requires a TTY password** — automated Hopper Docker build **not runnable by the agent** without you fixing Docker permissions or running the script manually.

---

## 3. Phase 1 milestone — status

**Phase 1 goal:** Prove guest→host transport is sound and **isolate** the failure of the second `cuModuleLoadFatBinary` (401312 bytes).

| Criterion | Result |
|-----------|--------|
| Transport + smaller module | **28120-byte** load → **CUDA_SUCCESS** |
| Large module | **401312-byte** load → **INVALID_IMAGE** (mediator); standalone test → **NO_BINARY_FOR_GPU (209)** |
| Not “mediator-only” | Standalone `test_fatbin_load` fails on same bytes → **not** explained by mediator glue alone |
| **Root cause (definitive)** | Host **`cuobjdump -elf /tmp/fail401312.bin`**: **`arch = sm_80`** (Ampere), kernels named `ampere_h16816gemm_*`. **H100 is Hopper (sm_90).** This fatbin slice has **no runnable image for the device**. |

**Conclusion:** Phase 1 is **achieved**. The blocker is **wrong embedded GPU architecture for this kernel pack** (sm_80 blob on sm_90 hardware), not “Ollama cannot run on H100” in principle.

---

## 4. “cuda.h not found” (Ollama / forum context)

- **Meaning:** compile-time missing **CUDA Toolkit headers** or wrong `-I`.
- **VM today:** no `cuda.h` / `nvcc` → to **build** `libggml-cuda.so` **on the VM**, you’d install CUDA 12 toolkit (or use Docker on a machine that can run it).
- **Host (dom0):** headers and `nvcc` **are** present — useful for a **human** building on dom0; **this agent still must not** build/install on dom0 per permissions.

---

## 5. Proposed next steps (in order)

1. **Build** `libggml-cuda.so` with **Hopper-native SASS and/or forward-compatible PTX** for **all** GGML CUDA compilation units — not only global `sm_90` strings in the `.so`. Use e.g. `CMAKE_CUDA_ARCHITECTURES=90` or `"80;90"` as needed so **GEMM paths** are not sm_80-only.
   - **Option A:** On your workstation: fix Docker (`sudo usermod -aG docker $USER` + re-login, or run `build_libggml_cuda_hopper_docker.sh` with `sudo`).
   - **Option B:** On **dom0** (you): clone Ollama, `export CMAKE_CUDA_ARCHITECTURES=90` (or include 90 in a multi-arch list), `make -j`, copy `libggml-cuda.so` to VM (see `BUILD_LIBGGML_CUDA_HOPPER.md`).
2. **Deploy** to VM: `python3 deploy_libggml_cuda_hopper.py /path/to/libggml-cuda.so` (from `phase3/`).
3. **Re-verify:** tiny model generate; host log should show `0x0042` **success** for the former 401312 path; optional re-dump and `cuobjdump` should show **sm_90** (or PTX) for that module.

---

## 6. References in repo

- `BUILD_LIBGGML_CUDA_HOPPER.md` — why sm_90 and how to build/deploy  
- `deploy_libggml_cuda_hopper.py` — install under `/usr/local/lib/ollama/cuda_v12/`  
- `build_libggml_cuda_hopper_docker.sh` — Docker build when toolkit not local  
- `ASSISTANT_PERMISSIONS.md` — host vs VM boundaries  
