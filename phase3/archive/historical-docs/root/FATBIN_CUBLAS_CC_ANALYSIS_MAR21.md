# Analysis: 401312 fatbin failure, libggml-cuda rebuild, and compute capability (Mar 21, 2026)

## Executive summary

1. **Phase 1 (transport + isolation)** is satisfied: the **401312-byte** `cuModuleLoadFatBinary` payload is **not** corrupted in transit; standalone host load fails; **`cuobjdump`** shows **`arch = sm_80`** inside the blob (`ampere_h16816gemm_*` symbols) — **Ampere/cuBLASLt-style** kernels, not GGML’s own CUDA TU set.
2. **Hopper-only `libggml-cuda.so`** was built locally (**`CMAKE_CUDA_ARCHITECTURES=90`**, ~250× **`sm_90`** ELF sections, **no `sm_80`** in that `.so`) and **deployed** to the VM (**MD5 match** with local `out/libggml-cuda.so`).
3. **After deploy**, mediator logs still show **`INVALID_IMAGE`** on the **same 401312** dump, and **`cuobjdump`** on **`/tmp/fail401312.bin`** is still **`sm_80`**. Therefore that fatbin is **not** coming from the rebuilt **GGML** CUDA library’s compiled kernels.
4. **Strong hypothesis:** the failing module is selected from **`libcublasLt.so.12`** (bundled under **`/usr/local/lib/ollama/cuda_v12/`** on the guest). Kernel choice follows **advertised device compute capability** and internal heuristics; wrong CC → wrong embedded fatbin → **H100 cannot run sm_80-only image** → **`INVALID_IMAGE` / `NO_BINARY_FOR_GPU`**.
5. **Ollama journal** showed **`compute=8.9`** for the H100 path while **`guest-shim/gpu_properties.h`** intentionally set **`GPU_DEFAULT_CC_MAJOR/MINOR` to `8`/`9`** (“Ada compatibility”). That matches **mis-routed** BLASLt selection. **Fix:** advertise **true Hopper `9.0`** and **force** that CC after host GPU-info overlay so live host structs cannot leave **8.9** in place.

## Evidence table

| Check | Result |
|--------|--------|
| `libggml-cuda.so` (Docker build, arch=90) | 250× `arch = sm_90` in `cuobjdump -elf` |
| VM deployed `.so` | Same size + **MD5** as local `out/libggml-cuda.so` |
| `/tmp/fail401312.bin` (host dump) | **`arch = sm_80`**, Ampere GEMM section names |
| VM `libcublasLt.so.12` | Resolved from **`/usr/local/lib/ollama/`** → **`cuda_v12/libcublasLt.so.12.8.4.1`** |
| `gpu_properties.h` (pre-fix) | **`#define GPU_DEFAULT_CC_MAJOR 8`**, **`MINOR 9`** |

## Authority reminder

- **Host (dom0):** read-only for this agent — no mediator edits/rebuilds.
- **VM:** full — deploy `.so`, transfer guest-shim, run **`install.sh`**, restart **`ollama`**.

## Implemented mitigations (this change set)

1. **`guest-shim/gpu_properties.h`:** Set **`GPU_DEFAULT_CC_MAJOR 9`**, **`GPU_DEFAULT_CC_MINOR 0`** with comment referencing this doc (replacing Ada 8.9 workaround).
2. **`guest-shim/libvgpu_cuda.c` `fetch_gpu_info()`:** After applying live host struct, **force** `g_gpu_info.compute_cap_major/minor` to **`GPU_DEFAULT_*`** so **`CUDA_CALL_GET_GPU_INFO`** cannot leave **8.9** (or other mismatches) in place for kernel selection.

## Verification steps (after rebuild + install on VM)

1. `sudo systemctl restart ollama`
2. `journalctl -u ollama -n 40 | grep inference` → expect **`compute=9.0`** (or equivalent) for H100 line if Ollama prints CC from device props.
3. Trigger **tinyllama** generate; on host: **`grep module-load /tmp/mediator.log | tail`** — expect **no** **`INVALID_IMAGE`** for **401312**, or new dump shows **`sm_90`** in **`cuobjdump -elf`**.
4. If still failing: next lever is **newer `libcublasLt`/`libcublas`** on the guest aligned with Hopper + driver, or NVIDIA matrix check for **12.8.4.1** vs dom0 driver **545.23.06**.

## Related docs

- **`BUILD_LIBGGML_CUDA_HOPPER.md`** — building **`libggml-cuda.so`** with **`sm_90`**
- **`VERIFICATION_AUTH_ENV_PHASE1_MAR20.md`** — role, environments, Phase 1
- **`ERROR_TRACKING_STATUS.md`** — rolling blocker notes

## Security note

Do not commit or log **sudo** or **SSH** passwords. Use group membership / `sudoers` for automation where possible.

---

## Operational recovery (Mar 21 session — `install.sh` side effects)

1. **`install.sh` line ~661** failed with `EOF: command not found` (broken heredoc). That **truncated** `/etc/systemd/system/ollama.service.d/vgpu.conf` to **only** `ExecStart=` lines, dropping all **`Environment=`** entries → **CPU-only** discovery (`initial_count=0`).
2. **Restore drop-in** from repo template: **`phase3/vm_ollama_vgpu.conf`** (copy to VM `vgpu.conf`, `systemctl daemon-reload`).
3. **`ExecStart` wrapper** must live under **`/usr/local/bin/`** (e.g. **`ollama_vgpu_wrapper.sh`**) — **`User=ollama`** cannot execute scripts under **`/home/test-4/...`** (permission denied).
4. **`LD_PRELOAD`:** **`libvgpu-exec.so`** + **`libvgpu-syscall.so`** with the Go **`ollama.bin.new`** binary caused **exit 126** *Inappropriate ioctl for device*. **Do not** preload those for the main server. Use **`libvgpu-cudart.so`**, **`libvgpu-cuda.so`**, **`libvgpu-nvml.so`** only (see **`guest-shim/ollama_wrapper.sh`**).
5. **Wrong binary chain:** **`exec /usr/local/bin/ollama serve`** runs the small launcher that **`exec`s `ollama.real`** (upstream), **not** **`ollama.bin.new`** (patched Phase3 build). Wrapper must **`exec /usr/local/bin/ollama.bin.new serve "$@"`**.
6. **Missing shims in `/usr/lib64`:** If **`libvgpu-exec.so` / `libvgpu-syscall.so`** are absent, older wrappers log **ld.so preload ignored** — either install them (`sudo cp` from `guest-shim/`) or **remove** them from **`LD_PRELOAD`** (recommended for Go).
7. **`journalctl` `compute=8.9`:** After **`gpu_properties.h` → 9.0** and **`fetch_gpu_info()`** CC force, shims **do** contain **CC=9** strings; Ollama may still **display 8.9** from another probe path — **re-verify** with a fresh **`module-load`** / **`cuobjdump`** on **`fail401312.bin`** after a full generate.

### Files touched in repo (this session)

- **`guest-shim/gpu_properties.h`** — **9.0**
- **`guest-shim/libvgpu_cuda.c`** — force CC after host GPU info overlay
- **`guest-shim/ollama_wrapper.sh`** — slim **`LD_PRELOAD`**, **`ollama.bin.new`**, **`/opt/vgpu/lib`** in **`LD_LIBRARY_PATH`**
- **`vm_ollama_vgpu.conf`** — restored **`Environment=`** block for systemd
- **`ERROR_TRACKING_STATUS.md`** — pointer to this doc
