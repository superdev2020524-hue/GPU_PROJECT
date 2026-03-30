# Ollama GPU Mode — Status and Next Steps

**Goal:** Ollama in the guest VM uses the vGPU (shim → VGPU-STUB → mediator → physical GPU) and reports `id=gpu`, `total_vram > 0`.

## What Was Done (aligned with goal)

1. **Runner gets vGPU env via `ollama.real` C wrapper**
   - Real binary moved to `/usr/local/bin/ollama.real.bin`.
   - `/usr/local/bin/ollama.real` is now a **C binary wrapper** (`ollama_real_wrapper.c`) that sets `LD_PRELOAD`, `LD_LIBRARY_PATH`, `OLLAMA_LLM_LIBRARY`, `OLLAMA_NUM_GPU` and `execve()`s `ollama.real.bin`.
   - Flow: systemd runs `ollama serve` → our `/usr/local/bin/ollama` (C wrapper) → `exec` ollama.real (this C wrapper) → `exec` ollama.real.bin serve. Main process has vGPU env.

2. **Shell script at `ollama.real` was reverted**
   - A shell script at `ollama.real` caused SEGV (shell run under `LD_PRELOAD`). Replaced with the C binary wrapper; service starts and stays up.

3. **Systemd override for vGPU env**
   - `/etc/systemd/system/ollama.service.d/vgpu.conf` adds `LD_PRELOAD` and `LD_LIBRARY_PATH` so the service (and any child that inherits) can see vGPU libs.

## Current Behavior

- **Service:** `ollama` is **active**; no SEGV.
- **Discovery:** Logs still show `inference compute id=cpu`, `total_vram="0 B"`.
- **Runner path:** Main spawns runner as `cmd="/usr/local/bin/ollama.real.bin runner ..."` (from `os.Executable()`), so the runner is **not** started via `ollama.real` (our wrapper). The runner is `ollama.real.bin` and may get a **filtered** environment (no `LD_PRELOAD`) if Ollama’s Go code sets `cmd.Env` when spawning the runner.
- **No shim logs from discovery:** Journal has no `[libvgpu-cuda] cuDeviceGetCount()` or `[libvgpu-nvml] nvmlDeviceGetCount_v2()` lines, which suggests either:
  - The runner does **not** have our shims (no `LD_PRELOAD`), or
  - Runner stderr is not in the journal, or
  - Discovery uses a different code path.

## Next Steps (to reach GPU mode)

**Confirmed (Mar 2):** File-based log in the shim wrote PIDs to `/tmp/cuda_get_count_called.txt` when `cuDeviceGetCount` ran. After deploy and restart, the file stayed **empty** — the runner does **not** load our shims; discovery runs in the runner.

1. **Get the runner to load our shims** (required):
   - **Option A:** Change the executable used for the runner to our wrapper (e.g. make the runner path `ollama.real` instead of `ollama.real.bin`). That would require the **main** process to report its executable as `ollama.real`, which would require the main to be the wrapper. That conflicts with the current design (main is `ollama.real.bin`).
   - **Option B:** **Binary-patch** the Ollama binary so the runner is started with a path that points to our wrapper (e.g. replace the runner executable path construction so it uses `/usr/local/bin/ollama.real`). Non-trivial and fragile.
   - **Option C:** Ask/contribute upstream so Ollama preserves or explicitly passes `LD_PRELOAD` (and related env) to the runner subprocess.

3. **If the runner does get our shims** but still reports CPU:
   - Verify that discovery APIs (`cuDeviceGetCount`, `nvmlDeviceGetCount_v2`, etc.) return 1 device and non-zero VRAM and that Ollama’s GPU detection logic accepts our PCI/driver strings (or add minimal logging to confirm what we return).

4. **Model load (“failed to read magic”)**
   - **Baseline restored:** `ollama.real` is again the **real** Ollama binary (no C wrapper); symlinks in `/usr/local/lib/ollama/` and `cuda_v12/` point to `/opt/vgpu/lib` shims.
   - **Blocker:** With our shim preloaded, `dlopen("/lib/.../libc.so.6")` returns NULL (tested on VM: small C program with `LD_PRELOAD=shim` gets `h=(nil)`), so `ensure_real_libc_resolved()` never gets a libc handle and `g_real_fopen_global` stays NULL. We always take the syscall+fdopen path for excluded paths; a `read()` fallback in `fread` for fd>=3 is in place, but model load still fails (“failed to read magic”), so either our `fread` is not used for the blob stream (e.g. GGUF in another DSO that binds to libc’s fread) or another read path is used.
   - **Attempted:** (1) Caching `real_dlopen` in a constructor — reverted (constructor + `dl_iterate_phdr` could SEGV during early init). (2) **`__libc_dlsym`**: resolve real `dlopen` via `__libc_dlsym(RTLD_NEXT, "dlopen")` when available (glibc); on the VM’s libc this symbol is not exported, so it is NULL. (3) **ELF fallback**: `resolve_real_dlopen()` now tries `dl_iterate_phdr` to find libc, then parses its ELF file (DT_DYNAMIC → DT_SYMTAB/STRTAB, vaddr→file offset via PT_LOAD) to get the `dlopen` symbol offset and returns `libc_base + offset`. Resolve is done on first `dlopen()` (no constructor). Transfer script now runs `make clean` before `make guest`. Next: run under `strace -f -e openat,open,read` during model load to see which process opens the blob; or bypass blob interception.

## Build / Deploy

- **`ollama_real_wrapper`:** Build with `make ollama-real-wrapper` (or on VM: compile `ollama_real_wrapper.c` and install the binary as `/usr/local/bin/ollama.real`; real binary must be at `/usr/local/bin/ollama.real.bin`).
- **Guest shims:** Build on VM with `make guest` and install under `/opt/vgpu/lib/` as usual. To push updated `libvgpu_cuda.c` from host: `python3 phase3/transfer_libvgpu_cuda.py` (chunked transfer, then build and install on VM).
