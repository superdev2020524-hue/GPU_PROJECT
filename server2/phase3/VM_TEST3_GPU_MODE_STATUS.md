# test-3 VM — Ollama GPU mode status (after direct interaction)

## Goal: Ollama operating in GPU mode

**Intended pipeline (from phase3 docs):** Ollama in the guest VM makes CUDA API calls → **guest SHIM** (libvgpu-cuda etc.) **intercepts** those calls → sends them to **VGPU-STUB** → **mediator** forwards to **host CUDA** on the **physical GPU** → work is done on the host → results return to Ollama. That is “Ollama in GPU mode”: inference uses the GPU via this chain, not CPU.

- **Discovery** = Ollama “sees” a GPU (reports it, uses it for context sizing). That requires the **runner** to load our shim so when the backend calls `cuDeviceGetCount` / `cudaGetDeviceCount`, the SHIM answers and the backend reports the device.
- **Inference in GPU mode** = When the user runs a model, the **runner** loads the model and makes **CUDA calls** (alloc, copy, kernel launch, etc.). Those calls are **intercepted by the SHIM** and sent through VGPU-STUB to the mediator; the host performs them on the physical GPU and returns results. No .go “trick” to “make it recognize a GPU” is required beyond ensuring the runner loads the shim and the backend uses the GPU; the rest is SHIM interception of real CUDA calls.

## Current state (Mar 5, 2026)

- **Discovery (working baseline):** When runner has no LD_PRELOAD, LD_LIBRARY_PATH with /opt/vgpu/lib first, and no CUBLAS shim in /opt/vgpu/lib: logs show device_count=2, library=CUDA, H100 80GB. If you see device_count=0 (e.g. after adding CUBLAS shim): remove it and restart (see CUBLAS regression below). CUBLAS shim removed; `libggml-cuda-v12.so` symlinks added. Runner's **child** loads libcuda/libcudart but our shim's device-count APIs are not called—child may not get `LD_LIBRARY_PATH` with `/opt/vgpu/lib` first. (Previously: Ollama reported CUDA (NVIDIA H100 80GB); server uses 80 GiB VRAM for context; the runner loads our shim (via `LD_LIBRARY_PATH` into `cuda_v12`, which uses `/opt/vgpu/lib`), so when the backend asks for device count it gets 1 device from the SHIM. So “Ollama recognizes the GPU” is done.
- **Hopper lib deployed (Mar 5):** `libggml-cuda.so` built with sm_90 (Option B Docker) was deployed to `/usr/local/lib/ollama/cuda_v12/`. Ollama restarted. Logs show inference runner starting with device_count=2 and "inference compute" library=CUDA, H100 80GB—no GGML arch crash.
- **Inference in GPU mode:** The runner can now complete model load and issue CUDA calls (SHIM → VGPU-STUB → mediator → host GPU). To confirm: on the VM run `ollama run llama3.2:1b 'Hi'` or use `/api/generate`; you should get a text response. `api/ps` after a run shows loaded models.
- **VM disk (not your local PC):** The **test-3 VM’s** root filesystem was full (0 bytes free). That was the **VM’s** disk, not your local machine. The unnecessary **CUDA 11.5 toolkit** was removed from the VM (it does not provide sm_90/Hopper and the VM does not need it to run Ollama with the shim). After removal + `apt autoremove`, the VM has **~3.5 GB free** (91% used). To get the Hopper-capable library: build on your **local PC** with **Option A** (host with CUDA 12) or **Option B** (Docker: `sudo ./build_libggml_cuda_hopper_docker.sh` in phase3); then deploy the built `libggml-cuda.so` to the VM with `deploy_libggml_cuda_hopper.py`. The VM does not need Docker or CUDA for the build—only to receive the deployed library.

## Verification record (Mar 5, 2026 — automated)

- **Hopper lib:** Present at `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so` (187 128 056 bytes, Mar 5 09:02). Ollama service **active**.
- **Discovery:** Logs show "runner /info response" device_count=2, "inference compute" library=CUDA, NVIDIA H100 80GB, 80.0 GiB total. No GGML arch crash.
- **Server:** `GET /` → "Ollama is running". `GET /api/tags` → llama3.2:1b listed. `GET /api/ps` → `{"models":[]}` when idle.
- **Generate (automated):** Multiple `/api/generate` attempts (90–120 s timeout) returned **0 bytes**. A **5-minute** timeout run (curl -m 300) ended with **exit 28 (timeout)**; no response body. Inference does **not** complete within 5 minutes. Logs show a runner started for generate (09:03:08) with device_count=2 and "unable to refresh free memory"; no completion or error logged. **Next step:** On the VM, run `ollama run llama3.2:1b 'Hi'` from the console (or strace the runner during a generate) to see if the runner is blocking on model load / first CUDA call; check host/mediator and vGPU transport if it never returns.

**BAR0 permissions (Mar 5):** Narrowed scope on VM: VGPU-STUB device (10de:2331) is present. **resource0 was root-only**; ollama service runs as user `ollama`, so the shim could not open BAR0 and transport init failed. Applied: `chmod 666` on resource0 and udev rule `/etc/udev/rules.d/99-vgpu-stub-resource0.rules` for persistence. After fix, inference still timed out → next suspect is **host mediator not responding** (A3); confirm on host that mediator_phase3 is running and accepting the VM’s device.

**Host mediator (Mar 5):** Mediator is running: v3.1, 1 QEMU VM with vgpu-cuda, socket at `/var/xen/qemu/root-213/tmp/vgpu-mediator.sock` (test-3 = root-213). Inference still times out with mediator up. **Next:** With mediator in foreground, trigger generate from VM and watch for `[SOCKET] New connection` or `[CONNECTION] New connection`. If none, guest may not be writing BAR0 or stub not connecting; if connection appears, debug mediator request handling.

**VM diagnostics (Mar 5):** During generate: (1) **No process had resource0 open** at 5–10 s (sudo lsof \| grep resource0 → empty). (2) **No runner process visible** in `ps`/`pgrep` during generate (pgrep -af 'ollama.bin runner' → 0 lines over 8 s). So either the runner exits very quickly or is not spawned as a separate process for generate. (3) Main ollama process (101491) has only thread children in pstree, no "ollama.bin runner" child. **Shim diagnostic:** A file marker was added in `libvgpu_cuda.c`: when the runner reaches `ensure_connected()` and is about to call `cuda_transport_init()`, the shim writes `pid=<pid>` to `/tmp/vgpu_ensure_connected_called`. **Deployed via `transfer_libvgpu_cuda.py`** (single-file transfer, no full phase3 SCP). **Result (Mar 5):** After generate: `NO_MARKER_FILE` — the runner **never reached ensure_connected()**. So the block is **before** the transport layer: either the inference runner does not load our shim, or it exits/crashes before the first CUDA call that invokes ensure_connected, or the first inference path does not go through that call.

**Prior PHASE3 fix verified (Mar 5):** cuda_v12 symlinks (ROOT_CAUSE_RUNNER_SUBPROCESSES.md, GPU_MODE_FIX_SYMLINK_CUDA_V12.md) are still correct: `/usr/local/lib/ollama/cuda_v12/libcuda.so.1` → `/opt/vgpu/lib/libcuda.so.1` (shim). So "runner not loading shim" is not regressed.

**Regression from CUBLAS shim in /opt/vgpu/lib:** cuda_v12 contains **real** libcublas (libcublas.so.12 → libcublas.so.12.8.4.1). We added `/opt/vgpu/lib/libcublas.so.12` → our libvgpu-cublas.so.12. Because `LD_LIBRARY_PATH` has `/opt/vgpu/lib` first, the runner then loaded **our** CUBLAS shim instead of real CUBLAS; our shim returns dummy handles, and discovery then reported **device_count=0** (id=cpu, total_vram="0 B"). **Fix:** On the VM run: `sudo rm -f /opt/vgpu/lib/libcublas.so.12 && sudo systemctl restart ollama` to restore discovery (libcublas will resolve from cuda_v12 again). Do **not** install our CUBLAS shim in /opt/vgpu/lib unless it is made compatible with GGML init, or discovery will stay at 0 devices.

**Checklist to restore working GPU discovery (do not remove or revert these):** (1) No `libcublas.so.12` in `/opt/vgpu/lib`. (2) Runner gets `LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12` and **no** LD_PRELOAD (patched `llm/server.go`). (3) **NeedsInitValidation** patch: in `ml/device.go`, return false for CUDA (e.g. `return d.Library == "ROCm"` only) so second-pass verification does not filter out CUDA devices; apply `patches/skip_cuda_init_validation_for_vgpu.patch` and rebuild `ollama.bin`. (4) Optional: `libggml-cuda-v12.so` symlinks in cuda_v12 and parent ollama dir. (5) Restart ollama after any change.

## Quick path to GPU inference (Mar 5, 2026)

Now that discovery-in-GPU-mode state is confirmed (Hopper lib, device_count=2 when conditions are met), use this order to get inference working and then fix remaining blockers:

1. **Apply Ollama vGPU patches (local or on VM)**  
   From phase3: `python3 apply_ollama_vgpu_patches.py` (optionally `OLLAMA_SRC=/path/to/ollama`). If files are read-only, run `chmod u+w ollama-src/ml/device.go ollama-src/llm/server.go` first. This script: (a) sets `NeedsInitValidation()` to return false for CUDA in `ml/device.go`; (b) in `llm/server.go` prepends `/opt/vgpu/lib` to runner `LD_LIBRARY_PATH` and `OLLAMA_LIBRARY_PATH`, and strips `LD_PRELOAD` from runner env.

2. **Build and install ollama.bin on the VM**  
   On the VM (or cross-build): in the patched Ollama tree run `go build -o ollama.bin .` and install the binary to `/usr/local/bin/ollama.bin`. Restart: `sudo systemctl restart ollama`.

3. **VM checklist before testing**  
   - No CUBLAS shim in `/opt/vgpu/lib`: `sudo rm -f /opt/vgpu/lib/libcublas.so.12` if present.  
   - BAR0 usable: `ls -la /sys/bus/pci/devices/0000:00:05.0/resource0` (or your VGPU-STUB BDF); udev rule or `chmod 666` so ollama user can open it.  
   - Hopper lib and symlinks: `libggml-cuda.so` in `/usr/local/lib/ollama/cuda_v12/`, and `libggml-cuda-v12.so` symlinks there and in parent dir (see BUILD_LIBGGML_CUDA_HOPPER.md).

4. **Test inference**  
   On VM: `ollama run llama3.2:1b 'Hi'` or `curl -s -m 120 'http://127.0.0.1:11434/api/generate' -d '{"model":"llama3.2:1b","prompt":"Hi","stream":false,"options":{"num_predict":2}}'`. Check logs: `sudo journalctl -u ollama -n 50 --no-pager | grep -E "runner /info|inference compute|starting runner"`.

5. **If inference still times out**  
   - Check whether the runner reaches the shim: after a generate attempt, look for `/tmp/vgpu_ensure_connected_called` (written by `libvgpu_cuda.c` when `ensure_connected()` runs). If absent, the runner is not hitting the first CUDA call that uses the transport (or is not loading our shim).  
   - On host: run mediator in foreground and trigger a generate; watch for `[SOCKET] New connection`. If no connection, guest is not opening BAR0 or stub not connecting; if connection appears, debug mediator request handling (see PHASE3_INFERENCE_ISSUES_AND_NEXT_STEPS.md).

**Done (Mar 5):** Patches applied in memory and transferred via `transfer_ollama_go_patches.py`; VM build used `/usr/local/go/bin/go` (Go 1.26). Installed `ollama.bin` and restarted ollama. Discovery still shows `id=cpu library=cpu`; `/tmp/vgpu_ensure_connected_called` absent after generate → runner still not using shim/CUDA path. **Next:** Confirm runner env (e.g. set `OLLAMA_DEBUG=1`, restart, trigger discovery, and check logs for runner `LD_LIBRARY_PATH`); or run runner manually with `LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12` and no LD_PRELOAD, then `curl localhost:<port>/info` to verify it returns GPU.

**If discovery is still 0 after removing CUBLAS shim:** Ensure runner gets `LD_LIBRARY_PATH=/opt/vgpu/lib:.../cuda_v12` and no LD_PRELOAD. Optional: `libggml-cuda-v12.so` symlinks in cuda_v12/parent. **GGML backend filename (reference):** The loader in `ggml-backend-reg.cpp` looks for `libggml-<name>-*` (e.g. `libggml-cuda-v12.so`), not `libggml-cuda.so`. Added symlinks so the loader finds the CUDA backend: (1) `/usr/local/lib/ollama/cuda_v12/libggml-cuda-v12.so` → `libggml-cuda.so`, (2) `/usr/local/lib/ollama/libggml-cuda-v12.so` → `cuda_v12/libggml-cuda.so`. Restart and manual runner tests still show CPU only; `/tmp/cudart_get_count_called.txt` and `/tmp/cuda_get_count_called.txt` are **not** created when triggering `/info`, so our shim’s device-count APIs are never called. **LD_DEBUG finding:** The **child** process (runner subprocess that handles `/info`) does load `libcuda.so.1`, `libcudart.so.12`, `libcublas.so.12` (i.e. it loads the CUDA backend). So the CUDA backend is loaded in the child, but either (a) the child does not inherit `LD_LIBRARY_PATH` with `/opt/vgpu/lib` first (so it loads real libcuda from cuda_v12, which reports 0 devices), or (b) the backend’s device-count call path is not used for the discovery `/info` flow. **Next:** On the VM, confirm whether the discovery child inherits the runner’s env: e.g. while the service is running, trigger discovery and inspect the child’s `/proc/<pid>/environ` (or run with `LD_DEBUG=libs` and confirm which `libcuda.so.1` is opened). If the child gets `/opt/vgpu/lib` first, the shim should be used and discovery should report 1 GPU; if not, the server/runner spawn logic may need to pass `LD_LIBRARY_PATH` explicitly to the child.

## What uses ~35 GB on the VM

Rough breakdown: **/var** ~13 GB (e.g. `/var/snap/ollama` ~5.5 GB, `/var/lib/snapd` ~5.2 GB, `/var/cache/apt` ~1.7 GB); **/usr** ~13 GB (e.g. `/usr/local/lib` ~4.3 GB, `/usr/share/ollama` ~1.3 GB, `/usr/lib`, `/usr/share`); **/snap** ~9.4 GB (Ollama snap ~4.4 GB, gnome-42 ~2.6 GB, firefox ~1.4 GB); **/home** ~3.1 GB. Ollama (snap + data + /usr/local) and desktop snaps (GNOME, Firefox) account for most usage.

## Done on VM (Mar 4, 2026)

1. **Guest shims built and installed**
   - Copied missing sources (libvgpu_cudart.c, libvgpu_cublas.c, libvgpu_cublasLt.c) via SCP.
   - Ran `make guest` in `/home/test-3/phase3`; built libvgpu-cuda.so.1, libvgpu-cudart.so, libvgpu-nvml.so.
   - Installed to `/opt/vgpu/lib/` with symlinks: libcuda.so.1, libcudart.so.12, libnvidia-ml.so.1.

2. **Ollama service**
   - Stopped snap listener so `ollama.service` can bind to 11434.
   - **Full LD_PRELOAD (NVML + CUDA) without SEGV:** Running the bash script (`/usr/local/bin/ollama serve`) with LD_PRELOAD caused SEGV in `is_caller_from_our_code()` when bash loaded the shims. Running **`ollama.bin` directly** with the same LD_PRELOAD does not crash. So the service was changed to run **`ollama.bin`** directly: drop-in `ollama.service.d/vgpu.conf` sets `ExecStart=` then `ExecStart=/usr/local/bin/ollama.bin serve` and `Environment=LD_PRELOAD=/opt/vgpu/lib/libvgpu-nvml.so:/opt/vgpu/lib/libvgpu-cuda.so.1` (and `LD_LIBRARY_PATH`, `OLLAMA_NUM_GPU=1`). Service stays up with full shims.

3. **cuda_v12 symlinks**
   - In `/usr/local/lib/ollama/cuda_v12/`: libcuda.so.1, libcudart.so.12, libnvidia-ml.so.1 point to `/opt/vgpu/lib/` shims so loading from that dir uses the vGPU shims.

4. **Snap listener**
   - Restarted `snap.ollama.listener.service`; both it and `ollama.service` can show active (port conflict possible if both try to serve).

5. **Discovery trace (Mar 4–5):** strace shows a child opens `libggml-cuda.so` and `cuda_v12/libcuda.so.1` (our shim), but `/tmp/cuda_get_count_called.txt` and `/tmp/nvml_get_count_called.txt` are never created — so `cuDeviceGetCount` and `nvmlDeviceGetCount_v2` are not called during discovery. Main process has `libvgpu-nvml.so` in maps (with NVML-only LD_PRELOAD) but does not call NVML device count. Adding `libvgpu-cuda.so.1` to LD_PRELOAD causes SEGV; reverted to NVML-only so service stays up.

## Current behavior

- **Discovery:** Logs still show `id=cpu`, `library=cpu`, `total_vram="0 B"`. Child loads our libcuda but never calls `cuDeviceGetCount`; main process has NVML shim (LD_PRELOAD) but never calls `nvmlDeviceGetCount_v2`.
- **Inference:** `ollama run llama3.2:1b '...'` works (e.g. “Hello”) and runs on CPU.

## Coredump analysis (Mar 5, 2026)

- Cores under `/tmp/cores/`; crash in `is_caller_from_our_code()` (from our `read()`), called during bash running `ollama serve`. Cause: `__builtin_return_address`/`dladdr` in that function SEGV during early startup. Read passthrough (first 100k read()s) and is_caller passthrough (first 100k calls return 0) added; service still core-dumps (crash may be in prologue or from open/fopen path). Use **NVML-only** LD_PRELOAD for a stable service.

## Blocker: runner does not get LD_PRELOAD (Mar 5, 2026)

- **Observed:** Runner processes have **no** `LD_PRELOAD` or `LD_LIBRARY_PATH` in their environment; the main server process has them (from systemd).
- **Cause:** Ollama's server sets a **custom** environment when starting the runner (Go `exec.Cmd` with `cmd.Env`), so the runner does not inherit the server's `LD_PRELOAD`.
- **Tried:** A small `libvgpu-exec-inject.so` that overrides `execve()` and injects vGPU env when `argv` contains `"runner"`. **Does not work:** Go uses the **raw** `execve` syscall, not libc's `execve`, so the hook is never called.
- **Fix (without modifying Ollama):** None. The runner executable path is the same as the server's; a wrapper chain cannot make “only the runner” go through a wrapper without changing what the server uses for that path.
- **Fix (with Ollama change):** Patch applied; runner now gets `LD_PRELOAD` (and `libvgpu-cudart.so` added to it). Discovery still uses a path that does not call the shimmed device-count APIs in the observed window.

## Current results (Mar 5, 2026) — patch deployed

- **Ollama LD_PRELOAD patch:** Applied to `llm/server.go` (line 431); patched binary built with Go 1.26 on VM and installed as `/usr/local/bin/ollama.bin`. Service runs with full LD_PRELOAD (NVML + CUDA); no SEGV.
- **Server process:** Has `LD_PRELOAD` and `LD_LIBRARY_PATH` in environ (from vgpu.conf).
- **Discovery:** Logs still show `id=cpu`, `library=cpu`, `total_vram="0 B"`. **Runner LD_PRELOAD confirmed:** debug log shows `runner env LD_PRELOAD` with value including `libvgpu-nvml.so:libvgpu-cuda.so.1:libvgpu-cudart.so`.
- **Shim usage:** With new Ollama engine, `cuDeviceGetCount`/`cudaGetDeviceCount`/`nvmlDeviceGetCount_v2` are not observed during discovery (debug files `/tmp/cuda_get_count_called.txt`, `/tmp/cudart_get_count_called.txt` never created). `libggml-cuda.so` has undefined `cudaGetDeviceCount@libcudart.so.12`; cuda_v12 symlinks point to our shims.
- **Manual runner test:** Running `ollama.bin runner --ollama-engine --port 33444` with same env and `curl /info` returns `[]`; no shim debug files created. Backend logs "system CPU.0.LLAMAFILE=1".
- **Full check:** See **phase3/VM_CHECK_RESULTS.md**.

## cuGetErrorString fix (Mar 5, 2026)

- **Cause:** `libggml-cuda.so` needs `cuGetErrorString`; our shim implemented it but inside an `#if 0` block, so it was not compiled/exported. That made `dlopen("libggml-cuda.so")` fail with "undefined symbol: cuGetErrorString" when using the shim via LD_LIBRARY_PATH.
- **Fix:** (1) Added `__attribute__((visibility("default")))` to `cuGetErrorString`. (2) Moved the Error handling block out of `#if 0` by inserting `#endif` after the malloc block and wrapping Error handling + stub macro in `#if 1`. Rebuilt shim on VM and installed to `/opt/vgpu/lib/`.
- **Result:** A minimal `dlopen("libggml-cuda.so")` test with only LD_LIBRARY_PATH (no LD_PRELOAD) now prints "dlopen OK". Runner with full vGPU env still returns `[]` and "system CPU.0.LLAMAFILE=1"; shim debug files still not created.

## Runner without LD_PRELOAD — GPU discovery works (Mar 5, 2026)

- **Finding:** With **only** `LD_LIBRARY_PATH` (no `LD_PRELOAD`), the runner's `/info` returns full GPU info (backend CUDA, name CUDA0, NVIDIA H100 80GB HBM3). With `LD_PRELOAD`, the runner's `dlopen` resolves to libdl, so the GGML backend never loads `libggml-cuda.so`; without `LD_PRELOAD`, the real `dlopen` loads `libggml-cuda.so`, which pulls in our shim via `LD_LIBRARY_PATH`, and the CUDA backend sees the GPU.
- **Change:** Patched `llm/server.go` so the **runner** is started **without** `LD_PRELOAD` (only `LD_LIBRARY_PATH`, `OLLAMA_LIBRARY_PATH`, `OLLAMA_LLM_LIBRARY`). Main server can still use `LD_PRELOAD` from systemd; runner uses real loader and discovers GPU.
- **Important:** The server sets `cmd.Env = os.Environ()`, so the runner was still receiving `LD_PRELOAD` from the parent. We added an explicit **removal** of `LD_PRELOAD` from `cmd.Env` (filter loop: copy only entries where `!strings.HasPrefix(e, "LD_PRELOAD=")`). Script: `apply_ld_preload_fix.py` (run on VM in ollama repo root); or apply `patches/remove_ld_preload_from_runner.patch`.
- **Patch:** `patches/ollama_runner_ld_preload.patch` (ensures runner gets library paths); plus LD_PRELOAD removal in `server.go`. VM: applied via `apply_ld_preload_fix.py`, `go mod tidy`, built with `/usr/local/go/bin/go`, installed new `ollama.bin`, restarted service. Logs no longer show "runner env LD_PRELOAD".
- **LD_LIBRARY_PATH for runner (Mar 5):** Added explicit prepend of `/opt/vgpu/lib` to runner's `LD_LIBRARY_PATH` in the ensure block when not already present (`fix_server_go_runner_env.py`). Logs now show `runner env LD_LIBRARY_PATH` with value `LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12`. Manual run of the runner with that env (as test-3) returns GPU in `/info`; server-started runner still reports CPU in discovery (id=cpu, total_vram="0 B").
- **ensure_init fallback for runner (Mar 5):** In `libvgpu_cuda.c`, added fallback in `ensure_init()` so that if `OLLAMA_LIBRARY_PATH` or `OLLAMA_LLM_LIBRARY` is set, we treat the process as app and initialize (runner without LD_PRELOAD). Patched on VM with sed; rebuilt `libvgpu-cuda.so.1`, installed to `/opt/vgpu/lib`, restarted ollama. Discovery still shows CPU. `/tmp/vgpu_ensure_init.log` only has older entries (manual runs); server-started runner may exit before we can inspect maps, or may not be loading our shim. Next: trace Ollama discovery (how many runners, how response is chosen) or capture runner process maps/stderr during discovery.

## Discovery flow and verification-runner crash (Mar 5, 2026)

- **Flow:** `discover/runner.go` → `GPUDevices()` does (1) first pass: for each libDir, `bootstrapDevices(ctx, dirs, nil)` starts a runner, calls `GetDevicesFromRunner` (GET runner `/info`), appends devices; (2) second pass: for each device with `NeedsInitValidation()` (CUDA/ROCm), starts another runner with `extraEnvs` (e.g. `CUDA_VISIBLE_DEVICES`, `GGML_CUDA_INIT=1`) and keeps the device only if that runner returns ≥1 device. Final list is passed to `LogDetails(devices)`; if empty, CPU fallback is logged.
- **Debug logging:** Added `slog.Info("runner /info response", "device_count", len(moreDevices))` and error-path logs in `ml/device.go` (`GetDevicesFromRunner`), rebuilt with `/usr/local/go/bin/go`, installed and restarted.
- **Observed:** First pass: one runner returns **device_count=2** (GPU discovered). Second pass: two verification runners are started (ports 34521, 40673); both hit **GetDevicesFromRunner error reason=runner_crashed** with **connection refused**. So the verification runners **exit before listening** on their HTTP port; both devices are then removed, and the final list is empty → CPU fallback.
- **Conclusion:** GPU discovery in the first pass works; the second-pass “init validation” runners crash (or exit immediately), so both GPU devices are filtered out. Next: (1) Run a verification runner manually with `CUDA_VISIBLE_DEVICES=0` and `GGML_CUDA_INIT=1` to see if it crashes; (2) If it does, fix the crash (shim or backend); (3) Or consider skipping init validation for this setup (e.g. patch `NeedsInitValidation` or avoid filtering when verification fails).

## Skip CUDA init validation — GPU discovery succeeds (Mar 5, 2026)

- **Change:** Patched `NeedsInitValidation()` in `ml/device.go` to return only `d.Library == "ROCm"` (skip CUDA), so the second-pass verification is not run for CUDA devices and the two devices from the first pass are kept.
- **Result:** Discovery now logs **inference compute** with **library=CUDA**, **name=CUDA0**, **description="NVIDIA H100 80GB HBM3"**, **total="80.0 GiB"**. No CPU fallback line when GPU is present.
- **Patch:** `patches/skip_cuda_init_validation_for_vgpu.patch` (apply from ollama repo root). On VM the edit was applied with sed; rebuild with `/usr/local/go/bin/go`, install `ollama.bin`, restart service.

## Inference verification (Mar 5, 2026)

- **Server uses GPU VRAM:** Journal shows `vram-based default context` total_vram="80.0 GiB" default_num_ctx=262144, so the server is using the discovered GPU for context sizing.
- **Request path:** On a generate/chat request the server starts a runner; journal shows `starting runner` and `runner /info response` device_count=2 for that runner, so the inference runner also sees the GPU.
- **To confirm inference uses GPU:** Run a short prompt (e.g. `ollama run llama3.2:1b 'Hi'`) and check `sudo journalctl -u ollama -f` for `starting runner` and `runner /info response` device_count=2 during the request. Runner stderr (e.g. "load_backend", "using device CUDA0") may appear in journal if the runner inherits the service's stderr; if not, set `OLLAMA_DEBUG=1` or increase log level and retry.

## Why verification runners exit (Mar 5, 2026)

- **Manual test:** Running a verification-style runner (with `CUDA_VISIBLE_DEVICES=0`, `GGML_CUDA_INIT=1`, same `LD_LIBRARY_PATH`/`OLLAMA_LIBRARY_PATH`) shows the runner **does** start and listen ("Server listening on 127.0.0.1:39998"). When `/info` is requested it does the dummy model load; CUDA init runs and the shim reports H100 (CC 9.0). Then **GGML aborts** with:
  - `ggml-cuda.cu:335: GGML_ASSERT(ggml_cuda_has_arch(info.devices[id].cc) && "ggml was not compiled with support for this arch") failed`
- **Cause:** `libggml-cuda.so` (Ollama's bundled build) was **not** compiled with support for compute capability 9.0 (Hopper). The second-pass init validation runs with `GGML_CUDA_INIT=1`, which triggers this arch check and the process exits, so the server sees "connection refused" and filters out the device.
- **Implication:** Skipping CUDA init validation (patch `NeedsInitValidation` to return false for CUDA) is the correct workaround until the VM uses an Ollama/GGML build that includes Hopper (sm_90). To re-enable validation later, rebuild Ollama/GGML with `GGML_CUDA_ARCH` or equivalent including 90.

## Inference and model load (Mar 5, 2026)

- **Observed:** `/api/generate` requests can timeout (e.g. 90s) with no response. When the inference runner loads the **real** model (not just the dummy /info load), it runs the same CUDA backend init path that checks `ggml_cuda_has_arch(cc)`. The bundled `libggml-cuda.so` does not support CC 9.0 (Hopper), so the runner can **abort** during model load and the server never gets a completion.
- **Conclusion:** End-to-end GPU inference likely fails for the same reason as second-pass verification: GGML was not built with Hopper support. **Fix:** Use or build an Ollama/GGML stack that compiles `libggml-cuda.so` with Hopper (sm_90) support, then deploy that library to the VM. See **BUILD_LIBGGML_CUDA_HOPPER.md** for steps (set `CMAKE_CUDA_ARCHITECTURES=90` on a build host with CUDA toolkit, build, copy `libggml-cuda.so` to `/usr/local/lib/ollama/cuda_v12/` on the VM, restart ollama).

## Next steps (to get GPU mode)

1. **Confirm runner loads libggml-cuda:** Run runner with `LD_DEBUG=libs` or `strace -e openat`, trigger `/info`, check if `libggml-cuda.so` is opened. (Runner env confirmed: debug log shows runner receives `OLLAMA_LIBRARY_PATH=/usr/local/lib/ollama:/usr/local/lib/ollama/cuda_v12` and `LD_PRELOAD` (with cudart). Discovery still reports CPU; `/tmp/cudart_get_count_called.txt` and `/tmp/cuda_get_count_called.txt` are **not** created during service restart, so `cudaGetDeviceCount`/`cuDeviceGetCount` are not called. So either `libggml-cuda.so` is not loaded (e.g. dlopen fails or path not used), or it loads but does not call our shimmed APIs. **Manual runner test (with `--ollama-engine`):** Curl `GET /info` returns **HTTP 200**. Runner stderr shows dummy model load and `system CPU.0.LLAMAFILE=1` (backend reports CPU, not CUDA). No "failed to initialize backend" in journal—backend init succeeds but only CPU device is used. **Shim debug files** (`/tmp/cudart_get_count_called.txt`, `/tmp/cuda_get_count_called.txt`) are **not** created when hitting `/info`, so `cudaGetDeviceCount`/`cuDeviceGetCount` are not called—either `libggml-cuda.so` is not loaded (e.g. dlopen fails silently in C++) or the CUDA backend's score/init path never calls them. **Confirmed:** With a temporary `slog.Info` in ggml.go, the runner logs show **both** paths being passed to the C++: `path=/usr/local/lib/ollama` and `path=/usr/local/lib/ollama/cuda_v12`. No "failed to load" appears in runner stderr, so either dlopen of `libggml-cuda.so` succeeds and the CUDA backend's score/init never calls `cudaGetDeviceCount`/`cuDeviceGetCount`, or dlopen fails and GGML_LOG_ERROR is not visible on stderr. **Next:** (1) Run the runner with `LD_DEBUG=bindings` and grep for `cudaGetDeviceCount`/`cuDeviceGetCount` to see if the CUDA backend binds to our shim when loaded. (2) If no binding, add a small test that dlopens `libggml-cuda.so` (with the same `LD_LIBRARY_PATH`/`LD_PRELOAD` as the runner) and calls `ggml_backend_score` to see whether the backend runs and calls into the shim.
2. **Fix SEGV with CUDA in LD_PRELOAD:** Done. Service runs `ollama.bin` directly with full LD_PRELOAD; no SEGV.
3. **Force discovery to use shims:** Our shim **does** report 1 GPU when `cudaGetDeviceCount`/`cuDeviceGetCount` is called; the goal is for Ollama's discovery path to call that. Ensuring the runner has `OLLAMA_LIBRARY_PATH` (so it loads `libggml-cuda.so` from `cuda_v12`) is the right direction; if the backend still reports 0 GPUs, the next check is dlopen success and symbol resolution for the loaded libggml-cuda.

## Useful commands on VM

```bash
# Service and env
systemctl status ollama.service
sudo cat /proc/$(systemctl show ollama.service -p MainPID --value)/environ | tr '\0' '\n' | grep LD_

# Libs in process
sudo grep -E 'vgpu|cuda|nvidia' /proc/$(systemctl show ollama.service -p MainPID --value)/maps

# Quick inference
ollama run llama3.2:1b 'Hi'
```

## Summary and suggested next steps (Mar 5, 2026)

- **GPU discovery:** Working. Discovery reports CUDA (NVIDIA H100 80GB); server uses 80 GiB VRAM for default context; request handlers start runners that report device_count=2.
- **Init validation:** Second-pass verification runners exit because `libggml-cuda.so` is not built with Hopper (sm_90) support; we skip CUDA init validation so devices are not filtered. See "Why verification runners exit" above.
- **Optional checks:** (1) Run a short inference (e.g. `ollama run llama3.2:1b 'Hi'`) and compare speed to CPU-only; GPU should be faster. (2) After loading a model, `ollama ps` or `curl -s http://127.0.0.1:11434/api/ps` may show processor/GPU when the API supports it. (3) To fix inference and optionally re-enable CUDA init validation, build and deploy a Hopper-capable `libggml-cuda.so` — see **BUILD_LIBGGML_CUDA_HOPPER.md**.

## Deploy (from host)

- **Build on VM only:** Copy the phase3 **source** tree to the VM; run `make guest` on the VM. Do not copy host-built .so files (host may not have GCC). See PHASE3_TEST3_DEPLOY.md.
- Use SCP-based deploy only (no chunked transfer):

```bash
cd /home/david/Downloads/gpu/phase3
python3 deploy_to_test3.py
```

For single-file updates, use `scp` (see PHASE3_TEST3_DEPLOY.md).
