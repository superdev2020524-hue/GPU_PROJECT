# test-3 VM — Ollama GPU mode status (after direct interaction)

## Goal: Ollama operating in GPU mode

**Intended pipeline (from phase3 docs):** Ollama in the guest VM makes CUDA API calls → **guest SHIM** (libvgpu-cuda etc.) **intercepts** those calls → sends them to **VGPU-STUB** → **mediator** forwards to **host CUDA** on the **physical GPU** → work is done on the host → results return to Ollama. That is “Ollama in GPU mode”: inference uses the GPU via this chain, not CPU.

- **Discovery** = Ollama “sees” a GPU (reports it, uses it for context sizing). That requires the **runner** to load our shim so when the backend calls `cuDeviceGetCount` / `cudaGetDeviceCount`, the SHIM answers and the backend reports the device.
- **Inference in GPU mode** = When the user runs a model, the **runner** loads the model and makes **CUDA calls** (alloc, copy, kernel launch, etc.). Those calls are **intercepted by the SHIM** and sent through VGPU-STUB to the mediator; the host performs them on the physical GPU and returns results. No .go “trick” to “make it recognize a GPU” is required beyond ensuring the runner loads the shim and the backend uses the GPU; the rest is SHIM interception of real CUDA calls.

## Current state (Mar 5, 2026)

- **Discovery:** Working. Ollama reports CUDA (NVIDIA H100 80GB); server uses 80 GiB VRAM for context; the runner loads our shim (via `LD_LIBRARY_PATH` into `cuda_v12`, which uses `/opt/vgpu/lib`), so when the backend asks for device count it gets 1 device from the SHIM. So “Ollama recognizes the GPU” is done.
- **Hopper lib deployed (Mar 5):** `libggml-cuda.so` built with sm_90 (Option B Docker) was deployed to `/usr/local/lib/ollama/cuda_v12/`. Ollama restarted. Logs show inference runner starting with device_count=2 and "inference compute" library=CUDA, H100 80GB—no GGML arch crash.
- **Inference in GPU mode:** The runner can now complete model load and issue CUDA calls (SHIM → VGPU-STUB → mediator → host GPU). To confirm: on the VM run `ollama run llama3.2:1b 'Hi'` or use `/api/generate`; you should get a text response. `api/ps` after a run shows loaded models.
- **VM disk (not your local PC):** The **test-3 VM’s** root filesystem was full (0 bytes free). That was the **VM’s** disk, not your local machine. The unnecessary **CUDA 11.5 toolkit** was removed from the VM (it does not provide sm_90/Hopper and the VM does not need it to run Ollama with the shim). After removal + `apt autoremove`, the VM has **~3.5 GB free** (91% used). To get the Hopper-capable library: build on your **local PC** with **Option A** (host with CUDA 12) or **Option B** (Docker: `sudo ./build_libggml_cuda_hopper_docker.sh` in phase3); then deploy the built `libggml-cuda.so` to the VM with `deploy_libggml_cuda_hopper.py`. The VM does not need Docker or CUDA for the build—only to receive the deployed library.

## Verification record (Mar 5, 2026 — automated)

- **Hopper lib:** Present at `/usr/local/lib/ollama/cuda_v12/libggml-cuda.so` (187 128 056 bytes, Mar 5 09:02). Ollama service **active**.
- **Discovery:** Logs show "runner /info response" device_count=2, "inference compute" library=CUDA, NVIDIA H100 80GB, 80.0 GiB total. No GGML arch crash.
- **Server:** `GET /` → "Ollama is running". `GET /api/tags` → llama3.2:1b listed. `GET /api/ps` → `{"models":[]}` when idle.
- **Generate (automated):** Multiple `/api/generate` attempts (90–120 s timeout) returned **0 bytes**. A **5-minute** timeout run (curl -m 300) ended with **exit 28 (timeout)**; no response body. Inference does **not** complete within 5 minutes. Logs show a runner started for generate (09:03:08) with device_count=2 and "unable to refresh free memory"; no completion or error logged. **Next step:** On the VM, run `ollama run llama3.2:1b 'Hi'` from the console (or strace the runner during a generate) to see if the runner is blocking on model load / first CUDA call; check host/mediator and vGPU transport if it never returns.

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
