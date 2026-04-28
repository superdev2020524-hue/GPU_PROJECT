# Phase 3 review and resume

*Created: Mar 18, 2026 — from review of phase3 path.*

---

## 1. Overall goal

**Ultimate goal (Phase 3):** All GPU-utilizing software in the VM sees the VGPU-STUB as a GPU and uses the data flow: guest → shims → VGPU-STUB → host mediator → physical GPU (host CUDA) → results back to the VM. General-purpose vGPU remoting, not Ollama-only.

**First-stage milestone (Phase 1 / Stage 1):** Successfully complete **GPU-mode inference in Ollama** in the VM as proof of the path: discovery in GPU mode, model load over vGPU, inference on host H100, response back to the VM.

Ref: **PHASE3_PURPOSE_AND_GOALS.md**, **ERROR_TRACKING_STATUS.md** § Goal and Phase 1.

---

## 2. Phase 1 milestones (summary)

| Milestone | Description |
|-----------|--------------|
| Discovery | Ollama sees vGPU as CUDA device (e.g. library=CUDA, "NVIDIA H100 80GB HBM3", total="80.0 GiB"). |
| Runner env | Runner subprocess gets `LD_LIBRARY_PATH` with `/opt/vgpu/lib` first so it loads guest shims. |
| Scheduler | When API omits `num_gpu` (decodes as 0), scheduler still uses GPU list from `getGpuFn` for load. |
| Load path | Load runner uses GPU: issues `cuMemAlloc` / `cuMemcpyHtoD` via shims (not CPU-only load). |
| Module load | Host loads CUDA fat binary (e.g. Hopper sm_90) so inference can run. |
| End-to-end | One full generate completes: model load + inference on vGPU path → host H100 → response. |

---

## 3. Steps taken and where we stand

### Done

- **Discovery (Issue A):** Skip CUDA init validation in `ml/device.go` (`NeedsInitValidation` → ROCm only). Patched and deployed; discovery shows CUDA, 80 GiB. Ref: BOOTSTRAP_FIX_SKIP_CUDA_INIT_VALIDATION, CURRENT_STAGE_SUMMARY_AND_NEXT_STEPS.
- **Runner LD_LIBRARY_PATH:** `llm/server.go` prepends `/opt/vgpu/lib` and passes LD_LIBRARY_PATH (and OLLAMA_*) to runner; LD_PRELOAD stripped so `dlopen` can load `libggml-cuda.so`. Verified: runner env shows `/opt/vgpu/lib` first. Ref: transfer_ollama_go_patches.py, ERROR_TRACKING_STATUS §4.
- **Refresh path:** `discover/runner.go` refresh uses `[]string{dir, ml.LibOllamaPath}` (GPU dir first). Patch in VM source; "unable to refresh free memory" still appears → refresh fails for another reason (timeout or CUDA in refresh path). Ref: REFRESH_AND_GPU_DETECTION_INVESTIGATION.
- **Scheduler NumGPU:** `server/sched.go` patched so when `NumGPU == 0` we still call `getGpuFn`; if GPUs returned, set `NumGPU = -1` and use GPU list. Applied via patch_sched_numgpu.py. Ref: STAGE1_SCHED_NUMGPU_FIX.
- **GPULayers fallback:** `llm/server.go` patched so when `len(gpus) > 0 && gpuLayers.Sum() == 0` we force one layer onto first GPU. Fallback never logged → createLayout already returns non-empty gpuLayers; server is sending GPU layers. Ref: LOAD_RUNNER_GPULAYERS_FIX, patch_llm_gpulayers_fallback.py.

### Current blocker (where work paused)

- **Load runner does not call CUDA alloc/HtoD.** Despite non-empty GPULayers from server, the process that performs model load **never** invokes `cuMemAlloc` (verified: `/tmp/vgpu_cuMemAlloc_called.log` empty; no SUBMIT 0x0030/0x0032 in verify log). So model load is using the **CPU** path. Conclusion (ERROR_TRACKING_STATUS §6): problem is **runner-side** — either (a) load runner selects CPU backend despite non-empty GPULayers, or (b) the runner we are tracing is not the one that receives the load. **Next step:** inspect runner/backend selection (allocModel, how CUDA vs CPU is chosen).

### Downstream (after load uses GPU)

- **CURRENT_STATE_AND_DIRECTION.md** describes a state where the pipeline runs through alloc/HtoD and model transfer, then **host** `cuModuleLoadFatBinary` fails with INVALID_IMAGE (no sm_90). Fix: deploy `libggml-cuda.so` built with `CMAKE_CUDA_ARCHITECTURES=90` to the VM (BUILD_LIBGGML_CUDA_HOPPER.md). That applies once the load path uses GPU.

---

## 4. Errors currently encountered (Phase 1)

| Error | Status / cause |
|-------|-----------------|
| "unable to refresh free memory, using old values" | Still present. Refresh dir order is correct; failure is another cause (timeout, refresh-runner env, or cuMemGetInfo in refresh path). |
| No alloc/HtoD (0x0030/0x0032) in verify log | Load runner not using GPU: either backend selection chooses CPU, or wrong process traced. |
| cuMemAlloc_called.log empty | Same: load runner never calls cuMemAlloc → CPU load. |
| (If GPU load path used) cuModuleLoadFatBinary INVALID_IMAGE | VM’s libggml-cuda.so has no sm_90; need Hopper-built .so on VM. |

---

## 5. Assistant permissions

| Scope | Allowed | Not allowed |
|-------|--------|-------------|
| **Host** | Check host logs; read file contents for investigation. | No editing of host files; no copy/build/make/restart on host. |
| **VM (test-4)** | Full: run commands, deploy, edit VM files, read logs, rebuild/install (e.g. ollama.bin, guest shims), restart services. | — |

Host-side fixes are documented; you apply them on the host. Ref: **ASSISTANT_PERMISSIONS.md**, **ASSISTANT_ROLE_AND_ANTICOUPLING.md**.

---

## 6. Stage that was paused and resumability

**Paused stage:** Stage 1 — getting the **load runner** to use the GPU so we see alloc/HtoD (0x0030/0x0032). Last conclusion: server sends non-empty GPULayers; load runner still does not call cuMemAlloc; next action is to **inspect runner/backend selection** (allocModel, how CUDA vs CPU is chosen).

**Can resume:** Yes. Under current permissions we can:

1. **On VM:** Add diagnostic logging in the Ollama runner (e.g. in the load handler) to confirm what the runner receives (GPULayers len/sum/op) and whether it selects GPU or CPU backend. LOAD_RUNNER_GPULAYERS_FIX §2 suggests appending to `/tmp/runner_load_gpulayers.txt` after `slog.Info("load", "request", req)` in **runner/ollamarunner/runner.go** (or equivalent path on VM: e.g. `runner/runner.go` or under `cmd/ollama`).
2. **On VM:** Run with OLLAMA_DEBUG=1 and capture "load_backend", "using device", or CPU fallback messages.
3. **On VM:** Confirm which binary is running and that getGpuFn returns non-empty at load time (e.g. existing sched log or inject_sched_gpu_count_log.py).
4. **Document** any host-side change needed; do not edit or build on the host.

---

## 7. Immediate next actions (resume)

1. **Locate runner load handler on VM**  
   In `/home/test-4/ollama/`, find the Go file where load requests are handled (e.g. `slog.Info("load", "request", req)` or "load" + request). Path may be `runner/runner.go`, `runner/ollamarunner/runner.go`, or under `llm/` / `cmd/ollama/`.

2. **Add runner-side GPULayers log**  
   From phase3 (or copy to VM): `python3 inject_runner_load_gpulayers_log.py /home/test-4/ollama/runner/runner.go` (adjust path if the load handler lives elsewhere, e.g. under `runner/ollamarunner/`). That injects a block that appends `gpulayers=<len> sum=<sum> op=<operation>` to `/tmp/runner_load_gpulayers.txt` after the load-request log. Rebuild ollama.bin on VM, install, restart ollama.

3. **Verify**  
   Clear `/tmp/runner_load_gpulayers.txt` and `/tmp/vgpu_cuMemAlloc_called.log`; trigger generate; check that runner_load_gpulayers.txt shows at least one line (e.g. Fit op) and whether cuMemAlloc_called.log gets lines. That confirms what the load runner receives and whether it then uses GPU.

4. **If runner receives non-empty GPULayers but still no alloc**  
   Search that runner code for where backend or device is chosen for load (allocModel, backend selection, or GPU vs CPU branch) and add a log or inspect logic so we see why CPU is used.

Ref: **ERROR_TRACKING_STATUS.md** §6–7, **LOAD_RUNNER_GPULAYERS_FIX.md**.
