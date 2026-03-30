# PHASE3: GPU mode and transmission path status (Mar 5, 2026)

## Fix: Hopper lib (libggml-cuda.so) — why discovery was CPU

The **runner** (Ollama subprocess that does GPU discovery and inference) loads the CUDA backend from `libggml-cuda.so`. The **bundled** build of that library is **not** compiled with Hopper (sm_90 / compute capability 9.0), so on a system where the visible GPU is H100 (Hopper), the backend does not recognize the device and discovery reports CPU.

**Fix (already applied):** Build `libggml-cuda.so` with **sm_90** on a machine that has the right processor/CUDA, then copy it to the VM so the runner uses it.

- **Build (choose one):**
  - **Option A (host with CUDA):** `export CMAKE_CUDA_ARCHITECTURES=90 && cd ollama && make -j $(nproc)` → use `ollama/build/lib/ollama/libggml-cuda.so` (or path from your tree).
  - **Option B (Docker):** `cd phase3 && ./build_libggml_cuda_hopper_docker.sh` → output **`phase3/out/libggml-cuda.so`** (default). Optional: `./build_libggml_cuda_hopper_docker.sh /path/to/out` for a custom output dir.
- **Deploy to VM:** From phase3:  
  `python3 deploy_libggml_cuda_hopper.py out/libggml-cuda.so`  
  (or `python3 deploy_libggml_cuda_hopper.py /path/to/libggml-cuda.so`).  
  The script SCPs the file to the VM, installs it under `/usr/local/lib/ollama/cuda_v12/`, and restarts ollama.

**Docs/scripts in phase3:**

| Item | Purpose |
|------|--------|
| **BUILD_LIBGGML_CUDA_HOPPER.md** | Full instructions: why sm_90, Option A/B/C, verify after deploy |
| **build_libggml_cuda_hopper_docker.sh** | Build libggml-cuda.so with Hopper via Docker; output default `./out/libggml-cuda.so` |
| **deploy_libggml_cuda_hopper.py** | Deploy the built `.so` to test-3 VM (`vm_config.py`), install into cuda_v12, restart ollama |
| **phase3/out/libggml-cuda.so** | Example archive: Hopper-built library (use as reference or as deploy source) |

After deploying the Hopper lib, discovery should show `library=CUDA` and the transmission path (shim → VGPU-STUB → mediator) is used when inference runs.

---

## Check results (VM: test-3)

### 1. Ollama GPU mode — **GPU DETECTED** (Mar 5 fix)

- **Discovery:** Logs now show `initial_count=2` and `inference compute ... library=CUDA ... description="NVIDIA H100 80GB HBM3"` with non-zero VRAM.
- **Root cause of “GPU not detected”:** The **CUBLASLt shim** at `/opt/vgpu/lib/libcublasLt.so.12` (symlink to `libvgpu-cublasLt.so.12`) was being loaded before the real CUBLASLt from cuda_v12. GGML/CUDA init then saw dummy handles and reported 0 devices. Same class of issue as the CUBLAS shim (see VM_TEST3_GPU_MODE_STATUS.md).
- **Fix applied:** `sudo rm -f /opt/vgpu/lib/libcublasLt.so.12` on the VM. Do **not** install the CUBLASLt shim in `/opt/vgpu/lib` unless it is made compatible with GGML init.
- **Also required:** (1) `vgpu.conf`: `LD_LIBRARY_PATH` and `OLLAMA_LIBRARY_PATH` with **cuda_v12 before** `/usr/local/lib/ollama` (see RESTORE_GPU_LOGIC_CHECKLIST.md). (2) No `libcublas.so.12` in `/opt/vgpu/lib`. (3) Hopper-capable `libggml-cuda.so` in `/usr/local/lib/ollama/cuda_v12/`. (4) Patched ollama.bin (device.go, server.go, discover/runner.go).
- **Inference (Mar 6):** Allocation path works; failure is client timeout ("context canceled") before slow vGPU load completes. Use long client timeout (e.g. 20 min). Previously: Generate failed with `unable to allocate CUDA0 buffer` — allocation path (shim → VGPU-STUB → mediator) is the next step to fix. **Cause:** Guest sends `CUDA_CALL_MEM_ALLOC`; host mediator/executor returns an error (e.g. `ensure_vm_context` or `cuMemAlloc` fails on the physical GPU). See **GPU_MODE_DO_NOT_BREAK.md** section “Inference: unable to allocate CUDA0 buffer” for host-side checks.

### 2. Transmission path (guest side) — **Ready; allocation failing**

- **VGPU-STUB:** Present at `0000:00:05.0`, vendor:device `10de:2331`.  
- **BAR0 (resource0):** Exists at `/sys/bus/pci/devices/0000:00:05.0/resource0`, permissions `-rw-rw-rw-` (666).  
- **Access:** User `ollama` **can** read resource0 (`sudo -u ollama test -r ...` succeeds).  
- **Service:** `ollama.service.d/vgpu.conf` sets `LD_LIBRARY_PATH=/opt/vgpu/lib:...`, `OLLAMA_LLM_LIBRARY=cuda_v12`, no `LD_PRELOAD` in the override.  
- **Conclusion:** The transmission path (shim → VGPU-STUB BAR0 → mediator) is set up correctly on the guest, but it is **never used** because the runner runs in CPU mode and never calls the shim/transport.

### 3. Host mediator

- Not checked from this run (would require host-side commands).  
- Once the guest is in GPU mode and the runner opens BAR0, you can confirm on the host: mediator running, and e.g. `[SOCKET] New connection` when a generate runs.

---

## Next step: get Ollama into GPU mode

Deploy a **patched** `ollama.bin` so that:

1. **Runner env:** Runner gets `LD_LIBRARY_PATH` and `OLLAMA_LIBRARY_PATH` with `/opt/vgpu/lib` first, and **no** `LD_PRELOAD` (patched `llm/server.go`).
2. **Discovery:** CUDA devices are not filtered out by init validation (patched `ml/device.go`: `NeedsInitValidation()` returns false for CUDA).
3. **Loader order:** GPU lib dir (e.g. cuda_v12) is searched before the generic ollama path (patched `discover/runner.go`).

**Option A — VM has Ollama source (e.g. `/home/test-3/ollama`):**

```bash
cd /home/david/Downloads/gpu/phase3
python3 transfer_ollama_go_patches.py
```

This applies patches in memory, transfers the three patched `.go` files to the VM, builds `ollama.bin` on the VM, and installs it.

**Option B — Apply patches and build locally, then copy binary to VM:**

```bash
cd phase3
python3 apply_ollama_vgpu_patches.py   # OLLAMA_SRC=./ollama-src or your ollama repo
# In the patched ollama repo:
go build -o ollama.bin .
scp ollama.bin test-3@<VM_IP>:/tmp/
# On VM: sudo cp /tmp/ollama.bin /usr/local/bin/ollama.bin && sudo systemctl restart ollama
```

After a patched binary is installed and ollama restarted:

- Startup logs should show `inference compute` with `library=CUDA` and non-zero `total_vram`.
- A short generate should create `/tmp/vgpu_ensure_connected_called` and some process should have `resource0` open during the run.

---

## Optional: confirm runner env and manual GPU check (on VM)

```bash
# Restart with debug to see runner env in logs
sudo systemctl set-environment OLLAMA_DEBUG=1
sudo systemctl restart ollama
# Trigger discovery (e.g. curl /api/tags or start a run), then:
sudo journalctl -u ollama -n 50 --no-pager | grep -E 'runner|LD_LIBRARY|OLLAMA_LIBRARY'

# Manual runner test (same env as service): should return GPU in response if shim is used
LD_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama \
OLLAMA_LIBRARY_PATH=/opt/vgpu/lib:/usr/local/lib/ollama/cuda_v12:/usr/local/lib/ollama \
OLLAMA_LLM_LIBRARY=cuda_v12 \
/usr/local/bin/ollama.bin runner --ollama-engine --port 39999 &
sleep 3
curl -s http://127.0.0.1:39999/info
kill %1 2>/dev/null
```

---

## Quick re-check commands (on VM)

```bash
# After restart, confirm GPU mode
sudo journalctl -u ollama -n 30 --no-pager | grep -E "inference compute|library=|total_vram"

# During or right after a generate: runner should hit transport
ls -la /tmp/vgpu_ensure_connected_called
sudo lsof | grep resource0
```
