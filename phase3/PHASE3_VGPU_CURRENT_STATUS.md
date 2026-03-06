# Phase 3 vGPU Ollama — current status

*Last updated: Mar 6, 2026*

## Summary

- **Transport path:** Guest → shim → VGPU-STUB → mediator → physical GPU is **working**. Allocations and HtoD copies succeed; model load over this path is **slow** (15–40+ min for ~1.3 GB) because every byte goes over the remoting pipe.
- **Previous failure (before Mar 6 unified-memory fix):** A full deploy using the patient client (~27m 58s on test-3) finished copying the model (mediator showed HtoD progress up to ~1250 MB), then the **llama runner process terminated** with `exit status 2` and Ollama returned **HTTP 500**: `llama runner process has terminated: exit status 2`.
- **Current state (after unified-memory fix):** `cuMemCreate` / `cuMemMap` / `cuMemRelease` in `libvgpu_cuda.c` now back unified memory with **real GPU allocations** via `cuMemAlloc_v2` / `cuMemFree_v2`, and the updated shim is deployed to test-3 with Ollama restarted. A new long generate run (using the patient client) is required to verify that the runner no longer crashes.
- **Design:** Deploy once (first load), then use. After a successful load, the model should stay on the host GPU and inference should be normal until the runner/mediator disconnects.
- **Client requirement:** Use a **patient client** with **no time limit** and a **progress bar** so users don’t think it’s stuck. Standard curl/CLI still time out too early and will abort the load.

---

## Server (VM) configuration

- **Ollama** with vGPU: `LD_LIBRARY_PATH` includes `/opt/vgpu/lib` and CUDA libs; no LD_PRELOAD for the main process (runner loads shims via dlopen).
- **Load timeout:** `OLLAMA_LOAD_TIMEOUT=20m` in the Ollama vGPU service drop-in so the server does not abort the load early.
- **VM (test-3):** `test-3@10.25.33.11` (see `vm_config.py`).

---

## Patient client script

| Item | Value |
|------|--------|
| **Script** | `phase3/ollama_vgpu_generate.py` |
| **On VM** | `/tmp/ollama_vgpu_generate.py` (copy via SCP when updated) |
| **Request timeout** | None (7-day timeout; effectively waits until done or Ctrl+C) |
| **Progress** | Bar + estimated % over **40 minutes** (not real server progress; time-based estimate) |
| **Cancel** | Only Ctrl+C |

**Usage on VM:**
```bash
python3 /tmp/ollama_vgpu_generate.py llama3.2:1b "Say hello."
# Or with custom prompt:
python3 /tmp/ollama_vgpu_generate.py llama3.2:1b "Your prompt"
```

**Copy updated script to VM (from host with phase3):**
```bash
scp -o StrictHostKeyChecking=no phase3/ollama_vgpu_generate.py test-3@10.25.33.11:/tmp/
```

---

## Quick reference — commands on the VM

1. **Run generate (patient client, no time limit):**
   ```bash
   python3 /tmp/ollama_vgpu_generate.py llama3.2:1b "Hi"
   ```

2. **Check Ollama and GPU path:**
   ```bash
   systemctl is-active ollama
   sudo journalctl -u ollama -n 30 --no-pager | grep -E "library=|cuda|GPU|listening"
   ```

3. **List models:**
   ```bash
   curl -s http://localhost:11434/api/tags
   ```

4. **Run generate in background and follow log:**
   ```bash
   nohup python3 /tmp/ollama_vgpu_generate.py llama3.2:1b "Hi" > /tmp/vgpu_gen.log 2>&1 &
   tail -f /tmp/vgpu_gen.log
   ```

---

## Host (mediator)

- **Mediator** logs HtoD progress every 10 MB (e.g. `HtoD progress: 650 MB total (vm=13)`). This is only on the host; the VM client does not see real copy percentage (Ollama does not expose load progress over HTTP).
- Rebuild mediator after changing `phase3/src/cuda_executor.c`: on host, `make mediator_phase3` (see `transfer_cuda_executor_to_host.py` for transfer).

---

## Related docs

- **Direction (goals):** `VGPU_CLIENT_DEPLOYMENT_DIRECTION.md`
- **What was verified:** `VM_INFERENCE_VERIFICATION.md`
- **Transport/GPU status:** `PHASE3_GPU_AND_TRANSPORT_STATUS.md`
