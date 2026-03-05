# Host-side verification (GPU vs CPU mode)

Run these on the **host** (where the mediator and physical GPU run).

## 1. Check mediator is running and processing

```bash
# Mediator process
ps aux | grep mediator

# If using a log/socket: show recent activity (adjust path to your setup)
journalctl -u mediator_phase3 -n 30 --no-pager   # if it's a systemd service
# or
tail -50 /var/log/mediator.log   # if logging to file
```

## 2. Check GPU memory and utilization during a model run

**Before** starting a model run on the VM, on the host run:

```bash
nvidia-smi
```

Note `memory.used` and `utilization.gpu` (e.g. 0% if idle).

**While** a model is running on the VM (e.g. `ollama run llama3.2:1b '...'`), in another terminal on the host run:

```bash
watch -n 2 nvidia-smi
```

- If **GPU mode** is active: `memory.used` should increase (e.g. model weights) and `utilization.gpu` may go above 0% during inference.
- If **CPU mode** is active: `memory.used` and utilization may stay unchanged.

## 3. Check mediator admin socket and stats (phase3)

Phase3 mediator uses an **admin socket**, not a text "stats" pipe:

- **Admin socket path:** `/var/vgpu/admin.sock` (defined in `include/vgpu_protocol.h`).
- **No** `/run/mediator_phase3.sock` — that path was an example only.
- **No** `/var/log/mediator.log` by default — mediator typically logs to stdout/stderr (or journalctl if run as a service).

**Use the vgpu-admin CLI** (from the phase3 build) to query the mediator:

```bash
# From the host, in the phase3 build directory (or wherever vgpu-admin is installed)
./vgpu-admin show-metrics
# or
./vgpu-admin show-health
# or
./vgpu-admin show-connections
```

If the mediator is not run as a systemd unit, its stdout/stderr may be in a terminal or a log file you configured. Check how you start `mediator_phase3` to see where its output goes.

- If you see **processed** counts or **cuMemAlloc** / **cuLaunchKernel** activity **while** the VM is running a model, the remoting path is in use.
- If there is no such activity during a run, inference is likely on CPU.

## 4. Summary

| What to check | GPU mode likely | CPU mode likely |
|---------------|-----------------|-----------------|
| `nvidia-smi` during VM inference | memory.used and/or utilization increase | No change |
| Mediator / vgpu-admin stats | New cuMemAlloc / kernel calls during run | No new GPU calls during run |

Current VM status: logs show **id=cpu** and **total_vram="0 B"**, so Ollama is currently selecting **CPU** for inference. Until discovery reports a GPU with VRAM, host-side checks will show no GPU activity during runs.

---

## 5. Interpreting your nvidia-smi output

If you see:

- **Processes: `./mediator_phase3` using ~4550 MiB**  
  → The mediator is running and has allocated GPU memory (e.g. from CUDA remoting or its own buffers). So the host/GPU side is up.

- **GPU-Util 0%**  
  → No kernel execution right now. With Ollama in CPU mode on the VM, you would not see utilization from inference; utilization would rise when the VM runs in GPU mode and sends kernel launches.
