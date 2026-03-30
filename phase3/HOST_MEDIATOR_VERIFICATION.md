# Host Mediator Verification (Read-Only)

**Date:** 2025-03-18  
**Method:** `phase3/connect_host.py` to run commands on `root@10.25.33.10`.

## Summary: **Mediator is operating correctly**

### 1. Process

- **mediator_phase3** is running: PID **3654548** (since Mar 17).
- `ps aux | grep mediator` → `./mediator_phase3`

### 2. Logs (`/tmp/mediator.log`)

- **Location:** `/tmp/mediator.log` (≈2.4 MB, updated Mar 18 14:52).
- **Connections:** 2 server sockets — `root-232`, `root-235` (two VMs).
- **CUDA traffic:** `call_id=0x32` (HtoD) for **vm_id=9**; results sent back as `result.status=0 -> stub sets DONE`.
- **HtoD progress (cuda-executor):**  
  `[cuda-executor] HtoD progress: 274 MB total (vm=9)` … up to **315 MB total (vm=9)** in the tail — host is performing real host-to-device copies every ~10 MB.
- **Stats (from log tail):** Total processed **974**, Pool A 974, GPU temp 49°C, GPU util 0%.

### 3. Physical GPU (nvidia-smi on host)

- **Memory used:** 2071 MiB / 81559 MiB.
- **Compute process:** PID **3654548** (mediator_phase3) using **2052 MiB** — mediator holds the CUDA context and has model data on the physical GPU.
- **Utilization:** 0% (idle between copies).  
- **Temperature:** 49°C.

### 4. Conclusion

- Mediator is running and writing to `/tmp/mediator.log`.
- It is receiving CUDA RPCs from at least one VM (vm_id=9), dispatching **HtoD (0x32)** to the executor, and returning DONE.
- The executor is doing real `cuMemcpyHtoD` on the host (HtoD progress lines).
- The mediator process is the one using GPU memory (~2 GB), consistent with model load in progress.

**Note:** vm_id=9 corresponds to one of the two VMs (root-232 or root-235). The VM you use (e.g. test-4@10.25.33.12) may have a different vm_id on this host; the docs’ vm_id=111 was for a different setup.
