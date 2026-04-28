# Setup checklist: new VM test_4@10.25.33.12 (Phase 3 vGPU for Ollama)

**Purpose:** Run Ollama inside the VM with GPU acceleration by **remoting** all CUDA/CUBLAS calls to the physical GPU on the host. The VM has no physical GPU; it sees a virtual PCI device (VGPU-STUB). Our guest shims intercept CUDA and send calls over the transport to the mediator on the host, which replays them on the real GPU.

---

## Prerequisites

- **vm_config.py** already points at test_4: `VM_USER=test_4`, `VM_HOST=10.25.33.12`, `VM_PASSWORD=Calvin@123`.
- You run scripts from a machine that can SSH to the VM (and optionally to the mediator host).

---

## Order of operations

### A. Host (XCP-ng / dom0) — once per new VM

1. **Register the VM** with the mediator so it gets the vGPU device and a mediator socket:
   ```bash
   # On mediator host (e.g. ssh root@10.25.33.10)
   vgpu-admin register-vm --vm-name=Test-4
   ```
   Use the exact name-label from `xe vm-list` (e.g. `Test-4` or `test-4`).

2. **Start the VM** (if not already running):
   ```bash
   xe vm-start name-label=Test-4
   ```

3. **Mediator** must be running (e.g. `./mediator_phase3` or via systemd). No need to restart for a new VM once registered.

---

### B. VM (test_4) — install and configure step by step

#### Step 1: Basic system and build tools

On the VM (SSH as test_4 or run via `python3 connect_vm.py "..."`):

```bash
sudo apt-get update
sudo apt-get install -y gcc make
```

#### Step 2: Install Ollama

Install Ollama so the VM can run models (CPU or, after we add shims, vGPU):

- **Option A (recommended):** From [ollama.com](https://ollama.com) Linux install (e.g. curl script or .deb).
- **Option B:** If your environment uses a package manager, install the `ollama` package.

Ensure the binary is at `/usr/local/bin/ollama` (or adjust paths below). The vGPU drop-in uses `ExecStart=/usr/local/bin/ollama serve`; if your install provides only `ollama` (no `ollama.bin`), the apply script’s default is updated on the VM to use `ollama` instead of `ollama.bin`.

Verify:

```bash
which ollama
ls /usr/local/bin/ollama.bin 2>/dev/null || ls /usr/local/bin/ollama
```

#### Step 3: Deploy phase3 tree and build guest shims

From your **local** machine (where the phase3 repo lives):

```bash
cd /path/to/phase3
python3 deploy_to_test3.py
```

This script (which reads target from **vm_config.py**, so it uses test_4@10.25.33.12):

1. SCPs the full phase3 tree to the VM (`/home/test_4/phase3`).
2. Runs `make guest` on the VM to build:
   - `libvgpu-cuda.so.1`, `libvgpu-cudart.so`, `libvgpu-nvml.so`
   - `libvgpu-cublas.so.12` (and optionally `libvgpu-cublasLt.so.12` if built).
3. Installs them under `/opt/vgpu/lib` with symlinks:
   - `libcuda.so.1` → `libvgpu-cuda.so.1`
   - `libcudart.so.12` → `libvgpu-cudart.so`
   - `libnvidia-ml.so.1` → `libvgpu-nvml.so`
   - `libcublas.so.12` → `libvgpu-cublas.so.12` (if built).
4. Copies the Ollama systemd drop-in `ollama.service.d/vgpu.conf` (LD_LIBRARY_PATH, OLLAMA_LLM_LIBRARY=cuda_v12, etc.).
5. Restarts the Ollama service.

If `deploy_to_test3.py` fails (e.g. no sshpass and SCP times out), use the chunked transfer scripts (see PHASE3_VGPU_CURRENT_STATUS.md) or copy phase3 manually and run the same commands on the VM.

#### Step 4: udev and BAR access (vGPU device)

The guest must be able to open the VGPU-STUB PCI BARs (e.g. `/sys/bus/pci/devices/.../resource0`, `resource1`). Apply the service override that includes udev rules and the `vgpu-devices.service`:

```bash
cd /path/to/phase3
python3 apply_vgpu_service_no_ldpreload.py
```

This installs:

- `/etc/systemd/system/ollama.service.d/vgpu.conf` (full env: `OLLAMA_LOAD_TIMEOUT=20m`, etc.)
- `/etc/udev/rules.d/99-vgpu-nvidia.rules` (chmod 0666 on the vGPU PCI resources)
- `vgpu-devices.service` (runs at boot to grant BAR access before Ollama starts)

Then on the VM (or via connect_vm):

```bash
sudo systemctl daemon-reload
sudo udevadm control --reload && sudo udevadm trigger
sudo systemctl enable vgpu-devices.service
sudo systemctl start vgpu-devices.service
sudo systemctl restart ollama
```

#### Step 5: Verify Ollama is using the vGPU path

On the VM:

```bash
systemctl is-active ollama
sudo journalctl -u ollama -n 50 --no-pager | grep -E "library=|cuda|GPU|listening|total_vram"
```

You should see `library=cuda` or similar and the service listening.

#### Step 6: Pull a model (optional)

```bash
ollama pull llama3.2:1b
```

#### Step 7: Test generate

Use the patient client (long timeout) or a short curl test:

```bash
# From your machine (VM must be reachable):
python3 connect_vm.py "curl -s -X POST http://127.0.0.1:11434/api/generate -d '{\"model\":\"llama3.2:1b\",\"prompt\":\"Hi\",\"stream\":false}' 2>&1 | head -5"
# Or copy and run on the VM:
python3 /tmp/ollama_vgpu_generate.py llama3.2:1b "Hi"
```

First load can take 15–40+ minutes (model is streamed over the remoting pipe). Use `OLLAMA_LOAD_TIMEOUT=20m` so the server does not abort the load early.

---

## Summary table

| Step | Where   | Action |
|------|---------|--------|
| A.1  | Host    | `vgpu-admin register-vm --vm-name=Test-4` |
| A.2  | Host    | `xe vm-start name-label=Test-4` |
| B.1  | VM      | Install gcc, make |
| B.2  | VM      | Install Ollama |
| B.3  | Local   | `python3 deploy_to_test3.py` (SCP phase3, make guest, install shims, drop-in, restart ollama) |
| B.4  | Local   | `python3 apply_vgpu_service_no_ldpreload.py` (udev + vgpu-devices.service); then on VM: daemon-reload, udev, start vgpu-devices, restart ollama |
| B.5  | VM      | Verify `journalctl -u ollama` shows library=cuda / GPU |
| B.6  | VM      | `ollama pull llama3.2:1b` (optional) |
| B.7  | Local/VM| Test generate (patient client or curl) |

---

## If something fails

- **Connection refused / no route to host:** Run from a machine that can SSH to 10.25.33.12 (see CONNECT_VM_README.md).
- **Ollama reports library=cpu:** Check LD_LIBRARY_PATH in the service and that `/opt/vgpu/lib` contains the shims and symlinks; check journalctl for errors.
- **Generate fails with exit status 2:** See RUNNER_DIAGNOSTIC_README.md; capture runner stderr ("CUDA error: ...") to identify the failing call.
- **Host mediator:** Ensure mediator is running and the VM is registered; check `/tmp/mediator.log` on the host.
