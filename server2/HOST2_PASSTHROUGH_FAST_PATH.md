# Host 2 passthrough fast path

This is the fast path if the Server 2 final goal is:

- `lspci` in the guest shows `HEXACORE`
- real GPU applications work like a normal attached GPU
- the deployment path does not depend on the unfinished mediator stack

For a beginner-facing handoff, see:

- `server2/HOST_DISTRIBUTION_MANAGER_GUIDE.md`

## Current confirmed host facts

- Host: `10.25.33.20`
- Real GPU on host: `0000:81:00.0`
- Xen PCI record for that GPU: `e3dfe1bb-1e88-655a-8031-06b22eea9433`
- Current target VM: `Ubuntu-VM-1`
- Current target VM UUID: `5b9acc4b-d62b-6dc6-576f-82175e87fc2b`

## Current live state

The current VM is now running with the real H100 exposed in the guest:

```bash
lspci -nn | grep -i '10de:2321\|10de:2331'
```

Observed result:

- `00:08.0 3D controller [0302]: NVIDIA Corporation HEXACORE vH100 CAP [10de:2321]`

Meaning:

- host-side passthrough is active on the current VM
- the remaining failure was in the guest, where old mediated-path shims were
  still shadowing the real NVIDIA driver

Current interpretation:

- the passthrough attach path is usable on the current setup
- the required guest-side action is to remove the old mediated-path library and
  linker overrides, then apply guest branding for the real device ID

## Important meaning of this path

PCI passthrough gives the guest the real NVIDIA GPU behavior.

That means:

- GPU applications should behave like a normal real-GPU VM
- `lspci` naming can still be branded as `HEXACORE` by patching the guest
  `pci.ids`
- raw numeric PCI IDs remain the real NVIDIA IDs

## Host steps

### 1. Hide the GPU from dom0

```bash
xe pci-disable-dom0-access uuid=e3dfe1bb-1e88-655a-8031-06b22eea9433
```

### 2. Reboot the host

This is the disruptive boundary for the passthrough cutover.

After reboot, verify the GPU is assignable:

```bash
xl pci-assignable-list
```

Expected:

- `0000:81:00.0` appears in the assignable list

### 3. Stop the target VM

```bash
xe vm-shutdown uuid=5b9acc4b-d62b-6dc6-576f-82175e87fc2b
```

### 4. Remove the old custom vgpu device-model args

```bash
xe vm-param-remove uuid=5b9acc4b-d62b-6dc6-576f-82175e87fc2b \
  param-name=platform param-key=device-model-args
```

### 5. Attach the real GPU to the VM

```bash
xe vm-param-set uuid=5b9acc4b-d62b-6dc6-576f-82175e87fc2b \
  other-config:pci=0/0000:81:00.0
```

If later you need to add more passthrough devices, append them in the same
comma-separated `other-config:pci=` value.

### 6. Keep Secure Boot disabled

```bash
xe vm-param-set uuid=5b9acc4b-d62b-6dc6-576f-82175e87fc2b \
  platform:secureboot=false
```

### 7. Start the VM

```bash
xe vm-start uuid=5b9acc4b-d62b-6dc6-576f-82175e87fc2b
```

## Current validated deployment path

Assume the script is already saved on the host at:

```bash
~/attach_passthrough_vm.sh
```

For a **brand-new VM**, the default host-side path is:

```bash
chmod +x ~/attach_passthrough_vm.sh
~/attach_passthrough_vm.sh <vm-uuid>
```

This script now:

- stops the target VM if needed
- stops any other VM currently using the GPU
- removes the GPU mapping from that other VM
- assigns the GPU to the target VM
- keeps Secure Boot disabled
- starts the target VM

Then, from the workstation registry, apply only the guest branding step if you
want `lspci` to show `HEXACORE`:

```bash
cd /home/david/Downloads/gpu/server2/phase3
python3 fix_pci_ids_vm.py
```

That is enough for a clean new VM because it does **not** contain the old
mediated-path shim files.

For an **older mixed VM** that previously used the mediated path, use:

```bash
cd /home/david/Downloads/gpu/server2/phase3
python3 clean_passthrough_vm.py
```

Use `clean_passthrough_vm.py` only when the guest still contains leftovers such
as `/usr/lib64/libvgpu-*.so`, `/etc/profile.d/vgpu-cuda.sh`, or
`/etc/ld.so.conf.d/vgpu-lib64.conf`.

## Current verified result

On the cleaned current VM:

- plain `lspci` shows `NVIDIA Corporation HEXACORE vH100 CAP`
- `lspci -nn` shows the real numeric ID `[10de:2321]`
- `nvidia-smi` works and reports `NVIDIA H100 NVL`
- Ollama discovery reports `library=CUDA compute=9.0`
- `POST /api/generate` using `qwen2.5:0.5b` returns `OK.`

## Historical note

Earlier in the session, passthrough was treated as blocked because one host-side
attempt failed with:

```text
xenopsd internal error:
Cannot_add(0000:81:00.0, Xenctrlext.Unix_error(25, "38: Function not implemented"))
```

Current live evidence supersedes that as the primary blocker for this VM,
because the guest now sees and successfully uses the real passthrough H100.

## Rollback note

If you decide to return this host to the mediated/dom0-owned path, the
recovery boundary is:

```bash
xe pci-enable-dom0-access uuid=e3dfe1bb-1e88-655a-8031-06b22eea9433
```

Then reboot the host again before expecting dom0/NVIDIA-driver ownership to
return.

## Guest steps

### 1. Install the normal NVIDIA guest driver path

Do not bootstrap the mediator guest shim stack for this final path.

### 2. For a fresh VM, brand guest-visible names as `HEXACORE`

From the workstation registry:

```bash
cd /home/david/Downloads/gpu/server2/phase3
python3 fix_pci_ids_vm.py
```

The branding helper now does three guest-only branding steps:

- `2321` = real passthrough H100
- `2331` = old vgpu-stub identity
- installs a thin `/usr/bin/nvidia-smi` wrapper so the normal
  `nvidia-smi` command displays `HEXACORE vH100 CAP` without changing the
  real NVIDIA driver or CUDA/NVML runtime underneath
- builds and enables a tiny `/etc/ld.so.preload` CUDA/NVML name-only shim so
  user-space frameworks such as TensorFlow, PyTorch, and Ollama can report
  `HEXACORE vH100 CAP` while still using the real passthrough compute path

### 2b. Only if this is an old mixed VM, clean old shim leftovers first

```bash
cd /home/david/Downloads/gpu/server2/phase3
python3 clean_passthrough_vm.py
```

### 3. Verify

In the guest:

```bash
lspci | grep -i 'HEXACORE\|NVIDIA'
lspci -nn | grep -i '10de:2321\|10de:2331'
nvidia-smi
```

Expected:

- plain `lspci` can show `HEXACORE vH100 CAP`
- `lspci -nn` still shows the real numeric NVIDIA ID
- `nvidia-smi` shows `HEXACORE vH100 CAP` through the wrapper path
- `HEXACORE_NVIDIA_SMI_BYPASS=1 /usr/bin/nvidia-smi` still shows the
  real hardware name for debugging
- TensorFlow can log `name: HEXACORE vH100 CAP` and still execute on `GPU:0`
- PyTorch can report `torch.cuda.get_device_name(0) -> HEXACORE vH100 CAP`
  and still execute on `cuda:0`
- Ollama can discover `library=CUDA compute=9.0 description="HEXACORE vH100 CAP"`
  and complete a short generate

## Why this is the preferred final path

The custom mediator route already proved:

- guest branding
- guest CUDA naming
- small and medium CUDA workloads

But it still fails a real application-level workload (`Ollama`) at runner start.

For the Server 2 final objective, passthrough is the cleaner production path
because application compatibility comes from the real GPU exposure method, not
from an incomplete CUDA remoting stack.
