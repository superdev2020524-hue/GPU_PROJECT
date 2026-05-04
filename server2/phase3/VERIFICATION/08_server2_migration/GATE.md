# Gate - Milestone 08 Server 2 Migration

## Lane

Server 2 migration / product demonstration path.

## Trust And Safety Boundary

- Work only under `server2/` and `server2/phase3/`.
- Do not edit root `phase3/` for Server 2 work.
- Do not run disruptive host actions such as VM shutdown, passthrough attach,
  host reboot, or GPU reassignment unless explicitly approved for the current
  step.
- Preserve Server 2 stable demo mode until the chosen cutover mode is confirmed.

## Required Gate Cases

1. **Connectivity gate**
   - SSH to Server 2 host `10.25.33.20`.
   - SSH to Server 2 target VM `10.25.33.21`.
   - If unreachable, stop runtime work and record `M08-E1`.

2. **Host inventory gate** (read-only)
   - Confirm host identity.
   - Confirm real GPU BDF `0000:81:00.0` or updated equivalent.
   - Confirm target VM UUID/name/power state.
   - Confirm current passthrough mapping and Secure Boot state.

3. **Guest inventory gate** (read-only)
   - Confirm `lspci` GPU presentation.
   - Confirm `nvidia-smi`.
   - Confirm whether mediated leftovers exist:
     `/usr/lib64/libvgpu-*`, `/etc/profile.d/vgpu-cuda.sh`,
     `/etc/ld.so.conf.d/vgpu-lib64.conf`.

4. **Deployment checklist gate**
   - Document current recommended path:
     `attach_passthrough_vm.sh <vm-uuid>` for clean VMs.
   - Document when to run `fix_pci_ids_vm.py`.
   - Document when to run `clean_passthrough_vm.py`.

5. **Rollback checklist gate**
   - Document how to detach or move the passthrough GPU.
   - Document how to restore the prior stable VM mode.
   - Document which commands are disruptive.

6. **Client demo gate**
   - Define repeatable commands for:
     `lspci`, `nvidia-smi`, Ollama, PyTorch, and optional TensorFlow.
   - No hidden manual state may be required.

## Pass Criteria

- Server 2 host and VM are reachable or an approved alternate access path is
  documented.
- Current Server 2 mode is classified with fresh evidence.
- Deployment and rollback paths are written and bounded.
- Client-facing demo checklist exists and matches the chosen Server 2 mode.
- No disruptive action is performed without explicit approval.

## Fail Criteria

- Server 2 host/VM are unreachable and no alternate access path exists.
- Current GPU mode cannot be classified.
- Deployment path depends on undocumented manual state.
- Rollback path is missing.
- Server 1 registry or runtime is modified as part of Server 2 work.
