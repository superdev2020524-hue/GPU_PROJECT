# Server 2 VM lockdown blocker and resolution - Apr 17, 2026

*Scope: Server 2 only. Authoritative path for this finding is `server2/phase3/`.*

---

## Mission status

- **Lane:** Server 2 fast-path general GPU bring-up
- **Plan A canary state:** unverified / not applicable to this Server 2 guest-compute track
- **Tracked error:** `S2-E1` - guest kernel lockdown blocked direct PCI BAR access, so real CUDA transport could not start
- **Current state:** `S2-E1` resolved on the current VM by disabling VM Secure Boot at the **host platform** layer
- **Last proven checkpoint:** full guest bootstrap and verification completed successfully on `root@10.25.33.21`

---

## Final working state

On VM `root@10.25.33.21` after the host-side Secure Boot change:

1. `lspci` shows the vGPU at `00:05.0` as:
   - `NVIDIA Corporation HEXACORE vH100 CAP`
2. Guest shim libraries are installed and active:
   - `/usr/lib64/libvgpu-cuda.so`
   - `/usr/lib64/libvgpu-nvml.so`
   - `/usr/lib64/libvgpu-cudart.so`
3. Guest firmware / kernel state is now compatible with transport:
   - `mokutil --sb-state -> SecureBoot disabled`
   - `/sys/kernel/security/lockdown -> [none] integrity confidentiality`
4. Direct MMIO access is working:
   - BAR0 `mmap()` succeeds on `/sys/bus/pci/devices/0000:00:05.0/resource0`
5. Real compute path is working:
   - `cuInit() -> 0`
   - `cuMemAlloc_v2() -> 0`
   - `cuMemcpyHtoD_v2() -> 0`
   - `cuMemcpyDtoH_v2() -> 0`
   - 8-byte round-trip payload matches exactly
   - `cuMemFree_v2() -> 0`
6. Application-level kernel path is now working too:
   - `cuModuleLoadData() -> 0`
   - `cuModuleGetFunction() -> 0`
   - `cuLaunchKernel() -> 0`
   - `run_server2_ptx_kernel_smoke.py` returns `result a=123 b=456 c=579`
7. Larger application-style workload is also working:
   - `run_server2_ptx_vector_add_workload.py` completed successfully on `10.25.33.21`
   - workload size: `65536` elements, `262144` bytes per vector
   - sample outputs and full checksum matched exactly
   - `kernel_elapsed_ms=3.947`
   - final signal: `VECTOR_ADD_WORKLOAD_OK`
8. Identity path still matches the mission goal:
   - `cuDeviceGetName -> HEXACORE vH100 CAP`
   - `nvmlDeviceGetName -> HEXACORE vH100 CAP`
   - `cudaGetDeviceCount -> 1`
9. Full verifier outcome:
   - `setup_server2_general_gpu_vm.py` completed successfully
   - the built-in verifier now checks BAR0 access, CUDA allocation, and a small `HtoD` / `DtoH` round-trip
10. Ollama remains intentionally suspended:
   - `systemctl is-active ollama -> inactive`

This satisfies the current Server 2 gate for `lspci` branding plus general GPU application bring-up on the test VM.

---

## Historical blocker (`S2-E1`)

Before the host-side fix, real compute was blocked **before** the mediator/transport path could be used normally.

### Exact bounded repro before resolution

On VM `10.25.33.21` before the host change:

1. `mokutil --sb-state` -> `SecureBoot enabled`
2. `cat /sys/kernel/security/lockdown` -> `none [integrity] confidentiality`
3. Direct BAR test:
   - open `/sys/bus/pci/devices/0000:00:05.0/resource0` succeeded
   - `mmap(... resource0 ...)` failed with `Operation not permitted`
4. Direct CUDA allocation test:
   - `cuInit()` returned `0`
   - `cuMemAlloc_v2()` returned `100`
   - shim logs showed `Cannot mmap BAR0: Operation not permitted`

### Why `S2-E1` was the real blocker

The failure happened at guest BAR mapping time. The Server 2 transport requires direct PCI BAR access for BAR0/BAR1 MMIO, so lockdown prevented real GPU work even though the synthetic device identity was already correct.

This was not just a Unix permission problem:

- `/sys/bus/pci/devices/0000:00:05.0/resource0` and `resource1` were already `0666`
- VM root could open the files
- PCI BARs were present and enabled in `lspci -vv`

This was also not the primary identity problem:

- `lspci`, CUDA name, and NVML name already matched the required `HEXACORE` branding

---

## What did not resolve it

### Guest-side shim / MokManager route on this VM

An in-guest disable request was staged with:

1. `mokutil --disable-validation`
2. password entry succeeded
3. EFI request variable appeared:
   - `MokSB-605dab50-e046-4300-abb6-3dd810dd8b23`
4. VM was rebooted and the MokManager VGA menu was reached and driven from the host console

### Outcome of that route

That path did **not** clear the runtime blocker on this VM:

- the VM rebooted normally
- but it still came back with `SecureBoot enabled`
- `/sys/kernel/security/lockdown` still showed `integrity`
- BAR0 `mmap()` still failed with `EPERM`
- CUDA allocation still failed (`cuMemAlloc_v2 rc=100`)

Interpretation:

- the guest-side request could be staged and the boot UI could be reached
- but on this XCP-ng VM, that path did not actually deliver the required unlocked boot state

So the durable fix for Server 2 was **not** further guest-side UI driving. The durable fix was to change the VM's Secure Boot setting at the host platform level.

---

## Resolution on the current VM

The working fix was applied on host `10.25.33.20` for VM UUID `5b9acc4b-d62b-6dc6-576f-82175e87fc2b` (`Ubuntu-VM-1`).

### Host proof before the fix

- `xe vm-param-get uuid=5b9acc4b-d62b-6dc6-576f-82175e87fc2b param-name=platform`
  showed `secureboot: true`
- `xe vm-param-get uuid=... param-name=HVM-boot-params param-key=firmware`
  showed `uefi`

### Host-side fix sequence

1. Cleanly shut down the VM:
   - `xe vm-shutdown uuid=5b9acc4b-d62b-6dc6-576f-82175e87fc2b`
2. Disable Secure Boot in the VM platform settings:
   - `xe vm-param-set uuid=5b9acc4b-d62b-6dc6-576f-82175e87fc2b platform:secureboot=false`
3. Start the VM again:
   - `xe vm-start uuid=5b9acc4b-d62b-6dc6-576f-82175e87fc2b`

### Guest proof after the fix

- `mokutil --sb-state` -> `SecureBoot disabled`
- `/sys/kernel/security/lockdown` -> `[none] integrity confidentiality`
- BAR0 `mmap()` succeeds
- `cuMemAlloc_v2 rc=0`
- `cuMemcpyHtoD_v2 rc=0`
- `cuMemcpyDtoH_v2 rc=0`
- round-trip payload returned unchanged
- `setup_server2_general_gpu_vm.py` completes successfully

This establishes that the **host platform Secure Boot flag** was the effective fix for `S2-E1`.

### Post-resolution host follow-up for kernel launch

After `S2-E1` was cleared, a stricter application-style test exposed one more host-side issue:

- `cuModuleLoadData()` succeeded
- `cuModuleGetFunction()` succeeded
- first `cuLaunchKernel()` failed on the host with `rc=400` (`CUDA_ERROR_INVALID_HANDLE`)

Root cause on Server 2:

- host `cuda_executor.c` loaded modules and resolved functions in `exec->primary_ctx`
- but `CUDA_CALL_LAUNCH_KERNEL` switched to the per-VM context via `ensure_vm_context(exec, vm)`
- that made the returned `CUfunction` handle invalid at launch time

Verified host fix now applied on `10.25.33.20`:

- update `server2/phase3/src/cuda_executor.c` so `CUDA_CALL_LAUNCH_KERNEL` uses `cuCtxSetCurrent(exec->primary_ctx)`
- rebuild `mediator_phase3` on the host
- replace `/usr/local/bin/mediator_phase3`
- restart the mediator

Proof after that host fix:

- `run_server2_ptx_kernel_smoke.py` on the workstation completed successfully against `10.25.33.21`
- VM output showed:
  - `cuModuleLoadData rc=0`
  - `cuModuleGetFunction rc=0`
  - `cuLaunchKernel rc=0`
  - `result a=123 b=456 c=579`
  - `PTX_KERNEL_TEST_OK`
- `run_server2_ptx_vector_add_workload.py` then completed successfully against the same VM
  - `elements=65536`
  - `bytes_per_vector=262144`
  - `checksum expected=6442418176 actual=6442418176`
  - `kernel_elapsed_ms=3.947`
  - `VECTOR_ADD_WORKLOAD_OK`

---

## Reproducible deployment rule

For future Server 2 deployments, the minimum host+guest sequence should be:

1. Create a new UEFI VM
2. On the host, ensure Secure Boot is **disabled** for that VM:
   - `xe vm-param-set uuid=<vm-uuid> platform:secureboot=false`
3. On the host, attach vGPU using the Server 2 `vgpu-admin register-vm` path
4. On the guest over SSH:
   - run `fix_pci_ids_vm.py`
   - run `setup_server2_general_gpu_vm.py`
   - optionally run `run_server2_ptx_kernel_smoke.py` from the workstation for a real kernel-launch proof
   - optionally run `run_server2_ptx_vector_add_workload.py` from the workstation for a larger multi-thread workload proof
5. Verify:
   - `lspci` shows `HEXACORE`
   - CUDA/NVML names show `HEXACORE`
   - BAR0 `mmap()` succeeds
   - `cuMemAlloc_v2` succeeds
   - a small CUDA `HtoD` / `DtoH` round-trip succeeds
   - optional PTX smoke returns `PTX_KERNEL_TEST_OK` with `result a=123 b=456 c=579`
   - optional vector-add workload returns `VECTOR_ADD_WORKLOAD_OK`

If step 5 fails with `Cannot mmap BAR0: Operation not permitted`, check the **host VM platform setting** first and confirm Secure Boot was actually disabled for that VM before investigating deeper guest transport issues.

---

## Current conclusion

The current VM `10.25.33.21` is now **functionally successful** for the Server 2 objective:

- **Branding goal met:** `lspci` shows `HEXACORE`
- **General GPU path met:** BAR mapping, CUDA allocation, NVML, CUDART verification, PTX kernel launch, and a larger vector-add workload all succeed
- **Reproducible host fix identified:** `platform:secureboot=false`
- **Post-resolution app-validation note:** a fresh upstream `ollama` `0.21.0` install on this VM already discovers `library=CUDA` / `compute=9.0` / `HEXACORE vH100 CAP` and offloads `qwen2.5:0.5b` layers to GPU, but a short `/api/generate` currently times out at runner start `progress 0.27` after about 5 minutes. Treat that as a separate Ollama-specific candidate, not a reopening of `S2-E1`.

`S2-E1` should remain in the registry as a resolved historical blocker, not as the current active error on this VM.

## Postscript - passthrough pivot candidate

After the mediated-path proofs above, we also tested a direct real-GPU
passthrough pivot on Host 2 because the final Server 2 objective is "works
perfectly for real applications", not "mediator only".

What was confirmed:

- Host GPU record: `0000:81:00.0`
- Xen PCI record: `e3dfe1bb-1e88-655a-8031-06b22eea9433`
- `xe pci-disable-dom0-access ...` succeeded and, after host reboot,
  `xl pci-assignable-list` showed `0000:81:00.0`
- VM `5b9acc4b-d62b-6dc6-576f-82175e87fc2b` was reconfigured for
  `other-config:pci=0/0000:81:00.0`

Current blocker from that pivot:

- `xe vm-start` failed before guest boot with
  `Cannot_add(0000:81:00.0, Xenctrlext.Unix_error(25, "38: Function not implemented"))`

Interpretation:

- This does **not** reopen `S2-E1`
- This does **not** disprove the prior mediated-path compute proofs
- This is a separate **host-platform passthrough candidate** (`S2-P1`)
  on the current `XCP-ng 8.3.0` stack
