# Host vGPU stub rebuild — copy, build, install

Use these steps to deploy the updated **vgpu-stub-enhanced.c** to the host and rebuild QEMU. The stub includes: CUDA result applied/IGNORED logging, **__sync_synchronize()** for status visibility, and **STATUS read returning 0x%x** when the guest reads DONE/ERROR (for reply-path diagnosis). so the guest’s MMIO read of REG_STATUS sees DONE (cross-thread visibility in QEMU). Host from `vm_config.py`: **root@10.25.33.10**.

---

## 1. Copy updated stub (and headers) to the host

Run from your **local machine** (where the `gpu` repo lives, e.g. `/home/david/Downloads/gpu`):

```bash
# From the gpu repo root (parent of phase3)
cd /home/david/Downloads/gpu

# Copy the updated vGPU stub source
scp -o StrictHostKeyChecking=no phase3/src/vgpu-stub-enhanced.c root@10.25.33.10:/root/phase3/src/vgpu-stub-enhanced.c

# Copy protocol headers (required for qemu-prepare; may already be present)
scp -o StrictHostKeyChecking=no phase3/include/vgpu_protocol.h root@10.25.33.10:/root/phase3/include/vgpu_protocol.h
scp -o StrictHostKeyChecking=no phase3/include/cuda_protocol.h root@10.25.33.10:/root/phase3/include/cuda_protocol.h
```

If you use a different host or path on the host, replace `root@10.25.33.10` and `/root/phase3` accordingly.

---

## 2. On the host: prepare RPM sources and build QEMU

**SSH to the host**, then run:

```bash
ssh -o StrictHostKeyChecking=no root@10.25.33.10
```

On the host:

```bash
cd /root/phase3

# Stage stub + headers into the RPM SOURCES directory
make qemu-prepare

# Build the QEMU RPM (takes about 30–45 minutes)
# Log is also written to /tmp/qemu-build.log
make qemu-build
```

If `make qemu-prepare` fails with “RPM build directory does not exist”, create it and ensure a `qemu.spec` exists (see **HOST_QEMU_REBUILD_STEPS.md**). If the spec is missing vgpu-stub, run `make qemu-integrate` once to add the source lines.

---

## 3. On the host: install the new QEMU and restart

**Stop any VMs that use the vGPU** (e.g. test-4), then on the host:

```bash
# Install the built RPM (replace path if your RPM is elsewhere)
RPM_FILE=$(ls -1 ~/vgpu-build/rpmbuild/RPMS/x86_64/qemu-*.rpm 2>/dev/null | head -1)
echo "Installing: $RPM_FILE"
rpm -Uvh --nodeps --force "$RPM_FILE"

# Restart the mediator (optional but recommended)
pkill -f mediator_phase3 2>/dev/null || true
sleep 2
cd /root/phase3
nohup ./mediator_phase3 > /tmp/mediator.log 2>&1 &

# Start the VM again (use your VM name or UUID)
# xe vm-start name-label=Test-4
# or: xe vm-start uuid=<VM_UUID>
```

Use your actual VM name or UUID for `xe vm-start`. If you use `vgpu-admin`, re-register and start the VM as you normally do.

---

## 4. Verify stub logging

After the VM is back up and you trigger a generate:

- **QEMU stderr** on the host should show lines like:
  - `[vgpu] vm_id=9: CUDA result applied seq=N status=0 (DONE)` when a response is applied
  - `[vgpu] vm_id=9: CUDA result IGNORED (recv seq=X pending_seq=Y)` when seq does not match

Capture QEMU stderr (e.g. via journalctl for the VM’s QEMU process, or by redirecting the process stderr to a file) to inspect these messages.

---

## One-liner copy (from local repo root)

```bash
cd /home/david/Downloads/gpu && \
scp -o StrictHostKeyChecking=no phase3/src/vgpu-stub-enhanced.c root@10.25.33.10:/root/phase3/src/ && \
scp -o StrictHostKeyChecking=no phase3/include/vgpu_protocol.h phase3/include/cuda_protocol.h root@10.25.33.10:/root/phase3/include/
```

Then SSH to the host and run the **build** and **install** steps above.
