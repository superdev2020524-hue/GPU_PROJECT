# Rebuild QEMU with updated vGPU stub (on the host)

Run these **on the XCP-ng host** (e.g. `ssh root@10.25.33.10`). You must already have the RPM build tree and `qemu.spec` set up (e.g. `~/vgpu-build/rpmbuild/`).

---

## 1. Prepare RPM sources (copy stub + headers into SOURCES)

From the phase3 directory that already has the new `src/vgpu-stub-enhanced.c`:

```bash
cd /root/phase3
make qemu-prepare
```

This copies `src/vgpu-stub-enhanced.c` → `~/vgpu-build/rpmbuild/SOURCES/vgpu-stub.c` and the protocol headers.

---

## 2. Build the QEMU RPM (30–45 minutes)

```bash
cd /root/phase3
make qemu-build
```

Log is also written to `/tmp/qemu-build.log` on the host. If it fails, check that file.

---

## 3. Install the new QEMU RPM

VMs using the vGPU should already be stopped (you ran `vgpu-admin remove-vm`). Then:

```bash
RPM_FILE=$(ls -1 ~/vgpu-build/rpmbuild/RPMS/x86_64/qemu-*.rpm 2>/dev/null | head -1)
echo "Installing: $RPM_FILE"
rpm -Uvh --nodeps --force "$RPM_FILE"
```

---

## 4. Restart mediator, reattach vGPU, start VM

```bash
pkill -f mediator_phase3 2>/dev/null || true
sleep 2
cd /root/phase3
./mediator_phase3 2>/tmp/mediator.log &

vgpu-admin register-vm --vm-name=Test-3
xe vm-start name-label=Test-3
```

(Use `--pool=A`, `--priority=medium`, etc. if you use them. Use `xe vm-list` to get the UUID if `name-label=Test-3` is not accepted.)

---

## If `make qemu-prepare` or `make qemu-build` fails

- **"RPM build directory does not exist"**  
  Create it and put a `qemu.spec` in place (see Makefile variables `RPM_BUILD`, `QEMU_SPEC`). Your XCP-ng/QEMU build docs should describe this.

- **"QEMU spec file not found"**  
  You need a `qemu.spec` in `~/vgpu-build/rpmbuild/SPECS/` that references `vgpu-stub.c` (and the protocol headers). The Makefile target `qemu-integrate` can add the Source3/Source4 lines if they are missing.

- **RPM build fails**  
  Inspect `/tmp/qemu-build.log` on the host for the actual error.
