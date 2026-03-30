# Host 2 (server2) — dom0 bring-up notes

**Host:** `xcp-ng-sfgagrpq` · **IP:** `10.25.33.20` · **OS:** XCP-ng 8.3  

This is the path we actually used to get QEMU (`vgpu-cuda`), the Phase 3 mediator, and `vgpu-admin` running on the second dom0. It’s not theory — it’s what worked and what tripped us up.

---

## Why not build QEMU on `/`

Root on this box was ~18 G with single-digit GB free. `rpmbuild` unpacks and compiles QEMU under `BUILD/`; that eats space fast. We parked the whole RPM tree on **`/mnt/cudawork`** (NVMe, ~1.8 T) and pointed **`RPM_BUILD`** there. Same idea if you add another disk: big filesystem for `SOURCES` + `BUILD`, keep `/root/phase3` on the small root volume — it’s only sources and the Makefile.

---

## Git LFS

XCP-ng’s repos don’t ship `git-lfs`. `yum install git-lfs` won’t find it. Pull the official Linux amd64 tarball from GitHub (`git-lfs` releases), unpack, run `./install.sh` — binary lands in `/usr/local/bin`. Then `git lfs install` once.

Without LFS, `SOURCES/qemu-*.tar.gz` in the clone is a pointer file and `rpmbuild` dies with “not a tar archive” or similar garbage during `%prep`.

---

## QEMU RPM (xcp-ng fork, not upstream tarballs)

1. Clone `https://github.com/xcp-ng-rpms/qemu` somewhere on the big disk (we used `/mnt/cudawork/vgpu/qemu-xcpng-git`).
2. `git lfs pull` in that clone so `SOURCES/` are real archives.
3. Match what’s installed: `rpm -qa | grep '^qemu'` — we had `qemu-4.2.1-5.2.12.1.xcpng8.3` before the rebuild; the spec from git tracked a slightly newer release (`5.2.17.1`) — that’s fine, it’s still the same XCP-ng 8.3 / 4.2.1 line.

Copy into your topdir:

```bash
export RPM_BUILD=/mnt/cudawork/vgpu/rpmbuild   # example
cp -a SPECS/qemu.spec  "$RPM_BUILD/SPECS/"
cp -a SOURCES/*        "$RPM_BUILD/SOURCES/"
```

Phase 3 tree lives at `/root/phase3` (sync from your dev machine — see below). Then:

```bash
export RPM_BUILD=/mnt/cudawork/vgpu/rpmbuild
cd /root/phase3
make qemu-check
make qemu-prepare
make qemu-build
```

First time `make qemu-build` failed on missing **BuildRequires** (`python3-devel`, `glib2-devel`, `pixman-devel`, Xen *-devel packages, etc.). Install what `rpmbuild` complains about — `yum install` the list, re-run. Expect 30–45+ minutes of compile.

Install the produced RPM (stop VMs using the device model first if you’re being careful):

```bash
rpm -Uvh --nodeps --force "$RPM_BUILD/RPMS/x86_64"/qemu-4.2.1-*.xcpng8.3.x86_64.rpm
```

Sanity check:

```bash
/usr/lib64/xen/bin/qemu-system-i386 -device help 2>/dev/null | grep -i vgpu
```

You want **`vgpu-cuda`** on **PCI** with the “CUDA Remoting” description. There’s also a **`vgpu`** device on the **System** bus — different thing; for our stack you care about **`vgpu-cuda`**.

Stub compile may warn on ignored `write()` return in `vgpu-stub.c` — harmless for now.

---

## Getting `/root/phase3` onto the host

Common mistake: running `rsync` **from the dom0** with a source path like `/home/david/Downloads/...` — that path exists on your **workstation**, not on the server. Either:

- run `rsync` / `scp` **from the machine that has the repo**, toward `root@10.25.33.20:/root/phase3/`, or  
- tarball Phase 3 and `scp` the tarball, then unpack under `/root/phase3`.

The `server2/phase3/` tree in this repo is meant to be that full mirror; refresh it with `PHASE3_SYNC.md` when upstream changes.

---

## Mediator and `vgpu-admin`

Dependencies we needed:

```bash
yum install -y sqlite-devel
```

CUDA on this host lived under `/usr/local/cuda-12.2` (not always the same as `/usr/local/cuda`). Build with:

```bash
cd /root/phase3
CUDA_PATH=/usr/local/cuda-12.2 make host
```

Install system-wide (optional but cleaner than only running from the build dir):

```bash
make install
```

DB: `init_db.sql` creates the schema including Phase 3 columns on `vms`. If you then run `migrate_phase3.sql` on that same fresh DB, SQLite may throw **duplicate column** errors — the columns already exist. Check with `sqlite3 /etc/vgpu/vgpu_config.db ".schema"`; if `vms` already has `weight`, `max_jobs_per_sec`, etc., ignore the migration noise or skip migrate.

Start mediator:

```bash
pkill -f mediator_phase3 2>/dev/null; sleep 2
nohup /usr/local/bin/mediator_phase3 >> /tmp/mediator.log 2>&1 &
```

Log: `/tmp/mediator.log`. Sockets we saw: `/tmp/vgpu-mediator.sock`, `/var/vgpu/admin.sock`.

Until a VM runs with **qemu-dm** and **`-device vgpu-cuda,...`**, discovery will log something like “no vgpu-cuda QEMU process found” — that’s expected. Spin up a VM with the device attached (same XAPI / `vgpu-admin` workflow as on the first host), then watch the log again.

### If `vgpu-admin` looks broken

The CLI does two different things; failures look different:

1. **SQLite (`/etc/vgpu/vgpu_config.db`)** — Used for `status`, `list-vms`, `register-vm`, etc. If the DB path is missing or unwritable, you get **“Failed to initialize database”** or a SQLite error. Fix: `install -d /etc/vgpu` and `sqlite3 /etc/vgpu/vgpu_config.db < /etc/vgpu/init_db.sql` (or from `/root/phase3/schema/init_db.sql` before `make install` copies it).

2. **Unix socket to the mediator** — `show-metrics`, `show-health`, `show-connections`, `reload-config`, and **notifying the mediator after `set-weight` / quarantine changes** use **`/var/vgpu/admin.sock`**, which **only exists while `mediator_phase3` is running**. If the mediator is stopped, you’ll see **“Cannot connect to mediator at /var/vgpu/admin.sock”** and **“Is the mediator daemon running?”** — start it with the `nohup … mediator_phase3` snippet above, then re-check `ls -l /var/vgpu/admin.sock`.

3. **`xe` (XAPI)** — Name-based commands and the **“Unregistered VMs”** section need **`xe`** on **pool master** dom0 as **root**. On a pool slave, `xe` often fails silently in older builds; you’ll see empty VM lists or misleading **“No VMs found”**. Run the same command on the master, or use **`--vm-uuid=`** when you already know the UUID.

Quick checks (on this dom0):

```bash
test -S /var/vgpu/admin.sock && echo "admin sock OK" || echo "admin sock missing — start mediator_phase3"
test -r /etc/vgpu/vgpu_config.db && echo "db OK" || echo "db missing — run init_db.sql"
xe host-list params=name-label --minimal | head -1
```

After code fixes in the tree, rebuild and reinstall: `CUDA_PATH=/usr/local/cuda-12.2 make host && make install`.

---

## Workstation ↔ dom0 config

`phase3/vm_config.py` defaults the first host’s IP. For this dom0, from a laptop use `export MEDIATOR_HOST=10.25.33.20` or the bundled `vm_config_server2.py` pattern in this folder.

---

## After this doc

Guest side is unchanged: deploy shims into the VM, `LD_LIBRARY_PATH`, Ollama wrapper / `vgpu.conf` — same as test-4 on the original pool. This host only replaces **which dom0** runs QEMU + mediator.

If you reboot dom0 after big `yum` rounds, restart `mediator_phase3` and re-check `tail /tmp/mediator.log`.
