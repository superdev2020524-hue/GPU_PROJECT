# Host 2 — paste-ready commands & snippets

Replace **`10.25.33.20`** / **`root-fgmbpy`** if your environment differs.

---

## 1. SSH (workstation)

```bash
ssh root@10.25.33.20
```

---

## 2. Diagnostics (run on dom0 after SSH)

```bash
uname -r
cat /proc/cmdline
```

```bash
lspci -nn | grep -iE 'vga|3d|nvidia|matrox'
```

```bash
ls -la /dev/fb0 /dev/dri/card0 2>&1
ls /sys/class/drm/
```

```bash
cat /sys/class/tty/console/active
```

```bash
zcat /proc/config.gz 2>/dev/null | grep -iE 'DRM_MGAG200|FB_MATROX'
```

```bash
lsmod | grep -iE 'nvidia|drm'
```

```bash
cat /sys/module/nvidia_drm/parameters/modeset 2>/dev/null || true
```

---

## 3. GRUB — **Linux** line only (`module2` … `vmlinuz` …)

**Minimal (text on tty0, no splash):**

```text
module2 /boot/vmlinuz-4.19-xen root=LABEL=root-fgmbpy ro nolvm hpet=disable console=tty0 console=hvc0
```

**+ try NVIDIA KMS off (iDRAC KVM test):**

```text
module2 /boot/vmlinuz-4.19-xen root=LABEL=root-fgmbpy ro nolvm hpet=disable console=tty0 console=hvc0 nvidia-drm.modeset=0
```

**+ broader test (may affect GPU — maintenance window):**

```text
module2 /boot/vmlinuz-4.19-xen root=LABEL=root-fgmbpy ro nolvm hpet=disable console=tty0 console=hvc0 nomodeset
```

**Xen line example (keep your `LABEL` / memory):**

```text
multiboot2 /boot/xen.gz dom0_mem=8192M,max:8192M watchdog ucode=scan dom0_max_vcpus=1-16 crashkernel=256M,below=4G console=vga iommu=pt dom0=pvh
```

```text
module2 /boot/initrd-4.19-xen.img
```

**Typo fixes (must be exact):** `console=vga` not `consol=vga`; `quiet` not `quit`.

---

## 4. XCP-ng `xen-cmdline` (preferred over hand-editing EFI forever)

```bash
/opt/xensource/libexec/xen-cmdline --list 2>/dev/null
```

Forum / docs: [Add kernel boot params for dom0](https://xcp-ng.org/forum/topic/8092/add-kernel-boot-params-for-dom0)

---

## 5. `mgag200` (will **fail** on stock XCP-ng 4.19 dom0 — expected)

```bash
modprobe mgag200
```

If you see **Module mgag200 not found**, the kernel has **`CONFIG_DRM_MGAG200` not set** — see **`server2/HOST2_IDRAC_KVM_BLACK.md`**.

---

## 6. XSConsole over SSH

```bash
xsconsole
```

---

## 7. Management IP via `xe` (shell on dom0)

```bash
xe pif-list management=true params=uuid,device,IP,netmask,gateway --minimal
```

**Static example (replace UUID and IPs):**

```bash
PIF='<uuid-from-above>'
xe pif-reconfigure-ip uuid=$PIF mode=static IP=10.25.33.20 netmask=255.255.255.0 gateway=10.25.33.254 DNS=8.8.8.8,8.8.4.4
```

**DHCP:**

```bash
xe pif-reconfigure-ip uuid=$PIF mode=dhcp
```

---

## 8. Toolstack / HTTPS recovery (orchestration)

```bash
rpm -qa | grep -E '^xapi|^xcp-networkd|^xapi-xe|^xapi-rrd' | sort
```

```bash
ss -tlnp | grep -E ':80|:443'
ls -la /var/run/xapi_startup.cookie
```

```bash
xe-toolstack-restart
```

See **`server2/XCP_TOOLSTACK_AND_HTTPS_FIX.md`**.

---

## 9. Phase 3 quick checks

```bash
test -S /var/vgpu/admin.sock && echo OK || echo missing
test -r /etc/vgpu/vgpu_config.db && echo OK || echo missing
pgrep -a mediator_phase3
```

```bash
vgpu-admin status
vgpu-admin show-health 2>&1 | head -15
```

---

## 10. `connect_host` from repo (workstation, `server2/phase3/`)

```bash
cd /path/to/gpu/server2/phase3
python3 connect_host.py 'hostname; cat /proc/cmdline'
```

---

*For iDRAC black screen root cause (no `mgag200` in kernel), see **`server2/HOST2_IDRAC_KVM_BLACK.md`**.
