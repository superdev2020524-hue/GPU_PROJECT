# Host 2 — iDRAC virtual console black (verified on dom0)

**Host:** `10.25.33.20` · **Kernel:** `4.19.0+1` (XCP-ng dom0)

## Verified facts (live)

| Item | Result |
|------|--------|
| **`modprobe mgag200`** | **FATAL: Module mgag200 not found** |
| **`/proc/config.gz`** | **`# CONFIG_DRM_MGAG200 is not set`** |
| **`/proc/config.gz`** | **`# CONFIG_FB_MATROX is not set`** |
| **PCI 62:00.0** | Matrox G200eW3 (typical **iDRAC / embedded** video tap) |
| **PCI 81:00.0** | NVIDIA H100 — **`nvidia_drm`**, **`/dev/dri/card0`** only |
| **`/dev/fb0`** | **Absent** |

**Conclusion:** The stock **XCP-ng Xen kernel does not ship Matrox (`mgag200`) or legacy `matrox` framebuffer**. You **cannot** “fix” the missing module with `modprobe` on this build. **iDRAC** watches the **embedded** path; **DRM** on dom0 is dominated by **NVIDIA** — that mismatch matches **black KVM + working SSH**.

## What to try next (no `mgag200`)

1. **NVIDIA KMS off** (test one boot), on the **Linux** `module2` line add:
   ```text
   nvidia-drm.modeset=0
   ```
   Or kernel-style:
   ```text
   nvidia_drm.modeset=0
   ```
   Confirm with: `cat /sys/module/nvidia_drm/parameters/modeset` after boot.

2. **Broader test:** add **`nomodeset`** on the **Linux** line (may affect GPU/CUDA — maintenance window only).

3. **VESA EFI framebuffer:** kernel has **`CONFIG_FB_VESA=y`** / **`CONFIG_FB_EFI=y`** — **`vesafb`** may be **built-in** (no `.ko`). Parameters like **`video=efifb:off`** or **`vga=normal`** are trial-and-error for **BMC capture**; there is no Dell-agnostic recipe.

4. **Real fix for Matrox + iDRAC:** only a **kernel build with `CONFIG_DRM_MGAG200=m`** (or Dell/XCP-ng **supported** path) — **out of scope** for a quick `modprobe`.

5. **Operational workaround:** **SSH** + **`xsconsole`**; **iDRAC SOL** (serial) for pre-boot; accept **black HTML5 KVM** until NVIDIA/Matrox routing is sorted.

## Do not do

- Do not expect **`mgag200`** on this kernel without **changing the kernel config** or **using a different kernel package**.

---

*Verified via SSH to dom0 2026-03-25.*
