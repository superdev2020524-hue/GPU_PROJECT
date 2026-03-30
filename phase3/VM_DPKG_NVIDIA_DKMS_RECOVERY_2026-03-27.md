# VM package recovery — NVIDIA DKMS + `dpkg` (2026-03-27)

**Context:** Running `sudo apt install systemd-coredump` triggered configuration of **pending** packages (`nvidia-dkms`, `linux-headers-*`, `cuda-drivers`, …). **`nvidia-dkms` failed to build**, leaving **`dpkg` half-configured** (**7 not fully installed or removed**).

**Session handoff:** After following the **fix** below, continue Phase 3 work (E3 tracing, **`RETURN`** journal correlation, **`core_pattern`** under **`/tmp`**) per **`PHASE3_RESUME_SESSION_2026-03-26.md`** and **`ERROR_TRACKING_STATUS.md`**. **Long-duration model loads** only with explicit approval (**`ASSISTANT_ROLE_AND_ANTICOUPLING.md` §5.4**).

---

## Root cause (from `make.log`)

```text
cc: error: unrecognized command-line option '-ftrivial-auto-var-init=zero'
```

- Linux **6.8** kernel build passes this flag when building out-of-tree modules.
- **GCC 11** (default on Ubuntu 22.04: `/usr/bin/gcc` → **11.4.0**) does **not** support that option.
- **GCC 12+** does. **`gcc-12`** was **not** installed on the VM at diagnosis.

This is a **toolchain vs kernel-headers** mismatch, not an Ollama/vGPU logic bug.

---

## Fix (recommended): install GCC 12 and rebuild DKMS

Run on the **VM** as a user with **`sudo`** (SSH or `connect_vm.py`).

### 1. Install compilers

```bash
sudo apt-get update
sudo apt-get install -y gcc-12 g++-12
```

(Optional but useful: set alternatives so `gcc` defaults to 12 — **only if** you accept system-wide default change.)

```bash
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 120 --slave /usr/bin/g++ g++ /usr/bin/g++-12
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110 --slave /usr/bin/g++ g++ /usr/bin/g++-11
sudo update-alternatives --set gcc /usr/bin/gcc-12
```

**Or** without changing defaults, pass **`CC`** / **`CXX`** only for DKMS (step 2).

### 2. Rebuild NVIDIA DKMS for the running kernel

Replace **`6.8.0-106-generic`** with **`$(uname -r)`** if different.

```bash
sudo dkms remove nvidia/595.45.04 --all 2>/dev/null || true
export CC=/usr/bin/gcc-12
export CXX=/usr/bin/g++-12
sudo -E dkms install nvidia/595.45.04 -k "$(uname -r)" -v
```

If you need **both** installed kernels (e.g. **6.8.0-101** and **6.8.0-106**), run **`dkms install`** once per **`-k`** kernel version, or let **`dkms autoinstall`** run after **`dpkg --configure -a`**.

### 3. Finish package configuration

```bash
sudo dpkg --configure -a
sudo apt-get install -f -y
```

### 4. Verify

```bash
dkms status | grep nvidia
dpkg -l | grep -E 'nvidia-dkms|linux-headers-6.8|cuda-drivers'
systemctl status systemd-coredump 2>/dev/null || true
```

---

## If DKMS still fails

1. **Capture the new error:**  
   `sudo tail -100 /var/lib/dkms/nvidia/595.45.04/build/make.log`

2. **Ensure headers match the kernel:**  
   `sudo apt-get install -y "linux-headers-$(uname -r)"`

3. **Build essentials:**  
   `sudo apt-get install -y build-essential dkms`

4. **Do not** blindly **`apt remove nvidia-*`** on a GPU guest without confirming your architecture (vGPU + mediated CUDA may still expect user-space NVIDIA packages). Prefer fixing the **compiler** first.

---

## Phase 3–specific note

- **`systemd-coredump`** was **already** the newest version; **`apt`** only tried to **finish** other half-installed packages.
- **Core files** can still use **`kernel.core_pattern=/tmp/core-%e-%p.%t`** (see **`ERROR_TRACKING_STATUS.md`**) if **`coredumpctl`** is unavailable until **`dpkg`** is healthy.

---

## Checklist after recovery

- [ ] `dpkg --configure -a` exits **0**
- [ ] `nvidia` line in `dkms status` shows **installed** for **`$(uname -r)`** (if you rely on guest NVIDIA driver)
- [ ] `sudo systemctl restart ollama` (or kill/restart per **`TEMPORARY_STAGE_E1.md`** if stuck)
- [ ] Re-run **Checkpoint A–C** before any long experiment (**`SYSTEMATIC_ERROR_TRACKING_PLAN.md`**)

---

*Saved 2026-03-27 — diagnostic: `cc: ... -ftrivial-auto-var-init=zero` with **GCC 11**.*

---

## Follow-up (2026-03-27) — recovery completed on VM

**What worked:** Installing **`gcc-12` / `g++-12`** allowed **`nvidia-dkms`** to build during **`apt-get install`** (both **6.8.0-101** and **6.8.0-106**).

**Avoid:** After a successful DKMS build, **do not** run **`sudo dkms remove nvidia/595.45.04 --all`** unless you intend to rebuild — it **deleted** the `.ko` files from **`/lib/modules/...`** for **both** kernels. A follow-up **`dkms install -k $(uname -r)`** only restored the **running** kernel until **`dkms install -k 6.8.0-101-generic`** was run again for the other boot entry.

**Verified state (tracing resume):**

- **`dkms status`:** **`nvidia/595.45.04`** **installed** for **6.8.0-101-generic** and **6.8.0-106-generic**.
- **`dpkg --configure -a`** / **`apt-get install -f`:** clean (no pending half-configured stack in follow-up).
- **`coredumpctl list`:** works (**“No coredumps found”** when empty).
- **`/proc/sys/kernel/core_pattern`:** **`/tmp/core-%e-%p.%t`** (Phase 3 fallback) still active.
- **Checkpoint A:** **`ollama` active**, **`/api/tags`** OK, **`inference compute`** **`compute=9.0`**.
- **Checkpoint C (host):** **`401312`/`INVALID_IMAGE`** count **0** in **`mediator.log`** tail sample.

**Next (E3 tracing):** correlate **`journalctl`** **`cublasGemmEx() CALLED` → `RETURN ok` → `SIGSEGV`** on next failure; **`coredumpctl info`** / **`/tmp/core-*`** + **`gdb`** if a dump appears.
