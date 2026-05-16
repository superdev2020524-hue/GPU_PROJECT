# SHMEM vs BAR1 — contiguity and performance

## Operational rule

From **`SESSION_RESUME_GUIDE`**: every load test must record whether the guest logged **`data_path=shmem`** or **`data_path=BAR1`**. **BAR1** is a fallback; throughput is often far below bare-metal GPU or proven **shmem**, so claims about “Phase 1 timing” must name the path.

## Why BAR1 was used

The guest selects BAR1 when **`setup_shmem()`** returns 0. Common stderr signatures:

- **`No contiguous GPA span >= N MB`** (`N` = guest **`SHMEM_MIN_SIZE`**, **4 MiB** in current tree when paired with a **4 MiB** `vgpu-stub` build) — **either** Linux/QEMU layout prevents **`find_contiguous_gpa_span()`** from finding enough contiguous **guest-physical** frames (**`VGPU_SHMEM_MIN_SIZE`** in **`include/vgpu_protocol.h`** must match the **built QEMU stub**), **or** (historically) the scanner mis-labeled all-failure sweeps (see **`pfn_hidden`** / fix in **`find_contiguous_gpa_span()`**).
- **`Cannot resolve GPA … reason=pfn_hidden`** — **`/proc/self/pagemap`** returned present pages with **PFN zeroed** (no **`CAP_SYS_ADMIN`** in the **effective** set for that process), or other visibility limits.
- **`mmap shmem … Resource temporarily unavailable`** — typically **`RLIMIT_MEMLOCK`** / capability; **`ollama.service.d_vgpu.conf`** already sets **`LimitMEMLOCK=infinity`** and **`CAP_IPC_LOCK`** when that drop-in is installed.

## Capabilities, `exec`, and dynamic linking (Ollama)

**`AmbientCapabilities=`** in **`ollama.service.d/vgpu.conf`** applies to the process **systemd starts**. The **ambient** set is **cleared on `execve`**. Ollama starts a **`runner`** via **`execve("/usr/local/bin/ollama.bin", …)`**, so the runner **must** regain **`CAP_SYS_ADMIN`** through **file capabilities** on **`ollama.bin`** (owned by **root** — capabilities are **ignored** on non-root-owned executables):

```bash
chown root:root /usr/local/bin/ollama.bin
chmod 0755 /usr/local/bin/ollama.bin
setcap cap_sys_admin,cap_ipc_lock+ep /usr/local/bin/ollama.bin
```

Binaries with file capabilities run in **secure mode**: the dynamic linker may **ignore `LD_LIBRARY_PATH`**. The mediated stack depends on **`/opt/vgpu/lib`** and Ollama’s CUDA dirs — install a **ldconfig** drop-in so those libraries resolve without **`LD_LIBRARY_PATH`**:

```bash
printf '%s\n' /opt/vgpu/lib /usr/local/lib/ollama/cuda_v12 /usr/local/lib/ollama | sudo tee /etc/ld.so.conf.d/ollama-vgpu.conf
sudo ldconfig
```

**Do not** use **`sudo -u ollama python …`** alone to probe pagemap PFNs: that drops ambient/file caps unless you match the service (e.g. **`systemd-run … AmbientCapabilities=…`**).

## Code change (guest `cuda_transport.c`)

To improve odds of meeting the **contiguous GPA** requirement (**4 MiB** minimum in current repo when guest and **`vgpu-stub`** are rebuilt together):

1. **Linux `memfd_create` + `MAP_SHARED`** is tried **before** anonymous **`MAP_PRIVATE`** (shared file-backed mappings often allocate cleaner runs for `pagemap` scans).
2. **`madvise(MADV_HUGEPAGE)`** when available, before **`memset`/`mlock`**, to encourage huge pages.
3. If the primary mapping fails **`find_contiguous_gpa_span()`**, a **`MAP_HUGETLB`** anonymous mapping is tried (**2 MiB**-aligned size; needs **`vm.nr_hugepages`**).

The backing **`memfd`** file descriptor is stored on the transport and **closed** on **`cuda_transport_destroy()`**.

After deploy, **`libvgpu-cuda.so`** under **`/opt/vgpu/lib`** should be **`ln -sf /opt/vgpu/lib/libvgpu-cuda.so.1`** (not **`/usr/lib64/...`**). **`deploy_to_test3.py`** copies the built **`.so.1`** to both **`/opt/vgpu/lib`** and **`/usr/lib64`** so either search path loads the same build.

If the guest reaches only **~4 MiB** contiguous GPA (common with **`MAP_HUGETLB`** on this VM), registration still fails until **`VGPU_SHMEM_MIN_SIZE`** / the **QEMU vgpu-stub** are rebuilt together (today’s live stub enforces **8 MiB** and returns **`INVALID_LENGTH`** for **4 MiB**). Reserve pages: **`sysctl vm.nr_hugepages`** (e.g. **256** for **2 MiB** pages).

## Verify after deploy

On the VM (same session as the run), grep transport diagnostics. **Preferred:** capture the **`Connected`** line and any **`poll … from=`** lines using a **narrow** wall-clock window (see below). A broad boot scan is optional:

Also capture the **post-connect** line emitted by **`cuda_transport.c`** (narrow **`journalctl --since`/`--until`** around service start or first runner attach):

```bash
journalctl -u ollama --since 'YYYY-MM-DD HH:MM:SS' --until 'YYYY-MM-DD HH:MM:SS' --no-pager \
  | grep -E 'Connected \\(vm_id|data_path='
```

Example (**VM-6 §8 restart window, May 2026):** **`data_path=shmem status_from=BAR1`** (bulk **shmem**, status from **BAR1** mirror) can precede a later **`data_path=BAR1`** reconnect in the same boot — record **both** for load tests (**`ERROR_TRACKING_STATUS.md`**).

```bash
journalctl -u ollama -b --no-pager | grep -E 'data_path=|Exhausted shmem|Shared-memory registered|using BAR1'
```

For **`connect_vm.py`** (**bounded SSH command time**), prefer **`--since`/`--until`** (or **`-S '10 min ago'`**) instead of **`-b`** or very wide **`-S`** windows — wide scans can exceed **`CONNECT_VM_COMMAND_TIMEOUT_SEC`** with no useful output.

Expect **`Connected (vm_id=…)`** with **`data_path=shmem`** when shared-memory bulk registration succeeded; **`status_from=BAR1`** only means the status mirror uses **BAR1**, not that bulk is BAR1-only. Reconnects in the same boot may log **`data_path=BAR1`** later — record the sequence (**`ERROR_TRACKING_STATUS.md`**).

**Mar 29 `Test-4` (VM-6):** guest **`from=BAR1`** / mixed **`data_path=shmem`** status plane does **not** block **Checkpoint D** when the operator follows **`PHASE3_NO_HTTP_TIMEOUT_STRATEGY.md` §8** (restart → §1 preload → **`/api/ps`** → **`num_gpu:0`** CPU prime → archival **`Test-4`** with **`curl -m` ~185). Literal **single-request** default-GPU cold **`load_duration` ~7.46 s** **`vm=9`** micro-parity remains this doc’s **shmem/GPA/stub** track.

## Documentation obligation

Per **`SYSTEMATIC_ERROR_TRACKING_PLAN.md`** §3 and **`ASSISTANT_ROLE_AND_ANTICOUPLING.md`** §5.2: paste **transport path** (shmem/BAR1) into **`ERROR_TRACKING_STATUS.md`** for any session that exercises load, alongside checkpoints **A–C**.
