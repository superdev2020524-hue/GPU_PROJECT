# Second pool host — operational report (dom0 `10.25.33.20`)

**Scope.** This note summarizes bring-up and recovery work on the second XCP-ng dom0 used for Phase 3 vGPU / CUDA remoting (`server2/`, mirrored `phase3/` on-host under `/root/phase3`). It is written for handover, not marketing.

## Platform and GPU

The system is **XCP-ng 8.3**, hostname **`xcp-ng-sfgagrpq`**, management **10.25.33.20**. Unlike the first pool host, this machine exposes **on the order of 90+ GB** of GPU frame-buffer capacity to the stack—a different class of hardware than the earlier environment and something we had to account for when sizing mediator expectations and sanity checks.

The physical GPU initially behaved as if it were in a **hidden / not-fully-visible** mode for dom0 tooling. After **disabling the relevant GPU hiding path in GRUB** and rebooting, **NVIDIA settings and visibility lined up** well enough to proceed with host-side CUDA and driver configuration. (Exact kernel parameters live with the host’s GRUB history; the point for the report is that **boot policy and GPU visibility were coupled**—we did not treat “driver install” as independent of firmware/boot.)

## Boot target and recovery

Early on, the host was effectively coming up in a **serial-oriented XCP-ng profile**—what shows in the boot menu as **XCP-ng (serial)** rather than the **normal service-oriented default** (**XCP-ng**, systemd stack fully in charge). That choice is fine for deep debugging; it is the wrong default when you want a predictable toolstack and stable remote management. After changes and a few rough iterations, the platform **stopped responding as a managed host altogether** (no reliable API/UI path). Recovery required a **full restore of the XCP-ng management plane**—not a casual `xe-toolstack-restart`—before we could trust `xe`, the bundled web UI, or orchestration again. I am deliberately vague on the exact restore mechanics here because they will differ if this is ever repeated; what matters is we validated **pool membership, storage, and API** before touching Phase 3 binaries.

## Management stack, HTTPS, and orchestration

Separately from GPU work, we hit a **split toolstack**: **`xapi-core` had moved to the 26.1.x line while companion packages (`xcp-networkd`, `xapi-xe`, `xapi-rrd2csv`) lagged on 25.6.x**, which broke the internal RPC surface during startup. Symptom set: **nothing listening on port 80**, **`xapi_startup.cookie` never appeared**, **HTTPS via stunnel reset** because there was nothing behind it. DNS on the management path also could not resolve **`mirrors.xcp-ng.org`** until we pointed **`resolv.conf`** at a resolver that could answer for the public mirror names.

Resolution, documented in **`server2/XCP_TOOLSTACK_AND_HTTPS_FIX.md`**, was: **align those packages to the same 26.1.x family**, run **`xe-toolstack-restart`**, confirm **xapi on :80** and **stunnel on :443**, then re-verify **XO Lite / HTTPS** to **10.25.33.20**. That work is orthogonal to Phase 3 but **blocking** for any sane iteration on VMs and ISO workflows.

## Follow-up — orchestration down again (`yum` / toolstack)

The pattern recurred: **orchestration** (Xen Orchestra or the **bundled XO Lite / HTTPS** path to dom0) was **not usable** until the **management stack** was healthy. **XCP-ng already ships `yum`**; the fix is not “install yum” for its own sake but **`yum update`** (or targeted installs) to **keep `xapi-core`, `xcp-networkd`, `xapi-xe`, and related packages on the same release line**, plus **`/etc/resolv.conf`** able to reach **`mirrors.xcp-ng.org`** when you pull updates. After package alignment, **`xe-toolstack-restart`** on dom0 is the usual way to bring **port 80 / API / stunnel** back in line. Same reference: **`server2/XCP_TOOLSTACK_AND_HTTPS_FIX.md`**. Until that completes, **XO-driven workflows** in **`HOST2_NEW_VM_SIMPLE.md`** cannot be assumed to work even if SSH returns.

## Phase 3 on this host

Once dom0 was stable, we followed **`server2/HOST2_DOM0_BRINGUP.md`** (and the same logical steps in root **`phase3/`**): **CUDA** under `/usr/local/cuda-12.2`, **`make host` / `make install`** for **`mediator_phase3`** and **`vgpu-admin`**, **SQLite** under `/etc/vgpu`, **mediator** backgrounded with logs under `/tmp`, sockets **`/tmp/vgpu-mediator.sock`** and **`/var/vgpu/admin.sock`**. **QEMU** was rebuilt from the **xcp-ng-rpms/qemu** fork with **`vgpu-cuda`** in the device model; build volume used **`/mnt/cudawork`** (~1.8 TB) so **`rpmbuild` did not consume root**.

Verification from this side included **SSH automation** against **10.25.33.20** and checks that **`vgpu-admin status`**, **`show-health`**, and **`show-metrics`** responded while the mediator was running—i.e. the **admin path and DB path both worked**, not only the CLI binary on disk.

## Bottom line

Second server is now on a **consistent XCP-ng toolstack**, **reachable over HTTPS**, with **GPU visibility and NVIDIA configuration** aligned after **GRUB policy changes**, and **Phase 3 host components** built and exercised on hardware that exposes **much larger GPU memory** than the first host. Remaining work is ordinary pool hygiene (DNS policy, coordinated `yum` on the xapi family) and guest-side shim deployment per existing Phase 3 docs—not dom0 triage.

---

*References: `server2/XCP_TOOLSTACK_AND_HTTPS_FIX.md`, `server2/HOST2_DOM0_BRINGUP.md`, `server2/HOST2_NEW_VM_SIMPLE.md`; `phase3/` Makefile and host sources.*

